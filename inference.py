"""Hidden-state extraction: tokenize a corpus, sample N real tokens uniformly, and gather their hidden
states at every layer into one preallocated [L+1, N, d] buffer."""
import logging
from dataclasses import dataclass

import numpy as np
import torch

from model_loader import ModelWrapper

# Child of run.py's "isotropy" logger, so --debug there also turns on these batch lines
log = logging.getLogger("isotropy.inference")


# Tokenized texts of one language, plus which tokens may be sampled
@dataclass
class TokenizedCorpus:
    ids: list[np.ndarray]    # token ids per text (with special tokens, truncated)
    real: list[np.ndarray]   # bool per token: eligible for sampling
    n_words: int
    n_nonspecial: int        # all non-special tokens (for fertility), incl. a skipped first token

    # @property: used like an attribute (corpus.n_real), computed on access
    @property
    def n_real(self) -> int:
        """Return the total number of sampleable tokens over all texts."""
        # Total number of sampleable tokens over all texts
        return int(sum(r.sum() for r in self.real))


class HiddenStateExtractor:
    def __init__(self, model: ModelWrapper, max_length: int = 512, max_tokens_per_batch: int = 16_384,
                 skip_first_token: bool = True):
        """Wrap a loaded model for tokenization and hidden-state extraction.

        Args:
            model: loaded ModelWrapper.
            max_length: max tokens per text (capped at the model's limit); longer texts are truncated.
            max_tokens_per_batch: max padded tokens (rows x longest length) per forward pass.
            skip_first_token: exclude the first non-special token of every text. Models without a BOS
                token (Qwen3-Embedding, Harrier) turn it into an attention sink whose norm is 100-500x the
                others, which would dominate every covariance-based metric.
        """
        self.model = model
        # Never exceed what the model itself supports
        self.max_length = min(max_length, model.max_length)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.skip_first_token = skip_first_token
        # Special token ids as an array, for fast np.isin lookups
        self._special = np.array(sorted(model.special_ids))

    def tokenize(self, texts: list[str]) -> TokenizedCorpus:
        """Tokenize texts and mark which tokens may be sampled.

        Args:
            texts: texts of one language.
        Returns:
            TokenizedCorpus with token ids, eligibility masks (not special, not skipped first token),
            word count and non-special token count.
        """
        # Tokenize all texts at once; add <bos>/<eos> as the model normally would; cut at max_length
        enc = self.model.tokenizer(texts, add_special_tokens=True, truncation=True, max_length=self.max_length)
        ids = [np.asarray(x, dtype=np.int64) for x in enc["input_ids"]]
        real, n_nonspecial = [], 0
        for x in ids:
            # True for normal tokens, False for special ones (~ = logical NOT)
            r = ~np.isin(x, self._special)
            n_nonspecial += int(r.sum())
            # Exclude the first normal token (np.argmax on bools = index of the first True)
            if self.skip_first_token and r.any():
                r[np.argmax(r)] = False
            real.append(r)
        log.debug("tokenized %d texts, %d non-special tokens", len(ids), n_nonspecial)
        # Words counted by whitespace split, for fertility (tokens per word)
        return TokenizedCorpus(ids, real, sum(len(t.split()) for t in texts), n_nonspecial)

    def _sample(self, corpus: TokenizedCorpus, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Choose n eligible tokens uniformly at random over the whole corpus.

        Args:
            corpus: tokenized texts.
            n: number of tokens to sample (must be <= corpus.n_real).
            rng: seeded NumPy generator.
        Returns:
            One bool mask per text over its positions; True = sampled.
        """
        # Number all eligible tokens of the corpus 0..total-1; offsets[s] = first number of text s
        counts = np.array([r.sum() for r in corpus.real])
        offsets = np.concatenate([[0], np.cumsum(counts)])
        # Pick n distinct token numbers uniformly at random
        chosen = np.zeros(offsets[-1], dtype=bool)
        chosen[rng.choice(offsets[-1], size=n, replace=False)] = True
        # Map the chosen numbers back to positions inside each text
        sel = []
        for s, r in enumerate(corpus.real):
            m = np.zeros(len(r), dtype=bool)
            # flatnonzero(r) = positions of eligible tokens in text s; keep the chosen ones
            m[np.flatnonzero(r)[chosen[offsets[s]:offsets[s + 1]]]] = True
            sel.append(m)
        return sel

    def _batches(self, order: list[int], lengths: list[int]):
        """Split a length-sorted list of texts into batches within the padded-token budget.

        Args:
            order: text indices sorted by length (ascending).
            lengths: token count of every text.
        Returns:
            Iterator of lists of text indices.
        """
        batch = []
        for s in order:
            # Padded batch size = rows x longest length; start a new batch when over budget
            if batch and (len(batch) + 1) * lengths[s] > self.max_tokens_per_batch:
                yield batch
                batch = []
            batch.append(s)
        if batch:
            yield batch

    # No gradients / autograd bookkeeping: faster and uses less memory
    @torch.inference_mode()
    def extract(self, corpus: TokenizedCorpus, n: int, rng: np.random.Generator) -> torch.Tensor:
        """Sample n tokens and collect their hidden states at every layer.

        Args:
            corpus: tokenized texts of one language.
            n: number of tokens to sample.
            rng: seeded NumPy generator (same seed -> same tokens).
        Returns:
            points [L+1, n, d] float32 on the model's device; index 0 = embedding output.
        Raises FloatingPointError if any hidden state is inf/NaN.
        """
        # Decide which tokens to keep before running the model
        sel = self._sample(corpus, n, rng)
        lengths = [len(x) for x in corpus.ids]
        # Texts without a sampled token never reach the model.
        # Sorting by length puts similar lengths together, so little padding is wasted
        order = sorted((s for s in range(len(sel)) if sel[s].any()), key=lambda s: lengths[s])
        dev = self.model.device
        # Output buffers are created after the first batch, once layers and d are known
        points = None
        filled = 0
        # Materialized so the progress lines can show "batch i/total"
        batches = list(self._batches(order, lengths))
        log.debug("forward pass: %d texts in %d batches", len(order), len(batches))
        for b, batch in enumerate(batches, 1):
            T = lengths[batch[-1]]                       # sorted ascending: last is longest
            # inp = token ids (pad elsewhere), att = 1 on real positions, msk = sampled positions
            inp = torch.full((len(batch), T), self.model.pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), T), dtype=torch.long)
            msk = torch.zeros((len(batch), T), dtype=torch.bool)
            for row, s in enumerate(batch):              # right padding for every model
                L = lengths[s]
                inp[row, :L] = torch.from_numpy(corpus.ids[s])
                att[row, :L] = 1
                msk[row, :L] = torch.from_numpy(sel[s])
            # Forward pass: a tuple with one [B, T, d] tensor per layer
            hs = self.model.hidden_states(inp.to(dev), att.to(dev))
            if points is None:
                points = torch.empty((len(hs), n, hs[0].shape[-1]), dtype=torch.float32, device=dev)
            # k = sampled tokens in this batch
            k = int(msk.sum())
            msk = msk.to(dev)
            # h[msk] picks only the sampled tokens -> [k, d]; write them into the next free slots
            for layer, h in enumerate(hs):
                points[layer, filled:filled + k] = h[msk].float()
            filled += k
            log.debug("batch %d/%d: %d texts x %d tokens, %d sampled (%d/%d total)",
                      b, len(batches), len(batch), T, k, filled, n)
        # Sanity checks: exactly n tokens collected, and no inf/NaN (e.g. from fp16 overflow)
        assert filled == n, (filled, n)
        if not torch.isfinite(points).all():
            raise FloatingPointError("non-finite hidden states; try --dtype float32")
        return points
