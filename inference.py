"""Hidden-state extraction: tokenize a corpus, sample N real tokens uniformly, and gather their hidden
states at every layer into one preallocated [L+1, N, d] buffer."""
from dataclasses import dataclass

import numpy as np
import torch

from model_loader import ModelWrapper


@dataclass
class TokenizedCorpus:
    ids: list[np.ndarray]    # token ids per text (with special tokens, truncated)
    real: list[np.ndarray]   # bool per token: eligible for sampling
    n_words: int
    n_nonspecial: int        # all non-special tokens (for fertility), incl. a skipped first token

    @property
    def n_real(self) -> int:
        return int(sum(r.sum() for r in self.real))


@dataclass
class LayerPoints:
    points: torch.Tensor     # [L+1, N, d] float32; index 0 = embedding output
    text_ids: torch.Tensor   # [N] index of the text each token came from


class HiddenStateExtractor:
    def __init__(self, model: ModelWrapper, max_length: int = 512, max_tokens_per_batch: int = 16_384,
                 skip_first_token: bool = True):
        """skip_first_token: exclude the first non-special token of every text. Models without a BOS
        token (Qwen3-Embedding, Harrier) turn it into an attention sink whose norm is 100-500x the
        others, which would dominate every covariance-based metric."""
        self.model = model
        self.max_length = min(max_length, model.max_length)
        self.max_tokens_per_batch = max_tokens_per_batch
        self.skip_first_token = skip_first_token
        self._special = np.array(sorted(model.special_ids))

    def tokenize(self, texts: list[str]) -> TokenizedCorpus:
        enc = self.model.tokenizer(texts, add_special_tokens=True, truncation=True, max_length=self.max_length)
        ids = [np.asarray(x, dtype=np.int64) for x in enc["input_ids"]]
        real, n_nonspecial = [], 0
        for x in ids:
            r = ~np.isin(x, self._special)
            n_nonspecial += int(r.sum())
            if self.skip_first_token and r.any():
                r[np.argmax(r)] = False
            real.append(r)
        return TokenizedCorpus(ids, real, sum(len(t.split()) for t in texts), n_nonspecial)

    def _sample(self, corpus: TokenizedCorpus, n: int, rng: np.random.Generator) -> list[np.ndarray]:
        """Per-text bool masks over positions, selecting n real tokens uniformly over the corpus."""
        counts = np.array([r.sum() for r in corpus.real])
        offsets = np.concatenate([[0], np.cumsum(counts)])
        chosen = np.zeros(offsets[-1], dtype=bool)
        chosen[rng.choice(offsets[-1], size=n, replace=False)] = True
        sel = []
        for s, r in enumerate(corpus.real):
            m = np.zeros(len(r), dtype=bool)
            m[np.flatnonzero(r)[chosen[offsets[s]:offsets[s + 1]]]] = True
            sel.append(m)
        return sel

    def _batches(self, order: list[int], lengths: list[int]):
        """Consecutive groups of a length-sorted order, each within the padded-token budget."""
        batch = []
        for s in order:
            if batch and (len(batch) + 1) * lengths[s] > self.max_tokens_per_batch:
                yield batch
                batch = []
            batch.append(s)
        if batch:
            yield batch

    @torch.inference_mode()
    def extract(self, corpus: TokenizedCorpus, n: int, rng: np.random.Generator) -> LayerPoints:
        sel = self._sample(corpus, n, rng)
        lengths = [len(x) for x in corpus.ids]
        # Texts without a sampled token never reach the model.
        order = sorted((s for s in range(len(sel)) if sel[s].any()), key=lambda s: lengths[s])
        dev = self.model.device
        points = text_ids = None
        filled = 0
        for batch in self._batches(order, lengths):
            T = lengths[batch[-1]]                       # sorted ascending: last is longest
            inp = torch.full((len(batch), T), self.model.pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), T), dtype=torch.long)
            msk = torch.zeros((len(batch), T), dtype=torch.bool)
            for row, s in enumerate(batch):              # right padding for every model
                L = lengths[s]
                inp[row, :L] = torch.from_numpy(corpus.ids[s])
                att[row, :L] = 1
                msk[row, :L] = torch.from_numpy(sel[s])
            hs = self.model.hidden_states(inp.to(dev), att.to(dev))
            if points is None:
                points = torch.empty((len(hs), n, hs[0].shape[-1]), dtype=torch.float32, device=dev)
                text_ids = torch.empty(n, dtype=torch.long, device=dev)
            k = int(msk.sum())
            msk = msk.to(dev)
            for layer, h in enumerate(hs):
                points[layer, filled:filled + k] = h[msk].float()
            text_ids[filled:filled + k] = torch.tensor(batch, device=dev).repeat_interleave(msk.sum(1))
            filled += k
        assert filled == n, (filled, n)
        if not torch.isfinite(points).all():
            raise FloatingPointError("non-finite hidden states; try --dtype float32")
        return LayerPoints(points, text_ids)
