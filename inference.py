"""Hidden-state extraction for one language.

With a token budget N: count every text's tokens (tokenizer only), pick random whole texts until they hold
>= N tokens, run the model on just those texts (so every token keeps its full sentence context), then pick
exactly N of their tokens, the same N at every layer. Without a budget: every token of every text."""
import logging
import math

import numpy as np
import torch

from model_loader import ModelWrapper

# Child of run.py's "isotropy" logger, so --debug there also turns on these batch lines
log = logging.getLogger("isotropy.inference")


class HiddenStateExtractor:
    def __init__(self, model: ModelWrapper, batch_size: int = 32):
        """Wrap a loaded model for hidden-state extraction.

        Args:
            model: loaded ModelWrapper.
            batch_size: texts (rows) per forward pass.
        """
        self.model = model
        self.batch_size = batch_size
        # Special token ids (<bos>, <eos>, <pad>, ...) as a tensor, for torch.isin lookups
        self._special = torch.tensor(sorted(model.special_ids), dtype=torch.long)

    def _encode(self, texts: list[str], **kwargs):
        """Tokenize like the model sees it: truncation only at the model's own limit."""
        return self.model.tokenizer(texts, truncation=True, max_length=self.model.max_length, **kwargs)

    def count_tokens(self, texts: list[str]) -> np.ndarray:
        """Number of tokens extract() would keep for each text (tokenizer only, no forward pass).

        Args:
            texts: texts of one language.
        Returns:
            [len(texts)] int array of non-special token counts.
        """
        special = self.model.special_ids
        return np.array([sum(i not in special for i in ids) for ids in self._encode(texts)["input_ids"]],
                        dtype=np.int64)

    # No gradients / autograd bookkeeping: faster and uses less memory
    @torch.inference_mode()
    def extract(self, texts: list[str]) -> list[torch.Tensor]:
        """Run every text through the model and collect the hidden states of all its non-special tokens.

        Args:
            texts: texts of one language; each row is one model input.
        Returns:
            One [n_tokens, d] tensor per layer (index 0 = embedding output), on the CPU in the model's
            dtype. Rows follow the texts in order, and token order within each text.
        """
        tok, dev = self.model.tokenizer, self.model.device
        n_batches = math.ceil(len(texts) / self.batch_size)
        layers = None
        for b in range(n_batches):
            batch = texts[b * self.batch_size:(b + 1) * self.batch_size]
            # Batch encode: pad to the longest text of the batch. Right padding keeps the positions of
            # real tokens the same as in an unbatched run; truncation only at the model's own limit
            enc = self._encode(batch, padding=True, padding_side="right", return_tensors="pt")
            ids, att = enc["input_ids"], enc["attention_mask"]
            # Tokens whose vectors are kept: real (not padding) and not special
            keep = att.bool() & ~torch.isin(ids, self._special)
            # Forward pass: a tuple with one [B, T, d] tensor per layer
            hs = self.model.hidden_states(ids.to(dev), att.to(dev))
            if layers is None:
                layers = [[] for _ in hs]
            keep_dev = keep.to(dev)
            # h[keep] -> [k, d]: the kept tokens of all rows, row by row
            for layer, h in enumerate(hs):
                layers[layer].append(h[keep_dev].cpu())
            if b == 0 and log.isEnabledFor(logging.DEBUG):
                # Batch decode the kept tokens: shows exactly which tokens' vectors are collected
                log.debug("kept tokens of batch 1: %s", tok.batch_decode([i[k].tolist() for i, k in zip(ids, keep)]))
            log.debug("batch %d/%d: %d texts x %d tokens, %d kept", b + 1, n_batches, len(batch), ids.shape[1],
                      int(keep.sum()))
        return [torch.cat(h) for h in layers]

    def sample(self, layers: list[torch.Tensor], n: int | None, rng: np.random.Generator) -> torch.Tensor:
        """Pick n tokens uniformly at random (the same tokens at every layer), or all of them if n is None.

        Args:
            layers: output of extract().
            n: number of tokens to keep (must be <= the number of collected tokens); None = keep all.
            rng: seeded NumPy generator (same seed -> same tokens).
        Returns:
            points [L+1, n, d] float32 on the model's device (n = all collected tokens if n is None).
        Raises FloatingPointError if any selected hidden state is inf/NaN.
        """
        if n is None:
            points = torch.stack(layers).to(self.model.device).float()
        else:
            idx = torch.from_numpy(np.sort(rng.choice(len(layers[0]), size=n, replace=False)))
            points = torch.stack([h[idx] for h in layers]).to(self.model.device).float()
        # No inf/NaN (e.g. from fp16 overflow)
        if not torch.isfinite(points).all():
            raise FloatingPointError("non-finite hidden states; try --dtype float32")
        return points


def pick_texts(counts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Pick random whole texts until they hold at least n tokens.

    Args:
        counts: tokens per text (from count_tokens); must sum to >= n.
        n: token budget.
        rng: seeded NumPy generator (same seed -> same texts).
    Returns:
        Sorted indices of the picked texts (dataset order). Their tokens sum to >= n, and dropping the
        last text drawn would leave fewer than n.
    """
    order = rng.permutation(len(counts))
    # First position in the shuffled order where the running total reaches n
    stop = int(np.searchsorted(np.cumsum(counts[order]), n)) + 1
    return np.sort(order[:stop])
