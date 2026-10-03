"""Hidden-state extraction for one language.

Every text runs through the model whole, so every token keeps its full sentence context. Each batch's
hidden states are reduced on the spot to three views, and the full token matrix is never kept:

    token          the tokens at the given global indices (the same ones at every layer), or all tokens.
                   Indices refer to the combined token matrix of all texts (text order, then token order),
                   whose size count_tokens() gives before any forward pass, so drawing them up front equals
                   sampling the full matrix after inference
    sentence-mean  mean of each text's tokens
    sentence-last  each text's last token

Pooling uses the same tokens as the token view: real (not padding) and not special."""
import logging
import math

import numpy as np
import torch

from model_loader import ModelWrapper

# Child of run.py's "isotropy" logger, so --debug there also turns on these batch lines
log = logging.getLogger("isotropy.inference")

# Names of the views extract() returns (also the results folder names)
VIEWS = ("token", "sentence-mean", "sentence-last")


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
        """Number of tokens extract() keeps for each text (tokenizer only, no forward pass).

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
    def extract(self, texts: list[str], token_idx: np.ndarray | None = None) -> dict[str, torch.Tensor]:
        """Run every text through the model and reduce each batch to the token and sentence views.

        Args:
            texts: texts of one language; each row is one model input and must keep >= 1 token.
            token_idx: sorted, unique positions in the combined token matrix of all texts to keep for the
                token view; None = keep every token.
        Returns:
            {"token": [L+1, N, d] in the model's dtype (N = len(token_idx) or all tokens),
             "sentence-mean": [L+1, n_texts, d] float32, "sentence-last": [L+1, n_texts, d] float32},
            all on the CPU; layer 0 = embedding output.
        Raises ValueError if a text keeps no token, FloatingPointError if any value is inf/NaN.
        """
        tok, dev = self.model.tokenizer, self.model.device
        n_batches = math.ceil(len(texts) / self.batch_size)
        n_keep = len(token_idx) if token_idx is not None else int(self.count_tokens(texts).sum())
        idx = None if token_idx is None else torch.as_tensor(token_idx, dtype=torch.long)
        out = None
        # off = global position of the batch's first token; w = token-view rows written so far
        off = w = 0
        for b in range(n_batches):
            r = b * self.batch_size
            batch = texts[r:r + self.batch_size]
            # Batch encode: pad to the longest text of the batch. Right padding keeps the positions of
            # real tokens the same as in an unbatched run; truncation only at the model's own limit
            enc = self._encode(batch, padding=True, padding_side="right", return_tensors="pt")
            ids, att = enc["input_ids"], enc["attention_mask"]
            # Tokens whose vectors are used: real (not padding) and not special
            keep = att.bool() & ~torch.isin(ids, self._special)
            per_text = keep.sum(1)
            if (per_text == 0).any():
                raise ValueError(f"text {r + int((per_text == 0).nonzero()[0])} has no non-special token")
            k = int(per_text.sum())
            # Which of this batch's k kept tokens (global positions off..off+k-1) the token view keeps
            pick = torch.ones(k, dtype=torch.bool) if idx is None else torch.isin(torch.arange(off, off + k), idx)
            m = int(pick.sum())
            # Forward pass: a tuple with one [B, T, d] tensor per layer
            hs = self.model.hidden_states(ids.to(dev), att.to(dev))
            if out is None:
                # Preallocate once the layer count and width are known: only the kept rows are ever stored
                L, d = len(hs), hs[0].shape[-1]
                out = {"token": torch.empty(L, n_keep, d, dtype=hs[0].dtype),
                       "sentence-mean": torch.empty(L, len(texts), d),
                       "sentence-last": torch.empty(L, len(texts), d)}
            keep_d, pick_d = keep.to(dev), pick.to(dev)
            # [B, 1, T] float weights for the masked mean (bmm sums the kept tokens of each row)
            weights = (keep_d.float() / per_text.to(dev)[:, None]).unsqueeze(1)
            rows = torch.arange(len(batch), device=dev)
            # Position of each row's last kept token
            last = (keep * torch.arange(keep.shape[1])).argmax(1).to(dev)
            for layer, h in enumerate(hs):
                # h[keep] -> [k, d]: the kept tokens of all rows, row by row; then the picked ones
                out["token"][layer, w:w + m] = h[keep_d][pick_d].cpu()
                out["sentence-mean"][layer, r:r + len(batch)] = torch.bmm(weights, h.float()).squeeze(1).cpu()
                out["sentence-last"][layer, r:r + len(batch)] = h[rows, last].float().cpu()
            off, w = off + k, w + m
            if b == 0 and log.isEnabledFor(logging.DEBUG):
                # Batch decode the kept tokens: shows exactly which tokens' vectors are used
                log.debug("kept tokens of batch 1: %s", tok.batch_decode([i[kp].tolist() for i, kp in zip(ids, keep)]))
            log.debug("batch %d/%d: %d texts x %d tokens, %d kept, %d picked", b + 1, n_batches, len(batch),
                      ids.shape[1], k, m)
        # Every requested position was seen: the indices really refer to this combined token matrix
        if w != n_keep:
            raise RuntimeError(f"token view got {w} of {n_keep} tokens: token_idx beyond the {off} tokens")
        # No inf/NaN (e.g. from fp16 overflow)
        if not all(torch.isfinite(v).all() for v in out.values()):
            raise FloatingPointError("non-finite hidden states; try --dtype float32")
        return out
