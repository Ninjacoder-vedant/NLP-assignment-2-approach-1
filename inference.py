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
        # Tokenize every text at once; one list of token ids per text
        all_ids = self._encode(texts)["input_ids"]

        counts = []
        for ids in all_ids:
            # Count only real tokens; <bos>, <eos>, <pad>, ... are dropped by extract()
            count = 0
            for token_id in ids:
                if token_id not in special:
                    count += 1
            counts.append(count)

        return np.array(counts, dtype=np.int64)

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
        # Get the tokenizer and device
        tok, dev = self.model.tokenizer, self.model.device
        # Number of batches is ceil(n_texts / batch_size)
        n_batches = math.ceil(len(texts) / self.batch_size)
        # Number of the non-special tokens in all texts
        n_non_special_tokens = self.count_tokens(texts).sum()
        # If token_idx is given, the token view keeps only those tokens; otherwise it keeps all non-special tokens
        n_keep = len(token_idx) if token_idx is not None else n_non_special_tokens
        # Convert the token indices to a torch tensor for torch.isin lookups
        idx = None if token_idx is None else torch.as_tensor(token_idx, dtype=torch.long)
        out = None
        # off = global position of the batch's first token; w = token-view rows written so far
        off = w = 0
        # b = current batch index
        for b in range(n_batches):
            # r = idx of the first text in this batch
            r = b * self.batch_size
            batch = texts[r:r + self.batch_size]
            # Batch encode: pad to the longest text of the batch. Right padding keeps the positions of
            # real tokens the same as in an unbatched run; truncation only at the model's own limit
            enc = self._encode(batch, padding=True, padding_side="right", return_tensors="pt")
            ids, att = enc["input_ids"], enc["attention_mask"]

            # Only keep tokens whose attention is 1 (no pad token) and it is not special token
            # [B, T] bool: True = keep this token, False = drop it
            keep = att.bool() & ~torch.isin(ids, self._special)
            # per_text kept token counts: [B] int, 0 for a text that has no non-special token
            per_text = keep.sum(1)

            # Check that every text has at least one non-special token; otherwise the sentence views would be NaN
            if (per_text == 0).any():
                raise ValueError(f"text {r + int((per_text == 0).nonzero()[0])} has no non-special token")

            # ---- Which tokens of this batch go into the token view ----
            # Total kept tokens in this batch (all rows together)
            n_kept = int(per_text.sum())
            # Global positions of these kept tokens in the combined token matrix: off .. off+n_kept-1
            positions = torch.arange(off, off + n_kept)
            if idx is None:
                # No token_idx given: the token view takes every kept token
                pick = torch.ones(n_kept, dtype=torch.bool)
            else:
                # pick[j] = True if the j-th kept token's global position is one of the requested indices
                pick = torch.isin(positions, idx)
            # How many of this batch's tokens go into the token view
            n_picked = int(pick.sum())

            # ---- Forward pass ----
            # Move the inputs to the model's device
            ids_d = ids.to(dev)
            att_d = att.to(dev)
            # Tuple with one [B, T, d] tensor per layer (layer 0 = embedding output)
            hs = self.model.hidden_states(ids_d, att_d)

            # ---- Allocate the outputs (first batch only) ----
            if out is None:
                # Only now are the layer count and hidden size known
                n_layers = len(hs)
                # Layer 0's hidden size is the model's embedding size; all layers have the same hidden size
                d = hs[0].shape[-1]

                # Token view: one row per picked token, in the model's dtype to save memory
                # Sentence views: one row per text, float32
                out = {"token": torch.empty(n_layers, n_keep, d, dtype=hs[0].dtype),
                       "sentence-mean": torch.empty(n_layers, len(texts), d),
                       "sentence-last": torch.empty(n_layers, len(texts), d)}

            # Masks on the model's device, to index the hidden states there
            keep_d = keep.to(dev)
            pick_d = pick.to(dev)

            # ---- Weights for the sentence mean: 1/count at a kept token, 0 elsewhere ----
            # T = padded length of this batch (max(seq_len); B = number of texts in this batch
            # [B, T] mask as 0.0 / 1.0
            keep_float = keep_d.float()
            # [B, 1] kept-token count of each row, shaped so it divides its whole row
            row_counts = per_text.to(dev)[:, None]
            # [B, T] -> [B, 1, T]: extra middle dim so torch.bmm can multiply it with [B, T, d]
            weights = (keep_float / row_counts).unsqueeze(1)

            # ---- Position of each row's last kept token ----
            # [T] positions 0 .. T-1
            positions_in_row = torch.arange(keep.shape[1])
            # [B, T]: the position where the token is kept, 0 elsewhere
            kept_positions = keep * positions_in_row
            # [B]: the largest kept position = the last kept token of each row
            last = kept_positions.argmax(1).to(dev)
            # [B] row numbers 0 .. B-1, paired with last to take one token per row
            rows = torch.arange(len(batch), device=dev)

            # ---- Where this batch's results go in the outputs ----
            # Token view rows w .. w+n_picked-1
            token_rows = slice(w, w + n_picked)
            # Sentence view rows: this batch's texts r .. r+B-1
            text_rows = slice(r, r + len(batch))

            # h: [B, T, d] hidden states of one layer
            for layer, h in enumerate(hs):
                # Token view
                # [n_kept, d]: kept tokens of all rows, row by row (same order as the global positions)
                kept_vecs = h[keep_d]
                # [n_picked, d]: only the ones the token view takes
                picked_vecs = kept_vecs[pick_d]
                out["token"][layer, token_rows] = picked_vecs.cpu()

                # Sentence-mean
                # [B, 1, T] @ [B, T, d] -> [B, 1, d] -> [B, d]: weighted sum = mean of each row's kept tokens
                mean_vecs = torch.bmm(weights, h.float()).squeeze(1)
                out["sentence-mean"][layer, text_rows] = mean_vecs.cpu()

                # Sentence-last
                # [B, d]: for each row, the vector at its last kept position
                last_vecs = h[rows, last].float()
                out["sentence-last"][layer, text_rows] = last_vecs.cpu()

            # Move on: the next batch's tokens start after this batch's ones
            off += n_kept
            w += n_picked

            # ---- Debug logging ----
            if b == 0 and log.isEnabledFor(logging.DEBUG):
                # First batch only: decode each row's kept tokens, shows exactly which tokens' vectors are used
                kept_ids = []
                for row_ids, row_keep in zip(ids, keep):
                    # Token ids of this row where keep is True
                    kept_ids.append(row_ids[row_keep].tolist())
                log.debug("kept tokens of batch 1: %s", tok.batch_decode(kept_ids))
            # ids.shape[1] = padded length T of this batch
            log.debug("batch %d/%d: %d texts x %d tokens, %d kept, %d picked", b + 1, n_batches, len(batch),
                      ids.shape[1], n_kept, n_picked)

        # ---- Final checks ----
        # Fewer rows written than requested means some token_idx was beyond the last token
        if w != n_keep:
            raise RuntimeError(f"token view got {w} of {n_keep} tokens: token_idx beyond the {off} tokens")
        # No inf/NaN in any view (e.g. from fp16 overflow)
        for view in out.values():
            if not torch.isfinite(view).all():
                raise FloatingPointError("non-finite hidden states; try --dtype float32")
        return out
