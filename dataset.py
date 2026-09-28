import os
import re
import gc

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from IsoScore import IsoScore
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

# Retrieve the secret from Kaggle
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")

# Set it as an environment variable
os.environ["HF_TOKEN"] = HF_TOKEN

# ----------------------------- config -----------------------------
MODEL_ID = "google/embeddinggemma-300m"
DATASET_ID = "ai4bharat/IN22-Gen"
SPLIT = "test"                    # single config; one column per language code
N_TOKENS_CAP = 10_000             # tokens per language; >= 10 * d (d = 768) keeps IsoScore bias ~2%
BATCH_SIZE = 64
N_PAIRS = 200_000                 # random pairs for AvgCos
ID_K = 20                         # neighbours for the MLE intrinsic dimension
SEED = 0
OUT_DIR = "anisotropy_results"
SAVE_POINTS = False               # True = also save sampled vectors (float16 .npy) for later analysis

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED) 

device = "cuda" if torch.cuda.is_available() else "cpu"
# EmbeddingGemma does not support float16 activations: use bfloat16 on GPU if available, else float32
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32


# ----------------------------- data -----------------------------
ds = load_dataset(DATASET_ID, split=SPLIT)

# Language columns are FLORES-style codes like "hin_Deva": 3 lowercase letters, "_", 4-letter script.
# Metadata columns (context, source, url, domain, num_words, bucket) do not match this pattern.
LANG_CODE = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
langs = sorted(c for c in ds.column_names if LANG_CODE.match(c))

# Keep only rows that are non-empty in every language, so the corpus stays fully parallel
cols = {l: ds[l] for l in langs}
keep_rows = [i for i in range(len(ds))
             if all(isinstance(cols[l][i], str) and cols[l][i].strip() for l in langs)]
texts = {l: [cols[l][i].strip() for i in keep_rows] for l in langs}
print(f"{len(langs)} language columns, {len(keep_rows)} parallel sentences: {langs}")

# # ----------------------------- model -----------------------------
# model = SentenceTransformer(MODEL_ID, device=device, model_kwargs={"torch_dtype": dtype})
# backbone = model[0].auto_model.eval()
# tokenizer = model.tokenizer
# max_len = model.max_seq_length 
# d = backbone.config.hidden_size
# special_ids = set(tokenizer.all_special_ids)
# pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
# left_pad = tokenizer.padding_side == "left"


# # ------------------ pass 1: tokenize, count real tokens ------------------
# tok = {}
# stats = []
# for l in langs:
#     ids = tokenizer(texts[l], add_special_tokens=True, truncation=True,
#                     max_length=max_len)["input_ids"]
#     real = [np.array([t not in special_ids for t in s], dtype=bool) for s in ids]
#     n_real = int(sum(r.sum() for r in real))
#     n_words = sum(len(s.split()) for s in texts[l])
#     tok[l] = (ids, real)
#     stats.append({"lang": l, "real_tokens": n_real, "words": n_words,
#                   "fertility": n_real / max(n_words, 1)})

# stats_df = pd.DataFrame(stats).sort_values("fertility")
# stats_df.to_csv(f"{OUT_DIR}/token_stats.csv", index=False)
# print(stats_df.to_string(index=False))

# N = min(N_TOKENS_CAP, int(stats_df.real_tokens.min()))
# print(f"\nSampling N = {N} tokens per language (d = {d}, N/d = {N / d:.1f})")
# if N < 10 * d:
#     print(f"WARNING: N < 10*d. IsoScore will be biased (isotropic ceiling ~ {N / (N + d):.3f}). "
#           "Add IN22-Conv sentences to increase N.")


# # ----------------------------- metrics -----------------------------
# def mev_from_cov(X64: torch.Tensor) -> float:
#     ev = torch.linalg.eigvalsh(torch.cov(X64.T)).clamp_min(0)
#     return float(ev[-1] / ev.sum())


# def avg_cos(X: torch.Tensor, sent_ids: torch.Tensor, n_pairs: int) -> float:
#     Xn = torch.nn.functional.normalize(X, dim=1)
#     n = X.shape[0]
#     i = torch.randint(0, n, (int(n_pairs * 1.3),), device=X.device)
#     j = torch.randint(0, n, (int(n_pairs * 1.3),), device=X.device)
#     ok = sent_ids[i] != sent_ids[j]                 # different sentences only
#     i, j = i[ok][:n_pairs], j[ok][:n_pairs]
#     cos = (Xn[i] * Xn[j]).sum(dim=1)
#     return float(1 - cos.mean().abs())


# def mle_id(X: torch.Tensor, k: int = ID_K, chunk: int = 2048) -> float:
#     """Levina & Bickel (2004) MLE, averaged over points."""
#     knn = []
#     for s in range(0, X.shape[0], chunk):
#         dist = torch.cdist(X[s:s + chunk], X)
#         knn.append(dist.topk(k + 1, largest=False).values[:, 1:])   # drop self
#     T = torch.cat(knn).double().clamp_min(1e-10)                     # (n, k)
#     m = (k - 1) / torch.log(T[:, -1:] / T[:, :-1]).sum(dim=1)
#     m = m[torch.isfinite(m)]
#     return float(m.mean())


# # ------------------ pass 2: inference + metrics per language ------------------
# results = []
# n_layers_out = backbone.config.num_hidden_layers + 1   # embeddings + each transformer layer

# for l in tqdm(langs, desc="languages"):
#     ids, real = tok[l]

#     # Choose N real tokens uniformly at random across the whole language corpus
#     offsets = np.cumsum([0] + [int(r.sum()) for r in real])
#     chosen = np.zeros(offsets[-1], dtype=bool)
#     chosen[rng.choice(offsets[-1], size=N, replace=False)] = True
#     sel = []
#     for s, r in enumerate(real):
#         m = np.zeros(len(ids[s]), dtype=bool)
#         m[np.flatnonzero(r)[chosen[offsets[s]:offsets[s + 1]]]] = True
#         sel.append(m)

#     pts = torch.empty((n_layers_out, N, d), dtype=torch.float32, device=device)
#     sent_of_pt = torch.empty(N, dtype=torch.long, device=device)
#     filled = 0

#     order = sorted(range(len(ids)), key=lambda s: len(ids[s]))   # length bucketing
#     with torch.inference_mode():
#         for b in range(0, len(order), BATCH_SIZE):
#             batch = order[b:b + BATCH_SIZE]
#             T = max(len(ids[s]) for s in batch)
#             inp = torch.full((len(batch), T), pad_id, dtype=torch.long)
#             att = torch.zeros((len(batch), T), dtype=torch.long)
#             msk = torch.zeros((len(batch), T), dtype=torch.bool)
#             for row, s in enumerate(batch):
#                 L = len(ids[s])
#                 sl = slice(T - L, T) if left_pad else slice(0, L)
#                 inp[row, sl] = torch.tensor(ids[s])
#                 att[row, sl] = 1
#                 msk[row, sl] = torch.from_numpy(sel[s])
#             k = int(msk.sum())
#             if k == 0:
#                 continue
#             out = backbone(input_ids=inp.to(device), attention_mask=att.to(device),
#                            output_hidden_states=True)
#             msk_d = msk.to(device)
#             for layer, hs in enumerate(out.hidden_states):
#                 pts[layer, filled:filled + k] = hs[msk_d].float()
#             sent_of_pt[filled:filled + k] = torch.tensor(batch).repeat_interleave(msk.sum(1)).to(device)
#             filled += k

#     assert filled == N, (filled, N)

#     for layer in range(n_layers_out):
#         X = pts[layer]
#         X64 = X.double()
#         results.append({
#             "lang": l,
#             "layer": layer,
#             "n_tokens": N,
#             "isoscore": float(IsoScore.IsoScore(X64.cpu().numpy())),
#             "avgcos": avg_cos(X, sent_of_pt, N_PAIRS),
#             "id_score": mle_id(X) / d,
#             "mev": mev_from_cov(X64),
#         })

#     if SAVE_POINTS:
#         np.save(f"{OUT_DIR}/points_{l}.npy", pts.cpu().numpy().astype(np.float16))

#     pd.DataFrame(results).to_csv(f"{OUT_DIR}/metrics.csv", index=False)  # checkpoint
#     del pts, sent_of_pt
#     gc.collect()
#     if device == "cuda":
#         torch.cuda.empty_cache()

# df = pd.DataFrame(results)
# df.to_csv(f"{OUT_DIR}/metrics.csv", index=False)
# print("\nMean over layers:")
# print(df.groupby("lang")[["isoscore", "avgcos", "id_score", "mev"]].mean()
#         .sort_values("isoscore").to_string())