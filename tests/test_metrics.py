import numpy as np
import pytest
import torch
from IsoScore import IsoScore as official

from isotropy_metrics import build_metrics, compute_layer

ALL = build_metrics(["isoscore", "mev", "avgcos", "id"])


def run(X, text_ids=None):
    X = torch.as_tensor(X, dtype=torch.float32)
    if text_ids is None:
        text_ids = torch.arange(X.shape[0])
    return compute_layer(ALL, X, torch.as_tensor(text_ids))


def test_isoscore_matches_official_package():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 64)) * rng.uniform(0.1, 5, size=64) + rng.normal(size=64) * 3
    X = X @ np.linalg.qr(rng.normal(size=(64, 64)))[0]     # rotated, anisotropic, shifted
    ours = run(X)["isoscore"]
    ref = float(official.IsoScore(X.astype(np.float32).astype(np.float64)))
    assert ours == pytest.approx(ref, abs=1e-6)


def test_isotropic_gaussian():
    X = np.random.default_rng(0).normal(size=(50_000, 16))
    r = run(X)
    assert r["isoscore"] > 0.99
    assert r["mev"] == pytest.approx(1 / 16, rel=0.05)
    assert abs(r["avg_cos"]) < 0.01


def test_rank_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 1)) * rng.normal(size=(1, 32))
    r = run(X)
    assert r["isoscore"] < 0.01
    assert r["mev"] > 0.999


def test_intrinsic_dimension_of_linear_subspace():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 5)) @ rng.normal(size=(5, 64))
    r = run(X)
    assert 4.5 < r["id_mle"] < 5.5
    assert r["id_score"] == pytest.approx(r["id_mle"] / 64)


def test_intrinsic_dimension_ignores_exact_duplicates():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1000, 3)) @ rng.normal(size=(3, 32))
    r = run(np.concatenate([base, base[:500]]))   # layer-0-like: repeated token embeddings
    assert r["id_n_unique"] == 1000
    assert 2.5 < r["id_mle"] < 3.5


def test_avgcos_equals_brute_force_over_cross_text_pairs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 8)) + 2.0
    text_ids = rng.integers(0, 40, size=300)
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = U @ U.T
    cross = text_ids[:, None] != text_ids[None, :]
    assert run(X, text_ids)["avg_cos"] == pytest.approx(C[cross].mean(), abs=1e-6)
