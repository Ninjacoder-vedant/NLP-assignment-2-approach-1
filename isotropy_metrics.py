"""Isotropy metrics on one layer's point cloud X [N, d] (all on GPU).

`LayerStats` caches quantities shared by several metrics (covariance eigenvalues serve both IsoScore
and MEV). To add a metric, subclass `IsotropyMetric`, implement `compute` (return a dict, so one metric
can emit several columns), and register it with `@METRICS.register("name")`.
"""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

import torch
import torch.nn.functional as F

from registry import METRICS


class LayerStats:
    def __init__(self, X: torch.Tensor, text_ids: torch.Tensor):
        self.X = X
        self.text_ids = text_ids
        self.n, self.d = X.shape

    @cached_property
    def eigvals(self) -> torch.Tensor:
        """Covariance eigenvalues, ascending, float64 (= the variances along the principal axes)."""
        return torch.linalg.eigvalsh(torch.cov(self.X.double().T)).clamp_min(0)

    @cached_property
    def X_unit(self) -> torch.Tensor:
        return F.normalize(self.X.double(), dim=1)


class IsotropyMetric(ABC):
    name: ClassVar[str]   # set by @METRICS.register

    @abstractmethod
    def compute(self, s: LayerStats) -> dict[str, float]:
        ...


@METRICS.register("isoscore")
class IsoScore(IsotropyMetric):
    """Rudman et al. (2022). Identical to `IsoScore.IsoScore(X)` from the official package, which uses
    the singular values of the covariance (= its eigenvalues, since it is PSD)."""

    def compute(self, s):
        pcs = s.eigvals
        n = torch.tensor(float(s.d), dtype=torch.float64)
        pcs_norm = pcs * n.sqrt() / torch.linalg.vector_norm(pcs)
        defect = torch.linalg.vector_norm(pcs_norm - 1) / torch.sqrt(2 * (n - n.sqrt()))
        score = ((n - defect ** 2 * (n - n.sqrt())) ** 2 - n) / (n * (n - 1))
        return {"isoscore": float(score)}


@METRICS.register("mev")
class MaxExplainableVariance(IsotropyMetric):
    """Fraction of variance explained by the first principal component (Ethayarajh, 2019)."""

    def compute(self, s):
        return {"mev": float(s.eigvals[-1] / s.eigvals.sum())}


@METRICS.register("avgcos")
class AvgRandomCosine(IsotropyMetric):
    """Mean cosine similarity between tokens of *different* texts (Ethayarajh, 2019).
    Computed exactly over all such pairs with sum vectors, O(N d), instead of sampling random pairs:
    sum_{i != j} cos = |S|^2 - N, and pairs within the same text are removed the same way per text."""

    def compute(self, s):
        U = s.X_unit
        per_text = torch.zeros((int(s.text_ids.max()) + 1, s.d), dtype=U.dtype, device=U.device)
        per_text.index_add_(0, s.text_ids, U)
        counts = torch.bincount(s.text_ids).double()
        total = U.sum(0)
        cross_sum = total @ total - (per_text * per_text).sum()
        cross_pairs = s.n ** 2 - (counts ** 2).sum()
        if cross_pairs == 0:
            return {"avg_cos": float("nan")}
        return {"avg_cos": float(cross_sum / cross_pairs)}


@METRICS.register("id")
class IntrinsicDimension(IsotropyMetric):
    """Levina & Bickel (2004) MLE of intrinsic dimension (k nearest neighbours), averaged over points.
    `id_score` = id_mle / d. Exact duplicate points (e.g. the same token id at layer 0, where there is
    no positional information) are removed first: zero distances make the estimator degenerate."""

    def __init__(self, k: int = 20, chunk: int = 2048):
        self.k, self.chunk = k, chunk

    def compute(self, s):
        X = torch.unique(s.X, dim=0)
        X = X - X.mean(0)                   # translation-invariant; improves fp32 cdist precision
        n, k = X.shape[0], self.k
        if n <= k:
            return {"id_mle": float("nan"), "id_score": float("nan"), "id_n_unique": n}
        knn = []
        for a in range(0, n, self.chunk):
            dist = torch.cdist(X[a:a + self.chunk], X)
            rows = torch.arange(dist.shape[0], device=X.device)
            dist[rows, rows + a] = float("inf")                 # exclude self
            knn.append(dist.topk(k, largest=False).values)
        T = torch.cat(knn).double().clamp_min(1e-12)            # [n, k] ascending
        m = (k - 1) / torch.log(T[:, -1:] / T[:, :-1]).sum(dim=1)
        m = m[torch.isfinite(m)]
        id_mle = float(m.mean())
        return {"id_mle": id_mle, "id_score": id_mle / s.d, "id_n_unique": n}


def build_metrics(names: list[str], params: dict[str, dict] | None = None) -> list[IsotropyMetric]:
    params = params or {}
    return [METRICS.get(n)(**params.get(n, {})) for n in names]


def compute_layer(metrics: list[IsotropyMetric], X: torch.Tensor, text_ids: torch.Tensor) -> dict[str, float]:
    stats = LayerStats(X, text_ids)
    out = {}
    for m in metrics:
        out.update(m.compute(stats))
    return out
