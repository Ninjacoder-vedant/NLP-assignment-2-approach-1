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


# One layer's points plus shared intermediate results, each computed at most once
class LayerStats:
    def __init__(self, X: torch.Tensor, text_ids: torch.Tensor):
        """Hold one layer's point cloud; shared quantities are computed lazily.

        Args:
            X: [N, d] points of one layer.
            text_ids: [N] index of the text each point came from.
        """
        self.X = X
        self.text_ids = text_ids
        self.n, self.d = X.shape

    @cached_property
    def eigvals(self) -> torch.Tensor:
        """Covariance eigenvalues, ascending, float64 (= the variances along the principal axes)."""
        # cov expects variables in rows, hence X.T; eigvalsh = eigenvalues of a symmetric matrix.
        # clamp_min(0) removes tiny negative values caused by rounding
        return torch.linalg.eigvalsh(torch.cov(self.X.double().T)).clamp_min(0)

    # Every point scaled to length 1 (for cosine similarity)
    @cached_property
    def X_unit(self) -> torch.Tensor:
        """Return X in float64 with every row scaled to unit length."""
        return F.normalize(self.X.double(), dim=1)


# Base class of all metrics: one compute() that returns {column name: value}
class IsotropyMetric(ABC):
    name: ClassVar[str]   # set by @METRICS.register

    @abstractmethod
    def compute(self, s: LayerStats) -> dict[str, float]:
        """Compute the metric on one layer.

        Args:
            s: the layer's LayerStats.
        Returns:
            {column name: value}; one metric may return several columns.
        """
        ...


@METRICS.register("isoscore")
class IsoScore(IsotropyMetric):
    """Rudman et al. (2022). Identical to `IsoScore.IsoScore(X)` from the official package, which uses
    the singular values of the covariance (= its eigenvalues, since it is PSD)."""

    def compute(self, s):
        """Return {"isoscore": value in [0, 1]}; 1 = variance spread equally over all d dims."""
        # Variance along each principal axis; n here is the dimension d, as in the paper
        pcs = s.eigvals
        n = torch.tensor(float(s.d), dtype=torch.float64)
        # Rescale the variance vector to length sqrt(n); a perfectly isotropic cloud gives all ones
        pcs_norm = pcs * n.sqrt() / torch.linalg.vector_norm(pcs)
        # Isotropy defect: distance from the all-ones vector, normalised to [0, 1]
        defect = torch.linalg.vector_norm(pcs_norm - 1) / torch.sqrt(2 * (n - n.sqrt()))
        # Map the defect to a score: 1 = isotropic, 0 = everything along one direction
        score = ((n - defect ** 2 * (n - n.sqrt())) ** 2 - n) / (n * (n - 1))
        return {"isoscore": float(score)}


@METRICS.register("mev")
class MaxExplainableVariance(IsotropyMetric):
    """Fraction of variance explained by the first principal component (Ethayarajh, 2019)."""

    def compute(self, s):
        """Return {"mev": largest eigenvalue / sum of eigenvalues}, in [1/d, 1]."""
        # eigvals is ascending, so [-1] is the largest
        return {"mev": float(s.eigvals[-1] / s.eigvals.sum())}


@METRICS.register("avgcos")
class AvgRandomCosine(IsotropyMetric):
    """Mean cosine similarity between tokens of *different* texts (Ethayarajh, 2019).
    Computed exactly over all such pairs with sum vectors, O(N d), instead of sampling random pairs:
    sum_{i != j} cos = |S|^2 - N, and pairs within the same text are removed the same way per text."""

    def compute(self, s):
        """Return {"avg_cos": mean cosine over all pairs of tokens from different texts}.

        NaN if all tokens come from a single text.
        """
        U = s.X_unit
        # per_text[t] = sum of the unit vectors of text t (index_add_ adds row i into row text_ids[i])
        per_text = torch.zeros((int(s.text_ids.max()) + 1, s.d), dtype=U.dtype, device=U.device)
        per_text.index_add_(0, s.text_ids, U)
        # Sampled tokens per text
        counts = torch.bincount(s.text_ids).double()
        total = U.sum(0)
        # |sum of all|^2 = sum of cos over all ordered pairs; subtract the same-text pairs
        cross_sum = total @ total - (per_text * per_text).sum()
        cross_pairs = s.n ** 2 - (counts ** 2).sum()
        # Only possible if every token comes from a single text
        if cross_pairs == 0:
            return {"avg_cos": float("nan")}
        return {"avg_cos": float(cross_sum / cross_pairs)}


@METRICS.register("id")
class IntrinsicDimension(IsotropyMetric):
    """Levina & Bickel (2004) MLE of intrinsic dimension (k nearest neighbours), averaged over points.
    `id_score` = id_mle / d. Exact duplicate points (e.g. the same token id at layer 0, where there is
    no positional information) are removed first: zero distances make the estimator degenerate."""

    def __init__(self, k: int = 20, chunk: int = 2048):
        """Set the estimator parameters.

        Args:
            k: number of nearest neighbours used by the estimator.
            chunk: rows per distance computation (limits GPU memory).
        """
        self.k, self.chunk = k, chunk

    def compute(self, s):
        """Estimate the intrinsic dimension of the layer's points.

        Returns:
            {"id_mle": estimated dimension, "id_score": id_mle / d, "id_n_unique": distinct points used}.
            id values are NaN if there are no more than k distinct points.
        """
        # Remove identical rows (duplicate points)
        X = torch.unique(s.X, dim=0)
        X = X - X.mean(0)                   # translation-invariant; improves fp32 cdist precision
        n, k = X.shape[0], self.k
        # Need more than k distinct points to have k neighbours
        if n <= k:
            return {"id_mle": float("nan"), "id_score": float("nan"), "id_n_unique": n}
        # Distances to the k nearest neighbours, computed in chunks of rows to limit GPU memory
        knn = []
        for a in range(0, n, self.chunk):
            dist = torch.cdist(X[a:a + self.chunk], X)
            rows = torch.arange(dist.shape[0], device=X.device)
            dist[rows, rows + a] = float("inf")                 # exclude self
            knn.append(dist.topk(k, largest=False).values)
        T = torch.cat(knn).double().clamp_min(1e-12)            # [n, k] ascending
        # Per-point estimate: m = (k-1) / sum_j log(T_k / T_j)
        m = (k - 1) / torch.log(T[:, -1:] / T[:, :-1]).sum(dim=1)
        m = m[torch.isfinite(m)]
        id_mle = float(m.mean())
        return {"id_mle": id_mle, "id_score": id_mle / s.d, "id_n_unique": n}


def build_metrics(names: list[str], params: dict[str, dict] | None = None) -> list[IsotropyMetric]:
    """Create metric objects from registered names.

    Args:
        names: e.g. ["isoscore", "mev", "avgcos", "id"].
        params: optional constructor arguments per metric, e.g. {"id": {"k": 20}}.
    Returns:
        List of metric objects, in the given order.
    """
    # Create each metric by name, passing its parameters if given, e.g. {"id": {"k": 20}}
    params = params or {}
    return [METRICS.get(n)(**params.get(n, {})) for n in names]


def compute_layer(metrics: list[IsotropyMetric], X: torch.Tensor, text_ids: torch.Tensor) -> dict[str, float]:
    """Run all metrics on one layer.

    Args:
        metrics: metric objects from build_metrics().
        X: [N, d] points of one layer.
        text_ids: [N] text index of each point.
    Returns:
        All metric columns merged into one dict.
    """
    # One shared LayerStats, so eigenvalues etc. are computed once for all metrics
    stats = LayerStats(X, text_ids)
    out = {}
    # Merge every metric's columns into one dict
    for m in metrics:
        out.update(m.compute(stats))
    return out
