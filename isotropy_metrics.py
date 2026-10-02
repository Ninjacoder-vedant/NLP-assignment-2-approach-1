"""Isotropy metrics on one layer's point cloud X [N, d] (all on GPU).

`LayerStats` caches quantities shared by several metrics (covariance eigenvalues, unit-length rows). To add a metric, subclass `IsotropyMetric`, implement `compute` (return a dict, so one metric
can emit several columns), and register it with `@METRICS.register("name")`.
"""
import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar

import torch
import torch.nn.functional as F

from registry import METRICS


def pca_normalization(X: torch.Tensor) -> torch.Tensor:
    """Rotate the points onto their principal axes.

    Args:
        X: [N, d] points.
    Returns:
        [N, min(N, d)] float32 coordinates of the centred points along the principal axes.
    """
    # float32: enough precision here, and SVD does not accept fp16/bf16
    X = X.float()
    # Centre, take the SVD, project onto the right singular vectors
    X_centered = X - X.mean(0)
    _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vh.T


# One layer's points plus shared intermediate results, each computed at most once
class LayerStats:
    def __init__(self, X: torch.Tensor):
        """Hold one layer's point cloud; shared quantities are computed lazily.

        Args:
            X: [N, d] points of one layer.
        """
        self.X = X
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
    def compute(self, stats: LayerStats) -> dict[str, float]:
        """Compute the metric on one layer.

        Args:
            stats: the layer's LayerStats.
        Returns:
            {column name: value}; one metric may return several columns.
        """
        ...


@METRICS.register("isoscore")
class IsoScore(IsotropyMetric):
    """Rudman et al. (2022). PyTorch port of `IsoScore.IsoScore` from the official numpy package,
    following the paper's steps 2-7 so everything runs on the points' device (GPU)."""

    def compute(self, stats):
        """Return {"isoscore": value in [0, 1]}; 1 = variance spread equally over all d dims."""
        # Number of dimensions
        d = stats.d

        # Step 2: PCA normalization - rotate the points onto their principal axes
        points_pca = pca_normalization(stats.X)                 # [N, min(N, d)]

        # Step 3: diagonal of the covariance matrix of the PCA-transformed points
        cov_diag = points_pca.var(dim=0)

        # Step 4: normalize the diagonal to length sqrt(d); a perfectly isotropic cloud gives all ones
        cov_diag_normalized = (cov_diag * math.sqrt(d)) / torch.linalg.vector_norm(cov_diag)

        # Step 5: isotropy defect - distance from the identity's diagonal (all ones), scaled to [0, 1]
        l2_norm = torch.linalg.vector_norm(cov_diag_normalized - 1)
        isotropy_defect = l2_norm / math.sqrt(2 * (d - math.sqrt(d)))

        # Steps 6 and 7: map the defect to a score (1 = isotropic, 0 = everything along one direction)
        score = ((d - isotropy_defect ** 2 * (d - math.sqrt(d))) ** 2 - d) / (d * (d - 1))
        return {"isoscore": float(score)}


@METRICS.register("mev")
class MaxExplainableVariance(IsotropyMetric):
    """Fraction of variance explained by the first principal component (Ethayarajh, 2019)."""

    def compute(self, stats):
        """Return {"mev": largest eigenvalue / sum of eigenvalues}, in [1/d, 1]."""
        # eigvals is ascending, so [-1] is the largest
        return {"mev": float(stats.eigvals[-1] / stats.eigvals.sum())}


@METRICS.register("avgcos")
class AvgRandomCosine(IsotropyMetric):
    """Average cosine similarity of random pairs of points: `cosine_score` from Rudman et al. (2022).
    As in the official code, pairs (i, j) are drawn uniformly with replacement (i == j is possible),
    but all at once on the GPU instead of one by one. The paper reports 1 - |avg_cos| for comparison;
    this returns the raw average."""

    def __init__(self, num_samples: int = 1_000_000, chunk: int = 2 ** 15, seed: int = 0):
        """Set the sampling parameters.

        Args:
            num_samples: number of random pairs to average over.
            chunk: pairs per batch (limits GPU memory: 2 x chunk x d gathered vectors at a time).
            seed: seed of the pair sampler, so every run gives the same value.
        """
        self.num_samples, self.chunk, self.seed = num_samples, chunk, seed

    def compute(self, stats):
        """Return {"avg_cos": mean cosine over num_samples random pairs of points}, in [-1, 1]."""
        # Every point scaled to length 1, so a dot product is a cosine
        U = stats.X_unit

        # Seeded generator on the points' device: reproducible, and indices are created on the GPU
        gen = torch.Generator(device=U.device).manual_seed(self.seed)
        total = torch.zeros((), dtype=U.dtype, device=U.device)

        for cur_samples in range(0, self.num_samples, self.chunk):
            # m = number of pairs in this batch
            m = min(self.chunk, self.num_samples - cur_samples)
            # m random pairs (i, j) of point indices
            i, j = torch.randint(stats.n, (2, m), generator=gen, device=U.device)
            # Row-wise dot products = cosines of the m pairs
            total += (U[i] * U[j]).sum()
        
        return {"avg_cos": float(total / self.num_samples)}


@METRICS.register("id")
class IntrinsicDimension(IsotropyMetric):
    """Levina & Bickel (2004) MLE of intrinsic dimension (k nearest neighbours). Torch port of
    `skdim.id.MLE().fit(X).dimension_` with its defaults (k = 20, per-point estimates combined with
    comb="mle", i.e. their harmonic mean), as used by `id_score` in Rudman et al. (2022).
    `id_score` = min(id_mle / d, 1). Exact duplicate points (e.g. the same token id at layer 0, where
    there is no positional information) are removed first: zero distances make the estimator degenerate."""

    def __init__(self, k: int = 20, chunk: int = 2048):
        """Set the estimator parameters.

        Args:
            k: number of nearest neighbours used by the estimator.
            chunk: rows per distance computation (limits GPU memory).
        """
        self.k, self.chunk = k, chunk

    def compute(self, stats):
        """Estimate the intrinsic dimension of the layer's points.

        Returns:
            {"id_mle": estimated dimension, "id_score": min(id_mle / d, 1),
             "id_n_unique": distinct points used}.
            id values are NaN if there are no more than k distinct points.
        """
        # Remove identical rows (duplicate points)
        X = torch.unique(stats.X, dim=0)
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
        # Per-point estimate: m_i = (k-1) / sum_j log(T_k / T_j)
        # Combined as skdim's comb="mle": harmonic mean 1 / mean(1 / m_i). Computing 1 / m_i directly
        # also avoids m_i = inf when all k distances of a point are equal
        inv_m = torch.log(T[:, -1:] / T[:, :-1]).sum(dim=1) / (k - 1)
        id_mle = float(1 / inv_m.mean())
        return {"id_mle": id_mle, "id_score": min(id_mle / stats.d, 1.0), "id_n_unique": n}


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


def compute_layer(metrics: list[IsotropyMetric], X: torch.Tensor) -> dict[str, float]:
    """Run all metrics on one layer.

    Args:
        metrics: metric objects from build_metrics().
        X: [N, d] points of one layer.
    Returns:
        All metric columns merged into one dict.
    """
    # One shared LayerStats, so eigenvalues etc. are computed once for all metrics
    stats = LayerStats(X)
    out = {}
    # Merge every metric's columns into one dict
    for m in metrics:
        out.update(m.compute(stats))
    return out
