"""Plots per (model, dataset, language, view), next to the view's data:

    {root}/{model}/{dataset}/{lang}/{view}/metrics.png   layer depth vs every isotropy metric
    {root}/{model}/{dataset}/{lang}/{view}/pca3d.png     3-D PCA scatter of the points, one panel per layer

view = token | sentence-mean | sentence-last.

Metrics plot: top panel = the bounded metrics (isoscore, mev, avg_cos, id_score) on one shared axis;
bottom panel = id_mle, which is in dimensions rather than [0, 1], so it gets its own axis.

3-D plot: drawn from {lang}/{view}/pca3d.npz, which run.py writes for every language and view (with or
without --plot) using project_3d().

    python plotting.py                                    # plot every saved CSV
    python plotting.py --models gemma-3-1b-pt --datasets in22-gen --langs hin_Deva
"""
import argparse
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")       # no display needed: figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator

from results import ResultStore

log = logging.getLogger("isotropy.plotting")

# Bounded metrics drawn on the top panel: column -> (legend label, colour). Fixed colour per metric,
# so the same metric looks the same in every plot
BOUNDED = {
    "isoscore": ("IsoScore", "#2a78d6"),
    "mev": ("MEV (1st PC variance share)", "#eb6834"),
    "avg_cos": ("Avg. random cosine", "#1baf7a"),
    "id_score": ("ID score (id_mle / d)", "#eda100"),
}
ID_MLE = ("Intrinsic dimension (MLE)", "#4a3aa7")

# Recessive chart chrome
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# 3-D panels: ordinary layers, the penultimate layer (N-1) and the final layer (N)
LAYER_COLOR, PENULT_COLOR, FINAL_COLOR = "#2a78d6", "#e34948", "#1baf7a"
PCA_COLS = 6              # panels per row of the 3-D figure
PCA_MAX_SHOW = 5000       # points drawn per panel (display only; the .npz keeps all N)

# What the points of each view are: (plural unit, description for titles)
VIEW_LABELS = {
    "token": ("tokens", "token representations"),
    "sentence-mean": ("sentences", "mean-pooled sentence embeddings"),
    "sentence-last": ("sentences", "last-token sentence embeddings"),
}


def _style(ax) -> None:
    """Hairline grid, no top/right spines, muted tick labels."""
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_lang(df: pd.DataFrame, path: Path) -> None:
    """Draw layer depth vs every isotropy metric of one (model, dataset, language, view) and save it.

    Args:
        df: the per-layer rows of one view of a metrics.csv.
        path: output .png file (parent folders are created).
    """
    df = df.sort_values("layer")
    row = df.iloc[0]
    has_id = "id_mle" in df and df["id_mle"].notna().any()
    # Second, shorter panel only when there is an id_mle column to show
    fig, axes = plt.subplots(2 if has_id else 1, 1, figsize=(8, 6.5 if has_id else 4.5), sharex=True,
                             squeeze=False, gridspec_kw={"height_ratios": [3, 1.4] if has_id else [1]})
    ax = axes[0, 0]

    for col, (label, color) in BOUNDED.items():
        if col in df:
            ax.plot(df["layer"], df[col], color=color, linewidth=2, marker="o", markersize=4, label=label)
    # avg_cos can go negative: mark zero so the sign is readable
    ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=0)
    ax.set_ylabel("metric value", color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    _style(ax)

    if has_id:
        label, color = ID_MLE
        ax2 = axes[1, 0]
        ax2.plot(df["layer"], df["id_mle"], color=color, linewidth=2, marker="o", markersize=4)
        ax2.set_ylabel(label, color=MUTED, fontsize=9)
        ax2.set_ylim(bottom=0)
        _style(ax2)

    bottom = axes[-1, 0]
    bottom.set_xlabel("model stage (embedding output, then transformer blocks)", color=MUTED)
    if "stage" in df:
        bottom.set_xticks(df["layer"], df["stage"], rotation=45, ha="right")
    else:
        bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    unit, what = VIEW_LABELS.get(row["view"], ("points", row["view"]))
    fig.suptitle(f"{row['model']} / {row['dataset']} / {row['lang']}: {what}\n(N={row['n_points']} {unit}, "
                 f"d={row['d']})", color=INK, fontsize=11)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)      # free memory: many figures are made in one run


# No gradients needed: this is pure linear algebra on saved activations
@torch.inference_mode()
def project_3d(points: torch.Tensor, device: str | torch.device | None = None) -> dict[str, np.ndarray]:
    """Project every layer's points onto that layer's first 3 principal axes.

    Each vector is scaled to unit length first (cosine geometry, the same view as avg_cos), so
    layers with very different norms share one scale and can be drawn on the same axes.

    Args:
        points: [L+1, N, d] hidden states (the same N points at every layer), on any device.
        device: where to compute, one layer at a time (None = where points are).
    Returns:
        {"coords": [L+1, N, 3] float32 PC1-3 coordinates of the centred unit vectors,
         "var_ratio": [L+1, 3] share of the layer's total variance on PC1, PC2, PC3,
         "spread": [L+1] RMS distance of the 3-D points from their centroid}.
        Columns beyond d are zero when d < 3.
    """
    coords, var_ratio, spread = [], [], []
    for X in points:
        Xc = F.normalize(X.to(device).float(), dim=1)
        Xc = Xc - Xc.mean(0)
        # d x d covariance; eigh in float64 (ascending eigenvalues), clamp rounding negatives to 0
        cov = (Xc.T @ Xc).double() / max(len(Xc) - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(0)
        k = min(3, len(eigvals))
        top, V = eigvals[-k:].flip(0), eigvecs[:, -k:].flip(1).float()
        # Eigenvector signs are arbitrary: make each axis's largest loading positive (stable plots)
        V = V * torch.sign(V[V.abs().argmax(0), torch.arange(k)])
        P = torch.zeros(len(Xc), 3, device=Xc.device)
        P[:, :k] = Xc @ V
        ratio = torch.zeros(3, dtype=torch.float64)
        ratio[:k] = (top / eigvals.sum().clamp_min(1e-30)).cpu()
        coords.append(P.cpu().numpy())
        var_ratio.append(ratio.numpy())
        # Mean squared distance to the centroid in 3-D = sum of the 3 variances
        spread.append(math.sqrt(float(top.sum())))
    return {"coords": np.stack(coords), "var_ratio": np.stack(var_ratio).astype(np.float32),
            "spread": np.array(spread, dtype=np.float32)}


def _layer_title(layer: int, last: int) -> tuple[str, str]:
    """Panel title and colour: the final and penultimate layers are marked like in the paper figure."""
    if layer == last:
        return f"Layer {layer} (final N)", FINAL_COLOR
    if layer == last - 1:
        return f"Layer {layer} (N-1)", PENULT_COLOR
    return (f"Layer {layer} (embeddings)" if layer == 0 else f"Layer {layer}"), LAYER_COLOR


def plot_lang_3d(data, title: str, path: Path, seed: int = 0, view: str = "token") -> None:
    """Draw one 3-D PCA scatter per layer (every layer, none skipped) and save the grid as a PNG.

    All panels share the same axis limits, so a layer whose cloud collapses really looks smaller.

    Args:
        data: mapping with the arrays of project_3d() (e.g. the loaded {lang}.npz).
        title: figure title, e.g. "gemma-3-1b-pt / in22-gen / hin_Deva".
        path: output .png file (parent folders are created).
        seed: picks which points are drawn when there are more than PCA_MAX_SHOW (same at every layer).
        view: which view the points are (VIEW_LABELS key), for the titles.
    """
    coords, var_ratio, spread = (np.asarray(data[k], dtype=np.float32) for k in ("coords", "var_ratio", "spread"))
    n_layers, n = coords.shape[:2]
    # Draw at most PCA_MAX_SHOW points: the same ones at every layer, so panels stay comparable
    if n > PCA_MAX_SHOW:
        coords = coords[:, np.sort(np.random.default_rng(seed).choice(n, PCA_MAX_SHOW, replace=False))]
    shown = coords.shape[1]
    # Fainter points when there are many, so dense cores still show their shape
    alpha = float(np.clip(1000 / max(shown, 1), 0.15, 0.6))
    # One symmetric limit for every axis of every panel: the widest layer's 99.5% quantile, so a few
    # far outliers don't shrink everything else
    lim = float(np.quantile(np.abs(coords), 0.995, axis=(1, 2)).max()) * 1.1 or 1.0
    ticks = [-lim * 0.8, 0, lim * 0.8]
    tick_fmt = "{x:.2f}" if lim < 1 else "{x:.1f}"

    ncols = min(PCA_COLS, n_layers)
    nrows = math.ceil(n_layers / ncols)
    fig = plt.figure(figsize=(3.1 * ncols, 3.1 * nrows + 0.9))
    for layer in range(n_layers):
        ax = fig.add_subplot(nrows, ncols, layer + 1, projection="3d")
        name, color = _layer_title(layer, n_layers - 1)
        # 3-D axes do not clip: drop the few outliers beyond the limits instead of drawing them outside
        inside = (np.abs(coords[layer]) <= lim).all(1)
        x, y, z = coords[layer][inside].T
        ax.scatter(x, y, z, s=2, color=color, alpha=alpha, linewidths=0, depthshade=False, rasterized=True)
        ax.set_title(name, fontsize=10, color=INK, fontweight="bold", pad=0)
        # Metrics box, top left (same idea as the paper figure)
        ax.text2D(0.02, 0.97, f"spread {spread[layer]:.4f}\ntop-3 var {var_ratio[layer].sum():.1%}",
                  transform=ax.transAxes, va="top", fontsize=7, color=INK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#c3c2b7", linewidth=0.6))
        for axis, label in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("PC1", "PC2", "PC3")):
            axis.set_ticks(ticks)
            axis.set_major_formatter(tick_fmt)
            axis.set_pane_color((0.96, 0.96, 0.95, 1.0))
            axis.set_label_text(label, fontsize=7, color=MUTED)
            axis.labelpad = -8
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.set_box_aspect((1, 1, 1), zoom=0.85)
        ax.tick_params(labelsize=6, colors=MUTED, pad=-3)
        ax.view_init(elev=20, azim=-60)

    unit, what = VIEW_LABELS.get(view, ("points", view))
    sample = f"{n} {unit}" + (f", {shown} drawn" if shown < n else "")
    fig.suptitle(f"{title}: 3-D PCA of {what} per layer ({sample})",
                 color=INK, fontsize=13, fontweight="bold", y=1 - 0.25 / fig.get_figheight())
    fig.text(0.5, 1 - 0.6 / fig.get_figheight(),
             "Vectors scaled to unit length, PCA fitted per layer; same axes in every panel.   "
             "spread = RMS distance from the centroid in PC1-3;  top-3 var = share of total variance on PC1-3",
             ha="center", va="top", fontsize=9, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.9 / fig.get_figheight(),
                        wspace=0.02, hspace=0.08)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_results(root: str | Path = "results", models: list[str] | None = None,
                 datasets: list[str] | None = None, langs: list[str] | None = None) -> list[Path]:
    """Plot every saved (model, dataset, language, view), optionally filtered: the metric CSVs and the
    3-D PCA projections.

    Args:
        root: results folder.
        models, datasets, langs: keep only these (None = all saved ones).
    Returns:
        The paths of the written PNGs; {root}/{model}/{dataset}/{lang}/{view}/metrics.png and
        .../{view}/pca3d.png.
    """
    store = ResultStore(root)

    def wanted(model: str, dataset: str, lang: str) -> bool:
        """True if (model, dataset, lang) passes the filters."""
        return all(keep is None or v in keep for v, keep in ((model, models), (dataset, datasets), (lang, langs)))

    paths = []
    df = store.load_all()
    if not df.empty:
        for (model, dataset, lang, view), g in df.groupby(["model", "dataset", "lang", "view"]):
            if wanted(model, dataset, lang):
                path = store.view_dir(model, dataset, lang, view) / "metrics.png"
                plot_lang(g, path)
                paths.append(path)
    for f, (model, dataset, lang, view) in store.pca3d_files():
        if wanted(model, dataset, lang):
            path = f.with_name("pca3d.png")
            with np.load(f) as data:
                plot_lang_3d(data, f"{model} / {dataset} / {lang}", path, view=view)
            paths.append(path)
    if paths:
        log.info("saved %d plot(s) under %s", len(paths), store.root)
    else:
        log.warning("no results under %s to plot", store.root)
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--langs", nargs="+")
    a = p.parse_args()
    plot_results(a.out_dir, a.models, a.datasets, a.langs)
