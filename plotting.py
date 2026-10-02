"""Layer-depth plots of the isotropy metrics: one PNG per (model, dataset, language).

    {root}/{model}/{dataset}/plots/{lang}.png

Top panel: the bounded metrics (isoscore, mev, avg_cos, id_score) on one shared axis.
Bottom panel: id_mle, which is in dimensions rather than [0, 1], so it gets its own axis.

    python plotting.py                                    # plot every saved CSV
    python plotting.py --models gemma-3-1b-pt --datasets in22-gen --langs hin_Deva
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")       # no display needed: figures are only saved to files
import matplotlib.pyplot as plt
import pandas as pd
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
    """Draw layer depth vs every isotropy metric of one (model, dataset, language) and save it.

    Args:
        df: the per-layer rows of one {lang}.csv.
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
    bottom.set_xlabel("layer (0 = embedding output)", color=MUTED)
    bottom.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{row['model']} / {row['dataset']} / {row['lang']}   (N={row['n_tokens']}, d={row['d']})",
                 color=INK, fontsize=11)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)      # free memory: many figures are made in one run


def plot_results(root: str | Path = "results", models: list[str] | None = None,
                 datasets: list[str] | None = None, langs: list[str] | None = None) -> list[Path]:
    """Plot every saved (model, dataset, language) CSV, optionally filtered.

    Args:
        root: results folder.
        models, datasets, langs: keep only these (None = all saved ones).
    Returns:
        The paths of the written PNGs; {root}/{model}/{dataset}/plots/{lang}.png.
    """
    store = ResultStore(root)
    df = store.load_all()
    if df.empty:
        log.warning("no results under %s to plot", store.root)
        return []
    # Keep only the requested models / datasets / languages
    for col, keep in (("model", models), ("dataset", datasets), ("lang", langs)):
        if keep is not None:
            df = df[df[col].isin(keep)]
    paths = []
    for (model, dataset, lang), g in df.groupby(["model", "dataset", "lang"]):
        path = store.root / model / dataset / "plots" / f"{lang}.png"
        plot_lang(g, path)
        paths.append(path)
    log.info("saved %d plot(s) under %s", len(paths), store.root)
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
