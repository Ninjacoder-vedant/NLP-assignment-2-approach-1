"""Result storage. One folder per (model, dataset, language), so an interrupted run loses at most one
language and a rerun skips everything already on disk.

    {root}/{model}/{dataset}/run_config.json             everything that determines the numbers
    {root}/{model}/{dataset}/token_stats.csv             tokens / words / fertility per language
    {root}/{model}/{dataset}/{lang}/metrics.csv          one row per (view, layer); written last = done
    {root}/{model}/{dataset}/{lang}/{view}/pca3d.npz     3-D PCA projection of the view's points per layer
    {root}/{model}/{dataset}/{lang}/{view}/points.safetensors   the points themselves (--save-points)

view = token | sentence-mean | sentence-last (inference.VIEWS); plotting.py adds metrics.png and pca3d.png
to each view folder.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file


# Raised when a results folder was produced with different settings
class ConfigMismatch(RuntimeError):
    pass


class ResultStore:
    def __init__(self, root: str | Path = "results"):
        """Point the store at a results folder (created lazily on first save).

        Args:
            root: folder that holds all results.
        """
        self.root = Path(root)

    # Folder of one (model, dataset), or of one of its languages; Path objects join with /
    def _dir(self, model: str, dataset: str, lang: str | None = None) -> Path:
        """Return the folder root/model/dataset, or root/model/dataset/lang."""
        folder = self.root / model / dataset
        return folder if lang is None else folder / lang

    def view_dir(self, model: str, dataset: str, lang: str, view: str) -> Path:
        """Return the folder root/model/dataset/lang/view (data and plots of one view)."""
        return self._dir(model, dataset, lang) / view

    # A language counts as done once its metrics.csv exists (it is written after everything else)
    def exists(self, model: str, dataset: str, lang: str) -> bool:
        """Return True if the metrics of (model, dataset, lang) are already saved."""
        return (self._dir(model, dataset, lang) / "metrics.csv").exists()

    def save_metrics(self, model: str, dataset: str, lang: str, rows: list[dict]) -> None:
        """Save the per-(view, layer) rows of one language as {lang}/metrics.csv (atomically).

        Args:
            model, dataset, lang: identify the file.
            rows: one dict per (view, layer).
        """
        path = self._dir(model, dataset, lang) / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first, then rename over the final name
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, path)      # atomic: a crash never leaves a half-written file

    def save_points(self, model: str, dataset: str, lang: str, view: str, points: torch.Tensor) -> None:
        """Save a view's vectors as {lang}/{view}/points.safetensors, in their own dtype (bf16 stays bf16).

        Args:
            model, dataset, lang, view: identify the file.
            points: [L+1, N, d] tensor; load one layer with safe_open(path, "pt").get_slice("points")[layer].
        """
        path = self.view_dir(model, dataset, lang, view) / "points.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"points": points.contiguous()}, path)

    def save_pca3d(self, model: str, dataset: str, lang: str, view: str, proj: dict[str, np.ndarray]) -> None:
        """Save a 3-D PCA projection (output of plotting.project_3d) as {lang}/{view}/pca3d.npz.

        Args:
            model, dataset, lang, view: identify the file.
            proj: {"coords": [L+1, N, 3], "var_ratio": [L+1, 3], "spread": [L+1]}.
        """
        path = self.view_dir(model, dataset, lang, view) / "pca3d.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        # float16 coordinates: unit-length vectors, so the precision is plenty for plotting
        np.savez_compressed(path, coords=proj["coords"].astype(np.float16), var_ratio=proj["var_ratio"],
                            spread=proj["spread"])

    def pca3d_files(self) -> list[tuple[Path, tuple[str, str, str, str]]]:
        """Every saved 3-D projection as (path, (model, dataset, lang, view))."""
        return [(f, tuple(f.relative_to(self.root).parts[:4]))
                for f in sorted(self.root.glob("*/*/*/*/pca3d.npz"))]

    def check_or_save_config(self, model: str, dataset: str, config: dict, overwrite: bool = False) -> None:
        """Refuse to mix results produced under different settings in one folder.

        Args:
            model, dataset: identify the folder.
            config: current settings (from Experiment.settings).
            overwrite: if settings differ, delete old results instead of raising.
        Saves config as run_config.json when the folder is new or was overwritten.
        Raises ConfigMismatch if settings differ and overwrite is False.
        """
        path = self._dir(model, dataset) / "run_config.json"
        if path.exists():
            old = json.loads(path.read_text())
            # Settings that changed: {name: (old value, new value)}; versions are ignored
            diff = {k: (old.get(k), v) for k, v in config.items() if k != "versions" and old.get(k) != v}
            # Same settings: resume normally
            if not diff:
                return
            if not overwrite:
                raise ConfigMismatch(
                    f"{path} was produced with different settings {diff} (old, new). "
                    "Use another --out-dir or pass --overwrite to delete these results.")
            # --overwrite with new settings: delete the old results of this folder (every language folder
            # and every CSV, which also clears results of the older flat layout)
            folder = self._dir(model, dataset)
            for f in folder.iterdir():
                if f.is_dir():
                    shutil.rmtree(f)
                elif f.suffix == ".csv":
                    f.unlink()
        # First run (or after overwrite): record the settings
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    def update_token_stats(self, model: str, dataset: str, stats: pd.DataFrame) -> None:
        """Upsert per-language token statistics (resumed runs only see the languages they process).

        Args:
            model, dataset: identify the folder.
            stats: one row per language; replaces existing rows of the same languages in token_stats.csv.
        """
        path = self._dir(model, dataset) / "token_stats.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_csv(path)
            # Keep old rows of other languages, replace rows of the languages just processed
            stats = pd.concat([old[~old["lang"].isin(stats["lang"])], stats], ignore_index=True)
        stats.sort_values("lang").to_csv(path, index=False)

    def load_all(self) -> pd.DataFrame:
        """Every language's metrics.csv concatenated into one tidy table (model, dataset, lang, view, layer, ...).

        Returns:
            DataFrame with one row per (model, dataset, lang, view, layer); empty if nothing is saved.
        """
        files = sorted(self.root.glob("*/*/*/metrics.csv"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
