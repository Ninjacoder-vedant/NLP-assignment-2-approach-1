"""Result storage. One CSV per (model, dataset, language), so an interrupted run loses at most one
language and a rerun skips everything already on disk.

    {root}/{model}/{dataset}/{lang}.csv         one row per layer
    {root}/{model}/{dataset}/token_stats.csv    tokens / words / fertility per language
    {root}/{model}/{dataset}/run_config.json    everything that determines the numbers
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# Raised when a results folder was produced with different settings
class ConfigMismatch(RuntimeError):
    pass


class ResultStore:
    def __init__(self, root: str | Path = "results"):
        self.root = Path(root)

    # Folder of one (model, dataset); Path objects join with /
    def _dir(self, model: str, dataset: str) -> Path:
        return self.root / model / dataset

    # A language counts as done once its CSV exists
    def exists(self, model: str, dataset: str, lang: str) -> bool:
        return (self._dir(model, dataset) / f"{lang}.csv").exists()

    def save_lang(self, model: str, dataset: str, lang: str, rows: list[dict]) -> None:
        path = self._dir(model, dataset) / f"{lang}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first, then rename over the final name
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, path)      # atomic: a crash never leaves a half-written file

    def save_points(self, model: str, dataset: str, lang: str, points: np.ndarray) -> None:
        path = self._dir(model, dataset) / "points" / f"{lang}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        # float16 halves the file size; precision is enough for later analysis
        np.save(path, points.astype(np.float16))

    def check_or_save_config(self, model: str, dataset: str, config: dict, overwrite: bool = False) -> None:
        """Refuse to mix results produced under different settings in one folder."""
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
            # --overwrite with new settings: delete the old results of this folder
            for f in [*self._dir(model, dataset).glob("*.csv"), *self._dir(model, dataset).glob("points/*.npy")]:
                f.unlink()
        # First run (or after overwrite): record the settings
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

    def update_token_stats(self, model: str, dataset: str, stats: pd.DataFrame) -> None:
        """Upsert per-language token statistics (resumed runs only see the languages they process)."""
        path = self._dir(model, dataset) / "token_stats.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old = pd.read_csv(path)
            # Keep old rows of other languages, replace rows of the languages just processed
            stats = pd.concat([old[~old["lang"].isin(stats["lang"])], stats], ignore_index=True)
        stats.sort_values("lang").to_csv(path, index=False)

    def load_all(self) -> pd.DataFrame:
        """Every per-language file concatenated into one tidy table (model, dataset, lang, layer, ...)."""
        # All {model}/{dataset}/{lang}.csv files (token_stats.csv has different columns)
        files = [f for f in sorted(self.root.glob("*/*/*.csv")) if f.name != "token_stats.csv"]
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
