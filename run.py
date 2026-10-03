"""Layer-wise isotropy of Indic languages: every (model, dataset, language) -> one CSV of per-layer metrics
for each view (sampled tokens, mean-pooled and last-token sentence embeddings).

    python run.py                                        # all models x all datasets x 22 languages
    python run.py --models gemma-3-1b-pt --datasets in22-gen flores-plus --langs hin_Deva tam_Taml
    python run.py --models embeddinggemma-300m --device cuda:1   # run a second process on GPU 1
    python run.py --list
    python run.py --models gemma-3-1b-pt --langs hin_Deva --debug   # per-batch / per-layer progress
    python run.py --models gemma-3-1b-pt --plot          # also save metric and 3-D PCA plots per language/view

Reruns skip every (model, dataset, language) already saved.
"""
import argparse
import dataclasses
import importlib.metadata as md
import logging
import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import dataset_loader  # noqa: F401  (registers datasets)
from inference import HiddenStateExtractor
from isotropy_metrics import build_metrics, compute_layer
from languages import ENGLISH, LANGUAGES
from model_loader import MODEL_SPECS, load_model, resolve_dtype
from plotting import plot_results, project_3d
from registry import DATASETS, METRICS
from results import ResultStore
from utils import free_cuda, setup_hf_token

# Logger for progress messages ("isotropy" namespace, so library logs can stay quiet)
log = logging.getLogger("isotropy")


# All settings of one run; defaults = everything (all models, datasets, 22 languages)
@dataclass
class RunConfig:
    # field(default_factory=...) builds a fresh list per config instead of sharing one
    models: list[str] = field(default_factory=lambda: list(MODEL_SPECS))
    datasets: list[str] = field(default_factory=lambda: DATASETS.names())
    langs: list[str] = field(default_factory=lambda: list(LANGUAGES))
    # Every text is always run (sentence views use all of them). Token view: None (default) = every
    # token; an int N = the same N for every language, dataset and model (IsoScore depends on N), drawn
    # uniformly from all tokens of the language. Languages with fewer than N tokens in total are skipped
    # and logged in token_stats.csv.
    n_tokens: int | None = None
    batch_size: int = 32                  # texts per forward pass
    # How many texts to take from monolingual corpora (parallel datasets use all rows)
    max_texts: dict[str, int] = field(default_factory=lambda: {"sentence": 2000, "document": 200})
    metrics: list[str] = field(default_factory=lambda: METRICS.names())
    metric_params: dict[str, dict] = field(default_factory=lambda: {"id": {"k": 20}})
    seed: int = 0
    device: str = "cuda:0"
    dtype: str = "auto"                   # auto = bf16 on Ampere+ GPUs, else fp32
    out_dir: str = "results"
    cache_dir: str = "cache/texts"
    save_points: bool = False             # also dump each view's vectors to .safetensors
    overwrite: bool = False               # allow replacing results made with other settings
    debug: bool = False                   # log every forward-pass batch and metric layer
    plot: bool = False                    # after the run, save metric and 3-D PCA plots per language/view


def make_dataset(name: str, cfg: RunConfig):
    """Build a registered dataset with the run's text cap, cache folder and seed.

    Args:
        name: registered dataset name, e.g. "in22-gen".
        cfg: run settings (max_texts per granularity, cache_dir, seed).
    Returns:
        The dataset object (parallel datasets get max_texts=None, i.e. all rows).
    """
    # Look up the dataset class by its registered name, e.g. "in22-gen" -> IN22Gen
    cls = DATASETS.get(name)
    # Parallel sets are small: use every row. Monolingual corpora: cap by granularity
    max_texts = None if cls.parallel else cfg.max_texts[cls.granularity]
    return cls(max_texts=max_texts, cache_dir=cfg.cache_dir, seed=cfg.seed)


class Experiment:
    def __init__(self, cfg: RunConfig):
        """Prepare the result store, metric objects and dataset objects for a run.

        Args:
            cfg: all run settings.
        """
        self.cfg = cfg
        # Where CSVs go; also answers "is this language already done?"
        self.store = ResultStore(cfg.out_dir)
        # Instantiate metric objects from their names, e.g. ["isoscore", "mev", ...]
        self.metrics = build_metrics(cfg.metrics, cfg.metric_params)
        # Dataset objects are created once and reused for every model
        self.datasets = [make_dataset(n, cfg) for n in cfg.datasets]

    def settings(self, spec, ds) -> dict:
        """Everything that determines the numbers for one (model, dataset) folder.

        Args:
            spec: the model's ModelSpec.
            ds: the dataset object.
        Returns:
            Dict saved as run_config.json and compared on resume (except "versions").
        """
        c = self.cfg
        return {
            "model": spec.key, 
            "hf_id": spec.hf_id, 
            "dtype": str(resolve_dtype(c.dtype, c.device)),
            "dataset": ds.name,
            "max_texts": ds.max_texts, 
            "n_tokens": c.n_tokens, 
            # How the N tokens are drawn (changes the numbers, so a change needs --overwrite)
            "token_sampling": None if c.n_tokens is None else "all texts run; n_tokens drawn uniformly from all tokens",
            "pooling": "mean / last over non-special tokens",
            "seed": c.seed,
            "metrics": c.metrics, 
            "metric_params": {m: c.metric_params.get(m, {}) for m in c.metrics},
            # Library versions are saved for reference but not compared on resume
            "versions": {p: md.version(p) for p in ("torch", "transformers", "sentence-transformers", "datasets")},
        }

    def run(self) -> None:
        """Run every configured model on every dataset and language, skipping finished ones.

        Loads each model once, computes the missing (dataset, language) results and saves them to CSV.
        Raises ConfigMismatch if a results folder was produced with different settings.
        """
        # Models are the outer loop: loading a model is the expensive step, so do it once
        log.info("run: %d model(s) x %d dataset(s) x %d language(s) on %s",
                 len(self.cfg.models), len(self.datasets), len(self.cfg.langs), self.cfg.device)

        # Traverse the model keys
        for key in self.cfg.models:
            # Find the model spec
            spec = MODEL_SPECS[key]
            log.info("[%s] checking saved results", key)

            # todo = {dataset name: languages still to compute for this model}
            todo = {}

            for ds in self.datasets:
                # Stop if this folder holds results made with different settings
                self.store.check_or_save_config(key, ds.name, self.settings(spec, ds), self.cfg.overwrite)
                # Languages this dataset actually has (e.g. Wikipedia has no Bodo/Dogri)
                have = set(ds.languages())
                missing = [l for l in self.cfg.langs if l not in have]
                if missing:
                    log.info("%s has no %s", ds.name, missing)
                # Keep only languages that exist and have no saved CSV yet (resume)
                todo[ds.name] = [l for l in self.cfg.langs if l in have and not self.store.exists(key, ds.name, l)]

            # Nothing left for this model: don't even load it
            if not any(todo.values()):
                log.info("[%s] everything already done", key)
                continue

            log.info("[%s] loading %s", key, spec.hf_id)
            model = load_model(spec, self.cfg.device, self.cfg.dtype)

            # try/finally: free GPU memory even if a dataset crashes
            try:
                # The extractor turns texts into per-layer hidden states for this model
                ext = HiddenStateExtractor(model, self.cfg.batch_size)
                for ds in self.datasets:
                    if todo[ds.name]:
                        log.info("[%s/%s] %d language(s) to do: %s", key, ds.name, len(todo[ds.name]), todo[ds.name])
                        self.run_dataset(key, ext, ds, todo[ds.name])
            finally:
                model.unload()
                log.info("[%s] model unloaded", key)

        # Plots also cover languages finished by earlier (resumed) runs
        if self.cfg.plot:
            plot_results(self.cfg.out_dir, self.cfg.models, self.cfg.datasets, self.cfg.langs)
        log.info("run finished")

    def run_dataset(self, key: str, ext: HiddenStateExtractor, ds, langs: list[str]) -> None:
        """Compute and save per-layer metrics of every view for the given languages of one dataset.

        Args:
            key: model key (results folder name).
            ext: extractor wrapping the loaded model.
            ds: the dataset object.
            langs: canonical language codes still to compute.
        Writes {lang}/metrics.csv and {lang}/{view}/pca3d.npz per language and updates token_stats.csv.
        Every text is run. Token view: with n_tokens set, exactly n_tokens tokens drawn uniformly from all
        tokens of the language (languages with fewer are skipped and recorded as such); with
        n_tokens=None every token. Sentence views: one mean-pooled and one last-token vector per text.
        """
        n = self.cfg.n_tokens
        dev = ext.model.device
        stats = []
        for i_lang, lang in enumerate(langs, 1):
            tag = f"[{key}/{ds.name}/{lang}]"
            log.info("%s (%d/%d) loading texts", tag, i_lang, len(langs))
            texts = ds.load(lang)
            # Token statistics for token_stats.csv, from the tokenizer alone (fertility = tokens per word,
            # words split on whitespace)
            counts = ext.count_tokens(texts)
            n_real, n_words = int(counts.sum()), sum(len(t.split()) for t in texts)
            stat = {"lang": lang, "source_code": ds.source_code(lang), "n_texts": len(texts),
                    "n_words": n_words, "n_tokens": n_real, "fertility": n_real / max(n_words, 1),
                    "status": "done"}
            stats.append(stat)
            # Too few tokens to sample N: skip (before any inference) instead of lowering N for everyone
            if n is not None and n_real < n:
                stat["status"] = f"skipped: {n_real} tokens < n_tokens={n}"
                log.warning("%s %s", tag, stat["status"])
                continue
            # A text with no non-special token has no sentence vector: drop it
            if (counts == 0).any():
                log.warning("%s dropping %d text(s) with no tokens", tag, int((counts == 0).sum()))
                texts = [t for t, c in zip(texts, counts) if c > 0]
            # Seed depends only on (seed, dataset, language): a resumed run picks the same tokens.
            # crc32 is a stable hash (Python's hash() changes between runs)
            rng = np.random.default_rng([self.cfg.seed, zlib.crc32(f"{ds.name}/{lang}".encode())])
            # Positions in the combined token matrix of all texts: drawn before inference, gathered during
            # it, so the full matrix is never held in memory
            token_idx = None if n is None else np.sort(rng.choice(n_real, size=n, replace=False))
            log.info("%s inference on %d texts (%d tokens), token view keeps %s", tag, len(texts), n_real,
                     "all" if n is None else n)
            views = ext.extract(texts, token_idx)
            stat["n_tokens_used"] = views["token"].shape[1]
            # One row per (view, layer): identifying columns + all metric values (** merges the metric dict in)
            rows = []
            for view, points in views.items():
                n_layers, n_points, d = points.shape
                stages = ext.model.layer_stages(n_layers)
                log.info("%s [%s] computing metrics on %d layers x %d points", tag, view, n_layers, n_points)
                for i in range(n_layers):
                    # One layer on the device at a time (float32 for the metrics)
                    X = points[i].to(dev).float()
                    rows.append({"model": key, "dataset": ds.name, "lang": lang, "source_code": stat["source_code"],
                                 "view": view, "layer": i, "stage": stages[i], "n_points": n_points, "d": d,
                                 **compute_layer(self.metrics, X)})
                    log.debug("%s [%s] layer %d/%d done", tag, view, i, n_layers - 1)
                del X
                # 3-D PCA projection of every layer, always saved (--plot only decides whether it is drawn)
                self.store.save_pca3d(key, ds.name, lang, view, project_3d(points, dev))
                if self.cfg.save_points:
                    self.store.save_points(key, ds.name, lang, view, points)
                # Progress line with the last layer's metrics
                last = rows[-1]
                log.info("%s [%s] last layer: isoscore=%.4f mev=%.4f avg_cos=%.4f id=%.1f", tag, view,
                         last.get("isoscore", np.nan), last.get("mev", np.nan), last.get("avg_cos", np.nan),
                         last.get("id_mle", np.nan))
                free_cuda()
            # Written last: its presence marks the language as done
            self.store.save_metrics(key, ds.name, lang, rows)
            log.info("%s saved %d views x %d layers", tag, len(views), n_layers)
            del views
            free_cuda()
        self.store.update_token_stats(key, ds.name, pd.DataFrame(stats))


def parse_args() -> RunConfig:
    """Read command-line flags (defaults come from RunConfig).

    Returns:
        The RunConfig for this run. With --list, prints models/datasets/metrics and exits instead.
    """
    # Defaults come from RunConfig, so CLI and code defaults never disagree
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="+" = one or more values; choices = reject unknown names early
    p.add_argument("--models", nargs="+", default=d.models, choices=list(MODEL_SPECS))
    p.add_argument("--datasets", nargs="+", default=d.datasets, choices=DATASETS.names())
    p.add_argument("--langs", nargs="+", default=d.langs, choices=list(LANGUAGES) + [ENGLISH.code])
    p.add_argument("--english", action="store_true", help="also run eng_Latn as a baseline")
    p.add_argument("--metrics", nargs="+", default=d.metrics, choices=METRICS.names())
    p.add_argument("--n-tokens", type=int, default=d.n_tokens,
                   help="token view: draw exactly this many tokens uniformly from all tokens of a language "
                        "(every text is still run; default: use all tokens)")
    p.add_argument("--batch-size", type=int, default=d.batch_size, help="texts per forward pass")
    p.add_argument("--max-texts-sentence", type=int, default=d.max_texts["sentence"])
    p.add_argument("--max-texts-document", type=int, default=d.max_texts["document"])
    p.add_argument("--id-k", type=int, default=d.metric_params["id"]["k"])
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--dtype", default=d.dtype, choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--cache-dir", default=d.cache_dir)
    p.add_argument("--save-points", action="store_true", help="also save each view's vectors (.safetensors, model dtype)")
    p.add_argument("--overwrite", action="store_true", help="replace results produced with other settings")
    p.add_argument("--debug", action="store_true", help="log every forward-pass batch and metric layer")
    p.add_argument("--plot", action="store_true",
                   help="save {out_dir}/{model}/{dataset}/{lang}/{view}/metrics.png (layer vs all metrics) "
                        "and .../{view}/pca3d.png (3-D PCA scatter of every layer)")
    p.add_argument("--list", action="store_true", help="list models, datasets and metrics, then exit")
    a = p.parse_args()
    if a.list:
        print("models:  ", *MODEL_SPECS, "\ndatasets:", *DATASETS.names(), "\nmetrics: ", *METRICS.names())
        raise SystemExit
    langs = a.langs + ([ENGLISH.code] if a.english and ENGLISH.code not in a.langs else [])
    # Copy of the default config with the parsed values filled in
    return dataclasses.replace(
        d, models=a.models, datasets=a.datasets, langs=langs, metrics=a.metrics, n_tokens=a.n_tokens,
        batch_size=a.batch_size,
        max_texts={"sentence": a.max_texts_sentence, "document": a.max_texts_document},
        metric_params={"id": {"k": a.id_k}}, seed=a.seed,
        device=a.device, dtype=a.dtype, out_dir=a.out_dir, cache_dir=a.cache_dir,
        save_points=a.save_points, overwrite=a.overwrite, debug=a.debug, plot=a.plot)


# Runs only when executed as a script (python run.py), not when imported by tests
if __name__ == "__main__":
    # Libraries log at WARNING only; our own logger shows INFO progress lines
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = parse_args()
    # --debug adds per-batch (inference.py) and per-layer lines; child loggers inherit this level
    log.setLevel(logging.DEBUG if cfg.debug else logging.INFO)
    setup_hf_token()
    Experiment(cfg).run()
