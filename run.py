"""Layer-wise isotropy of Indic languages: every (model, dataset, language) -> one CSV of per-layer metrics.

    python run.py                                        # all models x all datasets x 22 languages
    python run.py --models gemma-3-1b-pt --datasets in22-gen flores-plus --langs hin_Deva tam_Taml
    python run.py --models embeddinggemma-300m --device cuda:1   # run a second process on GPU 1
    python run.py --list

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
    # Same N for every language, dataset and model (IsoScore depends on N). Languages with fewer
    # eligible tokens are skipped and logged in token_stats.csv.
    n_tokens: int = 16_000
    max_length: int = 512                 # tokens per text; longer documents are truncated
    max_tokens_per_batch: int = 16_384    # padded tokens per forward pass
    # How many texts to take from monolingual corpora (parallel datasets use all rows)
    max_texts: dict[str, int] = field(default_factory=lambda: {"sentence": 2000, "document": 200})
    skip_first_token: bool = True         # drop the attention-sink first token of every text
    metrics: list[str] = field(default_factory=lambda: METRICS.names())
    metric_params: dict[str, dict] = field(default_factory=lambda: {"id": {"k": 20}})
    seed: int = 0
    device: str = "cuda:0"
    dtype: str = "auto"                   # auto = bf16 on Ampere+ GPUs, else fp32
    out_dir: str = "results"
    cache_dir: str = "cache/texts"
    save_points: bool = False             # also dump the sampled vectors to .npy
    overwrite: bool = False               # allow replacing results made with other settings


def make_dataset(name: str, cfg: RunConfig):
    # Look up the dataset class by its registered name, e.g. "in22-gen" -> IN22Gen
    cls = DATASETS.get(name)
    # Parallel sets are small: use every row. Monolingual corpora: cap by granularity
    max_texts = None if cls.parallel else cfg.max_texts[cls.granularity]
    return cls(max_texts=max_texts, cache_dir=cfg.cache_dir, seed=cfg.seed)


class Experiment:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        # Where CSVs go; also answers "is this language already done?"
        self.store = ResultStore(cfg.out_dir)
        # Instantiate metric objects from their names, e.g. ["isoscore", "mev", ...]
        self.metrics = build_metrics(cfg.metrics, cfg.metric_params)
        # Dataset objects are created once and reused for every model
        self.datasets = [make_dataset(n, cfg) for n in cfg.datasets]

    def settings(self, spec, ds) -> dict:
        """Everything that determines the numbers for one (model, dataset) folder."""
        c = self.cfg
        return {
            "model": spec.key, "hf_id": spec.hf_id, "dtype": str(resolve_dtype(c.dtype, c.device)),
            "dataset": ds.name, "max_texts": ds.max_texts, "n_tokens": c.n_tokens, "max_length": c.max_length,
            "skip_first_token": c.skip_first_token, "seed": c.seed,
            "metrics": c.metrics, "metric_params": {m: c.metric_params.get(m, {}) for m in c.metrics},
            # Library versions are saved for reference but not compared on resume
            "versions": {p: md.version(p) for p in ("torch", "transformers", "sentence-transformers", "datasets")},
        }

    def run(self) -> None:
        # Models are the outer loop: loading a model is the expensive step, so do it once
        for key in self.cfg.models:
            spec = MODEL_SPECS[key]
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
                ext = HiddenStateExtractor(model, self.cfg.max_length, self.cfg.max_tokens_per_batch,
                                           self.cfg.skip_first_token)
                for ds in self.datasets:
                    if todo[ds.name]:
                        self.run_dataset(key, ext, ds, todo[ds.name])
            finally:
                model.unload()

    def run_dataset(self, key: str, ext: HiddenStateExtractor, ds, langs: list[str]) -> None:
        n = self.cfg.n_tokens
        stats = []
        for lang in langs:
            # Load the language dataset and tokenize it
            corpus = ext.tokenize(ds.load(lang))
            # Token statistics for token_stats.csv (fertility = tokens per word)
            stat = {"lang": lang, "source_code": ds.source_code(lang), "n_texts": len(corpus.ids),
                    "n_words": corpus.n_words, "n_tokens": corpus.n_nonspecial, "n_eligible": corpus.n_real,
                    "fertility": corpus.n_nonspecial / max(corpus.n_words, 1), "status": "done"}
            stats.append(stat)
            # Too few tokens to sample N: skip instead of lowering N for everyone
            if corpus.n_real < n:
                stat["status"] = f"skipped: {corpus.n_real} eligible tokens < n_tokens={n}"
                log.warning("[%s/%s/%s] %s", key, ds.name, lang, stat["status"])
                continue
            # Seed depends only on (seed, dataset, language): a resumed run samples the same tokens.
            # crc32 is a stable hash (Python's hash() changes between runs)
            rng = np.random.default_rng([self.cfg.seed, zlib.crc32(f"{ds.name}/{lang}".encode())])
            # Run the model; lp.points has shape [layers+1, N, d]
            lp = ext.extract(corpus, n, rng)
            d = lp.points.shape[-1]
            # One row per layer: identifying columns + all metric values (** merges the metric dict in)
            rows = [{"model": key, "dataset": ds.name, "lang": lang, "source_code": stat["source_code"],
                     "layer": i, "n_tokens": n, "d": d,
                     **compute_layer(self.metrics, lp.points[i], lp.text_ids)}
                    for i in range(lp.points.shape[0])]
            self.store.save_lang(key, ds.name, lang, rows)
            if self.cfg.save_points:
                self.store.save_points(key, ds.name, lang, lp.points.cpu().numpy())
            # Progress line with the last layer's metrics
            last = rows[-1]
            log.info("[%s/%s/%s] %d layers | last layer: isoscore=%.4f mev=%.4f avg_cos=%.4f id=%.1f",
                     key, ds.name, lang, len(rows), last.get("isoscore", np.nan), last.get("mev", np.nan),
                     last.get("avg_cos", np.nan), last.get("id_mle", np.nan))
            # Release the large GPU buffer before the next language
            del lp
            free_cuda()
        self.store.update_token_stats(key, ds.name, pd.DataFrame(stats))


def parse_args() -> RunConfig:
    # Defaults come from RunConfig, so CLI and code defaults never disagree
    d = RunConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="+" = one or more values; choices = reject unknown names early
    p.add_argument("--models", nargs="+", default=d.models, choices=list(MODEL_SPECS))
    p.add_argument("--datasets", nargs="+", default=d.datasets, choices=DATASETS.names())
    p.add_argument("--langs", nargs="+", default=d.langs, choices=list(LANGUAGES) + [ENGLISH.code])
    p.add_argument("--english", action="store_true", help="also run eng_Latn as a baseline")
    p.add_argument("--metrics", nargs="+", default=d.metrics, choices=METRICS.names())
    p.add_argument("--n-tokens", type=int, default=d.n_tokens)
    p.add_argument("--max-length", type=int, default=d.max_length)
    p.add_argument("--max-tokens-per-batch", type=int, default=d.max_tokens_per_batch)
    p.add_argument("--max-texts-sentence", type=int, default=d.max_texts["sentence"])
    p.add_argument("--max-texts-document", type=int, default=d.max_texts["document"])
    p.add_argument("--id-k", type=int, default=d.metric_params["id"]["k"])
    p.add_argument("--keep-first-token", action="store_true", help="do not drop the first real token of each text")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default=d.device)
    p.add_argument("--dtype", default=d.dtype, choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--out-dir", default=d.out_dir)
    p.add_argument("--cache-dir", default=d.cache_dir)
    p.add_argument("--save-points", action="store_true", help="also save sampled vectors (float16 .npy)")
    p.add_argument("--overwrite", action="store_true", help="replace results produced with other settings")
    p.add_argument("--list", action="store_true", help="list models, datasets and metrics, then exit")
    a = p.parse_args()
    if a.list:
        print("models:  ", *MODEL_SPECS, "\ndatasets:", *DATASETS.names(), "\nmetrics: ", *METRICS.names())
        raise SystemExit
    langs = a.langs + ([ENGLISH.code] if a.english and ENGLISH.code not in a.langs else [])
    # Copy of the default config with the parsed values filled in
    return dataclasses.replace(
        d, models=a.models, datasets=a.datasets, langs=langs, metrics=a.metrics, n_tokens=a.n_tokens,
        max_length=a.max_length, max_tokens_per_batch=a.max_tokens_per_batch,
        max_texts={"sentence": a.max_texts_sentence, "document": a.max_texts_document},
        metric_params={"id": {"k": a.id_k}}, skip_first_token=not a.keep_first_token, seed=a.seed,
        device=a.device, dtype=a.dtype, out_dir=a.out_dir, cache_dir=a.cache_dir,
        save_points=a.save_points, overwrite=a.overwrite)


# Runs only when executed as a script (python run.py), not when imported by tests
if __name__ == "__main__":
    # Libraries log at WARNING only; our own logger shows INFO progress lines
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    setup_hf_token()
    Experiment(parse_args()).run()
