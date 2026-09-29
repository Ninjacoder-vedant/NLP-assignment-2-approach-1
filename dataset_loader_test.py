#!/usr/bin/env python3
"""
Low-bandwidth Hugging Face dataset integration test.

This test checks:
  * every registered monolingual dataset
  * every language in ALL_LANGUAGES
  * dataset-specific language-code mappings
  * dataset-specific file patterns
  * actual BaseDataset.load() output
  * text cleaning / empty-record handling

Network / disk safeguards:
  * one worker by default
  * temporary HF cache
  * one probe shard per language by default
  * only one text is requested from the real loader
  * never calls the expensive repository-wide _file_sizes scan
  * IndicCorpV2 lists `data/` once because its language files are FLAT:
        data/hi.txt
        data/hi-1.txt
        data/hi-2.txt
  * optional --probe-shards lets the test try another shard only if needed

Usage:
    python test_hf_datasets_lowbandwidth.py

Before running:
    Make sure `dataset_loader.py` is importable from this script's
    directory/environment. Edit the static settings near the top if needed.

NOTE:
This is a smoke/integration test, not a complete corpus audit.
It verifies that each language can resolve real HF files and that the
production `load()` pipeline can successfully return cleaned text.
"""

from __future__ import annotations

import gc
import inspect
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


# ---------------------------------------------------------------------------
# Temporary HF cache.
# Nothing is left in the user's normal HF cache when the test exits.
# ---------------------------------------------------------------------------

_TMP = tempfile.TemporaryDirectory(prefix="hf_dataset_test_")
CACHE_ROOT = Path(_TMP.name)

os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(CACHE_ROOT / "hub")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass
class LanguageResult:
    language: str
    code: str | None = None
    files: int = 0
    probed_files: int = 0
    samples: int = 0
    preview: str = ""
    elapsed_s: float = 0.0
    ok: bool = False
    skipped: bool = False
    reason: str = ""


def log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Static test configuration.
# Edit these values directly instead of passing command-line arguments.
# ---------------------------------------------------------------------------

SAMPLES = 1
PROBE_SHARDS = 1
WORKERS = 6
SEED = 12345
STRICT_COVERAGE = False
PRODUCTION_SETTINGS = False

# dataset_loader.py must be importable from this script's environment.
import dataset_loader as module


# ---------------------------------------------------------------------------
# Registry handling
# ---------------------------------------------------------------------------

def registry_items(registry: Any) -> list[tuple[str, type]]:
    if hasattr(registry, "items"):
        items = list(registry.items())
    elif hasattr(registry, "_items"):
        items = list(registry._items.items())
    elif hasattr(registry, "mapping"):
        items = list(registry.mapping.items())
    else:
        raise TypeError(
            "DATASETS is not a supported dict-like registry. "
            "Expected items(), _items, or mapping."
        )

    return [
        (str(name), cls)
        for name, cls in items
        if inspect.isclass(cls)
    ]


# ---------------------------------------------------------------------------
# Instantiate the REAL BaseDataset.
#
# Unlike the previous test, this calls cls(...), because BaseDataset.__init__
# is available in the supplied source and establishes cache_dir correctly.
# ---------------------------------------------------------------------------

def make_dataset(
    cls: type,
    *,
    samples: int,
    seed: int,
    cache_root: Path,
) -> Any:
    cache_root.mkdir(parents=True, exist_ok=True)

    dataset = cls(
        max_texts=samples,
        cache_dir=cache_root,
        seed=seed,
    )

    return dataset


# ---------------------------------------------------------------------------
# HF metadata discovery
# ---------------------------------------------------------------------------

class RemoteMissingLanguage(Exception):
    """Raised internally when a language path does not exist on HF."""


def path_prefix(dataset_name: str, dataset: Any, language: str) -> str:
    """
    Return the HF path that can be listed efficiently.

    Directory-based repositories:
        Sangraha       verified/<code>
        Wikipedia      20231101.<code>
        IITB           <code>

    FLAT repository:
        IndicCorpV2    data/
            data/hi.txt
            data/hi-1.txt
            ...
    """
    lang_obj = dataset.ALL_LANGUAGES[language]
    code = dataset.dataset_code(lang_obj)

    if dataset_name == "sangraha-verified":
        return f"verified/{code}"

    if dataset_name == "wikipedia":
        return str(code)

    if dataset_name == "iitb-indicmonodoc":
        return str(code)

    if dataset_name == "indiccorp-v2":
        # IMPORTANT:
        # IndicCorp uses files under data/, not directories data/<code>.
        return "data"

    raise ValueError(
        f"Unknown dataset layout for {dataset_name!r}. "
        "Add a rule in path_prefix()."
    )


def discover_files(
    dataset_name: str,
    dataset: Any,
    language: str,
    *,
    api: Any,
    indiccorp_data_cache: list[Any] | None,
) -> tuple[str, dict[str, int], list[Any]]:
    """
    Discover matching files without scanning the whole repository.

    For IndicCorpV2, `data/` is listed once and reused for all languages.
    For other datasets, only the language-specific directory/config is listed.
    """
    lang_obj = dataset.ALL_LANGUAGES[language]
    code = dataset.dataset_code(lang_obj)
    pattern = re.compile(dataset.file_pattern(code))

    if dataset_name == "indiccorp-v2":
        if indiccorp_data_cache is None:
            entries = list(
                api.list_repo_tree(
                    dataset.repo_id,
                    path_in_repo="data",
                    recursive=True,
                    repo_type="dataset",
                )
            )
            indiccorp_data_cache = entries
        else:
            entries = indiccorp_data_cache
    else:
        prefix = path_prefix(dataset_name, dataset, language)

        try:
            entries = list(
                api.list_repo_tree(
                    dataset.repo_id,
                    path_in_repo=prefix,
                    recursive=True,
                    repo_type="dataset",
                )
            )
        except Exception as exc:
            # HF's exact exception type can vary between hub versions.
            # Treat explicit 404 / entry-not-found messages as coverage gaps.
            message = str(exc)
            if (
                "404" in message
                or "Entry Not Found" in message
                or "RemoteEntryNotFoundError" in message
            ):
                raise RemoteMissingLanguage(message) from exc
            raise

    file_sizes: dict[str, int] = {}

    for entry in entries:
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)

        if path is None or size is None:
            continue

        if pattern.fullmatch(path):
            file_sizes[path] = int(size)

    return code, file_sizes, entries


# ---------------------------------------------------------------------------
# Test one real loader.
# ---------------------------------------------------------------------------

def text_preview(text: str, limit: int = 180) -> str:
    """Return a compact one-line preview for visual language verification."""
    preview = " ".join(text.split())
    return preview if len(preview) <= limit else preview[:limit - 3] + "..."


def validate_sample(dataset: Any, language: str, text: Any, index: int) -> None:
    if not isinstance(text, str):
        raise TypeError(
            f"{dataset.name}/{language}: sample {index} has type "
            f"{type(text).__name__}, expected str"
        )

    if not text.strip():
        raise ValueError(
            f"{dataset.name}/{language}: sample {index} is empty"
        )

    minimum = max(getattr(dataset, "min_chars", 0), 1)

    if len(text.strip()) < minimum:
        raise ValueError(
            f"{dataset.name}/{language}: sample {index} has "
            f"{len(text.strip())} chars; minimum is {minimum}"
        )


def load_from_probe_shards(
    dataset_name: str,
    cls: type,
    module: ModuleType,
    language: str,
    all_files: dict[str, int],
    *,
    samples: int,
    seed: int,
    probe_shards: int,
    production_settings: bool,
) -> tuple[list[str], int]:
    """
    Exercise the REAL BaseDataset.load() but use only a tiny subset of files.

    First, all files are discovered and verified.
    Then only up to probe_shards files are exposed to _iter_texts().

    This is the critical bandwidth optimization for Parquet datasets:
    we don't ask datasets to resolve/read every shard just to get one sample.
    """
    if not all_files:
        raise RemoteMissingLanguage(
            f"No matching files for {dataset_name}/{language}"
        )

    ordered_files = sorted(all_files)

    # # Prefer the smallest shards for a cheap smoke test.
    # # The file existence itself was already checked across ALL shards.
    # ordered_files.sort(key=lambda p: (all_files[p], p))
    # selected = ordered_files[:max(1, probe_shards)]

    # Prefer larger shards for the smoke test.
    # We already verify that ALL shards exist; only the probe shard is read.
    ordered_files.sort(key=lambda p: (-all_files[p], p))
    selected = ordered_files[:max(1, probe_shards)]

    cache_root = CACHE_ROOT / "load_cache" / dataset_name / language

    dataset = make_dataset(
        cls,
        samples=samples,
        seed=seed,
        cache_root=cache_root,
    )

    # BaseDataset._available normally comes from available(), which can be
    # an expensive full-HF metadata operation. We already verified this
    # language from targeted discovery, so inject the exact result.
    code = dataset.dataset_code(module.ALL_LANGUAGES[language])
    dataset._available = {language: code}

    # Verify the public source_code() method too.
    if dataset.source_code(language) != code:
        raise AssertionError(
            f"{dataset_name}/{language}: source_code() returned "
            f"{dataset.source_code(language)!r}, expected {code!r}"
        )

    # IMPORTANT:
    # The production _file_sizes property normally recursively lists the
    # entire repository. We replace it with only the tiny probe subset.
    dataset._file_sizes = {
        path: all_files[path]
        for path in selected
    }

    # Keep test-side network traffic tiny.
    if not production_settings:
        if hasattr(dataset, "max_shuffle_buffer"):
            dataset.max_shuffle_buffer = 1

        if hasattr(dataset, "records_per_seek"):
            dataset.records_per_seek = 1

        if hasattr(dataset, "block_size"):
            dataset.block_size = 16 * 1024

    texts = dataset.load(language)

    for idx, text in enumerate(texts):
        validate_sample(dataset, language, text, idx)

    return texts, len(selected)


# ---------------------------------------------------------------------------
# One language
# ---------------------------------------------------------------------------

def check_language(
    dataset_name: str,
    cls: type,
    module: ModuleType,
    language: str,
    *,
    samples: int,
    seed: int,
    probe_shards: int,
    strict_coverage: bool,
    production_settings: bool,
    api: Any,
    indiccorp_data_cache: list[Any] | None,
) -> LanguageResult:
    started = time.perf_counter()
    result = LanguageResult(language=language)

    try:
        # ---------------------------------------------------------------
        # 1. Build a real dataset object
        # ---------------------------------------------------------------
        dataset = make_dataset(
            cls,
            samples=samples,
            seed=seed,
            cache_root=CACHE_ROOT / "metadata_cache" / dataset_name,
        )

        dataset.ALL_LANGUAGES = module.ALL_LANGUAGES

        # ---------------------------------------------------------------
        # 2. Dataset-code mapping
        # ---------------------------------------------------------------
        lang_obj = module.ALL_LANGUAGES[language]
        code = dataset.dataset_code(lang_obj)
        result.code = str(code)

        # ---------------------------------------------------------------
        # 3. Targeted HF file discovery
        # ---------------------------------------------------------------
        try:
            code, files, _ = discover_files(
                dataset_name,
                dataset,
                language,
                api=api,
                indiccorp_data_cache=indiccorp_data_cache,
            )
        except RemoteMissingLanguage as exc:
            result.code = str(code)
            result.skipped = True
            result.reason = "Language path does not exist on HF"
            if strict_coverage:
                result.skipped = False
                result.reason = f"STRICT COVERAGE: {exc}"
            return result

        result.code = str(code)
        result.files = len(files)

        if not files:
            result.skipped = not strict_coverage
            result.reason = (
                "No matching files for this language. "
                "Either the dataset does not provide the language "
                "or the code/file-pattern mapping is wrong."
            )
            if strict_coverage:
                result.skipped = False
            return result

        # ---------------------------------------------------------------
        # 4. Re-run the ACTUAL lang_files() against our narrow metadata.
        # ---------------------------------------------------------------
        dataset._file_sizes = files

        actual_files = dataset.lang_files(language)

        if set(actual_files) != set(files):
            raise AssertionError(
                "lang_files() did not return exactly the files discovered "
                "from Hugging Face:\n"
                f"  discovered={sorted(files)[:10]}\n"
                f"  actual={actual_files[:10]}"
            )

        # ---------------------------------------------------------------
        # 5. Use the REAL BaseDataset.load().
        #
        # This verifies:
        #   _available
        #   source_code()
        #   _iter_texts()
        #   stripping
        #   empty-text filtering
        #   min_chars
        #   max_texts
        #   cache writing
        # ---------------------------------------------------------------
        texts, probed = load_from_probe_shards(
            dataset_name,
            cls,
            module,
            language,
            files,
            samples=samples,
            seed=seed,
            probe_shards=probe_shards,
            production_settings=production_settings,
        )

        result.probed_files = probed
        result.samples = len(texts)

        if texts:
            result.preview = text_preview(texts[0])

        if not texts:
            raise AssertionError(
                "HF files were found and load() completed, "
                "but no valid text was returned"
            )

        result.ok = True
        return result

    except RemoteMissingLanguage as exc:
        result.skipped = not strict_coverage
        result.reason = "Language path does not exist on HF"
        if strict_coverage:
            result.skipped = False
            result.reason = f"STRICT COVERAGE: {exc}"
        return result

    except Exception as exc:
        result.reason = f"{type(exc).__name__}: {exc}"
        return result

    finally:
        result.elapsed_s = time.perf_counter() - started
        gc.collect()


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(
    module: ModuleType,
    dataset_name: str,
    cls: type,
    *,
    samples: int,
    seed: int,
    probe_shards: int,
    workers: int,
    strict_coverage: bool,
    production_settings: bool,
) -> list[LanguageResult]:

    languages = sorted(module.ALL_LANGUAGES)

    log("\n" + "=" * 76)
    log(f"DATASET : {dataset_name}")
    log(f"CLASS   : {cls.__name__}")
    log(f"REPO    : {getattr(cls, 'repo_id', getattr(cls, 'hf_id', '<unknown>'))}")
    log(f"MODE    : {'production' if production_settings else 'low-bandwidth'}")
    log("=" * 76)

    # One HfApi object can be reused.
    from huggingface_hub import HfApi
    api = HfApi()

    # IndicCorp uses one flat data/ directory.
    # Cache its listing so 20+ languages don't repeat the same API call.
    indiccorp_data_cache: list[Any] | None = None

    results: list[LanguageResult] = []

    # Default is ONE worker.
    # More workers can increase instantaneous bandwidth substantially.
    worker_count = max(1, min(workers, len(languages)))

    def run(lang: str) -> LanguageResult:
        return check_language(
            dataset_name,
            cls,
            module,
            lang,
            samples=samples,
            seed=seed,
            probe_shards=probe_shards,
            strict_coverage=strict_coverage,
            production_settings=production_settings,
            api=api,
            indiccorp_data_cache=indiccorp_data_cache,
        )

    # IndicCorp cache needs to be initialized once before parallel calls.
    if dataset_name == "indiccorp-v2":
        lang_obj = module.ALL_LANGUAGES[languages[0]]
        tmp_dataset = make_dataset(
            cls,
            samples=samples,
            seed=seed,
            cache_root=CACHE_ROOT / "_metadata",
        )
        try:
            from huggingface_hub import HfApi
            indiccorp_data_cache = list(
                HfApi().list_repo_tree(
                    tmp_dataset.repo_id,
                    path_in_repo="data",
                    recursive=True,
                    repo_type="dataset",
                )
            )
        except Exception as exc:
            log(f"FATAL: unable to list IndicCorpV2 data/: {exc}")
            return [
                LanguageResult(
                    language=lang,
                    skipped=False,
                    reason=f"{type(exc).__name__}: {exc}",
                )
                for lang in languages
            ]

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(run, lang): lang for lang in languages}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.ok:
                log(
                    f"  PASS  {result.language:<10} "
                    f"code={str(result.code):<14} "
                    f"files={result.files:<4} "
                    f"probe={result.probed_files:<2} "
                    f"samples={result.samples:<2} "
                    f"{result.elapsed_s:.2f}s"
                )
                log(f"        TEXT: {result.preview}")
            elif result.skipped:
                log(
                    f"  SKIP  {result.language:<10} "
                    f"code={str(result.code):<14} "
                    f"reason={result.reason}"
                )
            else:
                log(
                    f"  FAIL  {result.language:<10} "
                    f"code={str(result.code):<14} "
                    f"reason={result.reason}"
                )

    return sorted(results, key=lambda r: r.language)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    results_by_dataset: dict[str, list[LanguageResult]],
    *,
    strict_coverage: bool,
) -> int:
    log("\n" + "#" * 76)
    log("SUMMARY")
    log("#" * 76)

    total_pass = 0
    total_skip = 0
    total_fail = 0

    for dataset_name, results in results_by_dataset.items():
        passed = sum(r.ok for r in results)
        skipped = sum(r.skipped for r in results)
        failed = sum(not r.ok and not r.skipped for r in results)

        total_pass += passed
        total_skip += skipped
        total_fail += failed

        status = "PASS" if failed == 0 else "FAIL"

        log(
            f"{status:<5} {dataset_name:<24} "
            f"pass={passed:<3} "
            f"skip={skipped:<3} "
            f"fail={failed:<3}"
        )

    log("-" * 76)
    log(
        f"TOTAL: pass={total_pass}, "
        f"skip={total_skip}, "
        f"fail={total_fail}"
    )

    if total_fail:
        log("\nFailures:")
        for dataset_name, results in results_by_dataset.items():
            for result in results:
                if not result.ok and not result.skipped:
                    log(
                        f"  {dataset_name}/{result.language}: "
                        f"{result.reason}"
                    )

    if total_skip:
        log("\nMissing-language coverage:")
        for dataset_name, results in results_by_dataset.items():
            for result in results:
                if result.skipped:
                    log(
                        f"  {dataset_name}/{result.language}: "
                        f"{result.reason}"
                    )

    log(f"\nTemporary cache: {CACHE_ROOT}")
    log("Temporary cache is deleted automatically.")

    return 1 if total_fail else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not hasattr(module, "DATASETS"):
        log("ERROR: DATASETS registry not found")
        return 2

    if not hasattr(module, "ALL_LANGUAGES"):
        log("ERROR: ALL_LANGUAGES not found")
        return 2

    try:
        entries = registry_items(module.DATASETS)
    except Exception as exc:
        log(f"ERROR reading DATASETS: {exc}")
        return 2

    if not entries:
        log("ERROR: DATASETS registry is empty")
        return 2

    # Some dataset implementations reference ALL_LANGUAGES as a module global.
    module.__dict__["ALL_LANGUAGES"] = module.ALL_LANGUAGES

    log("Low-bandwidth HF dataset integration test")
    log(f"Registered datasets : {len(entries)}")
    log(f"Languages/dataset   : {len(module.ALL_LANGUAGES)}")
    log(f"Samples/language    : {SAMPLES}")
    log(f"Probe shards        : {PROBE_SHARDS}")
    log(f"Workers             : {WORKERS}")
    log(f"Strict coverage     : {STRICT_COVERAGE}")
    log(f"Production settings : {PRODUCTION_SETTINGS}")
    log(f"Temporary cache     : {CACHE_ROOT}")

    results_by_dataset: dict[str, list[LanguageResult]] = {}

    for dataset_name, cls in entries:
        try:
            results_by_dataset[dataset_name] = run_dataset(
                module,
                dataset_name,
                cls,
                samples=SAMPLES,
                seed=SEED,
                probe_shards=PROBE_SHARDS,
                workers=WORKERS,
                strict_coverage=STRICT_COVERAGE,
                production_settings=PRODUCTION_SETTINGS,
            )
        except Exception as exc:
            log(
                f"\nFATAL while testing {dataset_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            return 2

    return print_summary(
        results_by_dataset,
        strict_coverage=STRICT_COVERAGE,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        _TMP.cleanup()
