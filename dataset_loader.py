"""Datasets: every dataset turns a canonical language code into a list of texts.

To add a dataset, subclass one of the generic bases below (usually 5-10 lines) and register it with
`@DATASETS.register("name")`. The only contract the rest of the pipeline relies on is:
    languages() -> list[str]        canonical codes this dataset has
    source_code(lang) -> str        the dataset's own code for that language (recorded in results)
    load(lang) -> list[str]         cleaned, capped, cached texts
"""
import json
import os
import re
import zlib
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Iterable

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

from languages import ALL_LANGUAGES, Language
from registry import DATASETS

LANG_COLUMN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def match_codes(candidates: Iterable[str]) -> dict[str, str]:
    """Map canonical codes to FLORES-style candidate codes: exact match first, else same language
    in another script (e.g. mni_Mtei -> mni_Beng in Flores+)."""
    candidates = sorted(set(candidates))
    out = {}
    for code, lang in ALL_LANGUAGES.items():
        if code in candidates:
            out[code] = code
        else:
            same_lang = [c for c in candidates if c.startswith(lang.iso3 + "_")]
            if same_lang:
                out[code] = same_lang[0]
    return out


class BaseDataset(ABC):
    name: ClassVar[str]                    # set by @DATASETS.register
    granularity: ClassVar[str] = "sentence"  # "sentence" | "document"
    parallel: ClassVar[bool] = False
    min_chars: ClassVar[int] = 0           # drop shorter texts (never used for parallel data)

    def __init__(self, max_texts: int | None = None, cache_dir: str | Path = "cache/texts", seed: int = 0):
        self.max_texts = max_texts
        self.seed = seed
        tag = f"n{max_texts if max_texts else 'all'}_s{seed}"
        self.cache_dir = Path(cache_dir) / self.name / tag

    # ---- to implement ----
    @abstractmethod
    def available(self) -> dict[str, str]:
        """canonical code -> dataset's own code, for every language the dataset has."""

    @abstractmethod
    def _iter_texts(self, lang: str) -> Iterable[str]:
        """Raw texts for one canonical language, in a fixed (seeded) order."""

    # ---- shared, do not override ----
    def languages(self) -> list[str]:
        return sorted(self._available)

    def source_code(self, lang: str) -> str:
        return self._available[lang]

    @cached_property
    def _available(self) -> dict[str, str]:
        return self.available()

    def load(self, lang: str) -> list[str]:
        if lang not in self._available:
            raise KeyError(f"{self.name} has no language '{lang}'")
        path = self.cache_dir / f"{lang}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for t in self._iter_texts(lang):
            t = t.strip() if isinstance(t, str) else ""
            if len(t) < max(self.min_chars, 1):
                if self.parallel:
                    raise ValueError(f"{self.name}/{lang}: empty text would break the parallel alignment")
                continue
            texts.append(t)
            if self.max_texts and len(texts) >= self.max_texts:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return texts


# =============================== parallel ===============================

class WideParallelDataset(BaseDataset):
    """One HF split, one column per language (IN22). Keeps only rows non-empty in every language."""
    parallel = True
    hf_id: ClassVar[str]
    split: ClassVar[str] = "test"

    @cached_property
    def _columns(self) -> dict[str, list[str]]:
        ds = load_dataset(self.hf_id, split=self.split)
        cols = {c: ds[c] for c in ds.column_names if LANG_COLUMN.match(c)}
        n = len(ds)
        keep = [i for i in range(n) if all(isinstance(v[i], str) and v[i].strip() for v in cols.values())]
        return {c: [v[i] for i in keep] for c, v in cols.items()}

    def available(self) -> dict[str, str]:
        return match_codes(self._columns)

    def _iter_texts(self, lang):
        return self._columns[self.source_code(lang)]


@DATASETS.register("in22-gen")
class IN22Gen(WideParallelDataset):
    hf_id = "ai4bharat/IN22-Gen"


@DATASETS.register("in22-conv")
class IN22Conv(WideParallelDataset):
    hf_id = "ai4bharat/IN22-Conv"


@DATASETS.register("flores-plus")
class FloresPlus(BaseDataset):
    """openlanguagedata/flores_plus (gated): `{split}/{code}.jsonl`, rows aligned by `id`."""
    parallel = True
    repo_id = "openlanguagedata/flores_plus"
    splits: ClassVar[tuple[str, ...]] = ("dev", "devtest")

    def available(self) -> dict[str, str]:
        files = HfApi().list_repo_files(self.repo_id, repo_type="dataset")
        per_split = [{f[len(s) + 1:-len(".jsonl")] for f in files if f.startswith(s + "/") and f.endswith(".jsonl")}
                     for s in self.splits]
        return match_codes(set.intersection(*per_split))   # variant must exist in every split

    def _iter_texts(self, lang):
        for split in self.splits:
            path = hf_hub_download(self.repo_id, f"{split}/{self.source_code(lang)}.jsonl", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            for r in sorted(rows, key=lambda r: r["id"]):
                yield r["text"]


# =============================== monolingual (sampled) ===============================

class HubFilesDataset(BaseDataset):
    """Monolingual corpus stored as per-language files in a Hub repo. Subclasses give the dataset's
    code for a language and the regex of its files; `available()` comes from the repo listing."""
    repo_id: ClassVar[str]
    min_chars = 10

    @abstractmethod
    def dataset_code(self, lang: Language) -> str:
        """The dataset's own code for a language, e.g. 'nep' or '20231101.ne'."""

    @abstractmethod
    def file_pattern(self, code: str) -> str:
        """Regex (full match) for the repo files that hold this language."""

    @cached_property
    def _file_sizes(self) -> dict[str, int]:
        tree = HfApi().list_repo_tree(self.repo_id, repo_type="dataset", recursive=True)
        return {e.path: e.size for e in tree if hasattr(e, "size")}

    def lang_files(self, lang: str) -> list[str]:
        pat = re.compile(self.file_pattern(self.dataset_code(ALL_LANGUAGES[lang])))
        return sorted(f for f in self._file_sizes if pat.fullmatch(f))

    def available(self) -> dict[str, str]:
        return {code: self.dataset_code(l) for code, l in ALL_LANGUAGES.items() if self.lang_files(code)}


class HubParquetDataset(HubFilesDataset):
    """Parquet shards, streamed with `datasets` and shuffled with a fixed seed (shard order + buffer),
    so only a few shards are read."""
    text_field: ClassVar[str] = "text"
    max_shuffle_buffer: ClassVar[int] = 10_000

    def _iter_texts(self, lang):
        urls = [f"hf://datasets/{self.repo_id}/{f}" for f in self.lang_files(lang)]
        ds = load_dataset("parquet", data_files=urls, split="train", streaming=True, columns=[self.text_field])
        buffer = min(self.max_shuffle_buffer, 4 * self.max_texts) if self.max_texts else self.max_shuffle_buffer
        for ex in ds.shuffle(seed=self.seed, buffer_size=buffer):
            yield ex[self.text_field]


class HubTextDataset(HubFilesDataset):
    """Large plain-text files, sampled by seeking to random byte offsets (HTTP range requests).
    This gives a uniform sample over the whole corpus, not just its first lines, and downloads only a
    few MB. Subclasses implement `_records`, which parses texts from a binary file handle positioned
    at an arbitrary byte."""
    records_per_seek: ClassVar[int] = 20
    block_size: ClassVar[int] = 64 * 1024

    @abstractmethod
    def _records(self, fh) -> Iterable[str]:
        """Yield texts from `fh`, starting at an arbitrary byte (skip the partial first record)."""

    def _iter_texts(self, lang):
        if not self.max_texts:
            raise ValueError(f"{self.name} is sampled by random seeks and needs max_texts")
        files = self.lang_files(lang)
        sizes = np.array([self._file_sizes[f] for f in files], dtype=np.float64)
        rng = np.random.default_rng([self.seed, zlib.crc32(lang.encode())])
        n_seeks = int(np.ceil(1.5 * self.max_texts / self.records_per_seek))
        which = rng.choice(len(files), size=n_seeks, p=sizes / sizes.sum())
        fs, seen = HfFileSystem(), set()
        for i in np.unique(which):
            offsets = np.sort(rng.integers(0, int(sizes[i]), size=int((which == i).sum())))
            with fs.open(f"datasets/{self.repo_id}/{files[i]}", "rb", block_size=self.block_size) as fh:
                for off in offsets:
                    fh.seek(int(off))
                    for n, text in enumerate(self._records(fh)):
                        if n >= self.records_per_seek:
                            break
                        if text not in seen:          # overlapping seeks must not duplicate texts
                            seen.add(text)
                            yield text


@DATASETS.register("sangraha-verified")
class SangrahaVerified(HubParquetDataset):
    repo_id = "ai4bharat/sangraha"
    granularity = "document"   # rows are documents (median ~1.4k chars), truncated at max_length
    _codes = {"npi": "nep", "ory": "ori"}

    def dataset_code(self, lang):
        return self._codes.get(lang.iso3, lang.iso3)

    def file_pattern(self, code):
        return rf"verified/{code}/.+\.parquet"


@DATASETS.register("wikipedia")
class Wikipedia(HubParquetDataset):
    repo_id = "wikimedia/wikipedia"
    snapshot = "20231101"
    granularity = "document"

    def dataset_code(self, lang):
        return f"{self.snapshot}.{lang.iso1}"

    def file_pattern(self, code):
        return rf"{re.escape(code)}/.+\.parquet"


# IndicCorp v2 and IITB IndicMonoDoc share a short-code scheme with two non-standard codes.
_INDICCORP_CODES = {"brx": "bd", "doi": "dg"}


@DATASETS.register("indiccorp-v2")
class IndicCorpV2(HubTextDataset):
    """One text file per language (Hindi split in hi-1/2/3), one sentence per line."""
    repo_id = "ai4bharat/IndicCorpV2"

    def dataset_code(self, lang):
        return _INDICCORP_CODES.get(lang.iso1, lang.iso1)

    def file_pattern(self, code):
        return rf"data/{code}(-\d+)?\.txt"

    def _records(self, fh):
        fh.readline()                                  # partial line
        for raw in fh:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                yield line


@DATASETS.register("iitb-indicmonodoc")
class IITBIndicMonoDoc(HubTextDataset):
    """`{code}/shard-N.txt`, documents between <DOC_START> and <DOC_END> lines."""
    repo_id = "cfilt/IITB-IndicMonoDoc"
    granularity = "document"
    records_per_seek = 4

    def dataset_code(self, lang):
        return _INDICCORP_CODES.get(lang.iso1, lang.iso1)

    def file_pattern(self, code):
        return rf"{code}/shard-\d+\.txt"

    def _records(self, fh):
        doc = None                                     # None until the first complete <DOC_START>
        for raw in fh:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "<DOC_START>":
                doc = []
            elif line == "<DOC_END>":
                if doc:
                    yield "\n".join(doc)
                doc = None
            elif doc is not None:
                doc.append(line)
