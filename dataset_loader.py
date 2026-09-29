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

# Language column names look like "hin_Deva": 3 lowercase letters, "_", 4-letter script
LANG_COLUMN = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")


def match_codes(candidates: Iterable[str]) -> dict[str, str]:
    """Map canonical codes to a dataset's FLORES-style codes.

    Exact match first, else the same language in another script (e.g. mni_Mtei -> mni_Beng in Flores+).

    Args:
        candidates: language codes the dataset provides.
    Returns:
        {canonical code: dataset code} for every canonical language that was found.
    """
    candidates = sorted(set(candidates))
    out = {}
    for code, lang in ALL_LANGUAGES.items():
        if code in candidates:
            out[code] = code
        else:
            # Fall back to any script of the same language, e.g. "snd_" -> "snd_Arab"
            same_lang = [c for c in candidates if c.startswith(lang.iso3 + "_")]
            if same_lang:
                out[code] = same_lang[0]
    return out


# ABC = abstract base class: cannot be used directly, subclasses must implement the @abstractmethods
class BaseDataset(ABC):
    # ClassVar = setting shared by the class (overridden in subclasses), not per object
    name: ClassVar[str]                    # set by @DATASETS.register
    granularity: ClassVar[str] = "sentence"  # "sentence" | "document"
    parallel: ClassVar[bool] = False
    min_chars: ClassVar[int] = 0           # drop shorter texts (never used for parallel data)

    def __init__(self, max_texts: int | None = None, cache_dir: str | Path = "cache/texts", seed: int = 0):
        """Store settings; nothing is downloaded until load() or languages() is called.

        Args:
            max_texts: keep at most this many texts per language (None = all).
            cache_dir: root folder for cached texts.
            seed: sampling seed for monolingual corpora.
        """
        self.max_texts = max_texts
        self.seed = seed
        # Cache folder depends on max_texts and seed, so different settings never mix
        tag = f"n{max_texts if max_texts else 'all'}_s{seed}"
        self.cache_dir = Path(cache_dir) / self.name / tag

    # ---- to implement ----
    @abstractmethod
    def available(self) -> dict[str, str]:
        """List the languages this dataset has.

        Returns:
            {canonical code: dataset's own code}, e.g. {"npi_Deva": "nep"}.
        """

    @abstractmethod
    def _iter_texts(self, lang: str) -> Iterable[str]:
        """Yield raw texts for one language, in a fixed (seeded) order.

        Args:
            lang: canonical language code.
        Returns:
            Iterable of raw strings (cleaning and capping happen in load()).
        """

    # ---- shared, do not override ----
    def languages(self) -> list[str]:
        """Return the sorted canonical codes this dataset has."""
        return sorted(self._available)

    def source_code(self, lang: str) -> str:
        """Return the dataset's own code for canonical `lang` (e.g. "npi_Deva" -> "nep")."""
        return self._available[lang]

    # cached_property: available() may hit the network, so call it once and remember the result
    @cached_property
    def _available(self) -> dict[str, str]:
        """available(), computed once per object and then reused."""
        return self.available()

    def load(self, lang: str) -> list[str]:
        """Return the cleaned texts of one language, from the cache if possible.

        Args:
            lang: canonical language code.
        Returns:
            List of stripped, non-empty texts (at most max_texts). Also written to the cache on first call.
        Raises KeyError if the dataset lacks `lang`, ValueError if a parallel dataset has an empty text.
        """
        if lang not in self._available:
            raise KeyError(f"{self.name} has no language '{lang}'")
        # Cache hit: every model reads exactly the same texts, without downloading again
        path = self.cache_dir / f"{lang}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        texts = []
        for t in self._iter_texts(lang):
            # Clean: strip whitespace, treat non-strings as empty
            t = t.strip() if isinstance(t, str) else ""
            # Drop empty / too-short texts (for parallel data that would misalign rows, so fail)
            if len(t) < max(self.min_chars, 1):
                if self.parallel:
                    raise ValueError(f"{self.name}/{lang}: empty text would break the parallel alignment")
                continue
            texts.append(t)
            # Stop reading as soon as we have enough texts
            if self.max_texts and len(texts) >= self.max_texts:
                break
        # Write to a temp file, then rename: a crash never leaves a half-written cache
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

    # Load the whole (small) table once and keep it for all languages
    @cached_property
    def _columns(self) -> dict[str, list[str]]:
        """Load the table once and keep rows that are non-empty in every language.

        Returns:
            {column name, e.g. "hin_Deva": list of texts}, all lists aligned row by row.
        """
        ds = load_dataset(self.hf_id, split=self.split)
        # Keep language columns only (drop metadata like "domain", "url")
        cols = {c: ds[c] for c in ds.column_names if LANG_COLUMN.match(c)}
        n = len(ds)
        # Row indices that are non-empty in every language, so all languages stay aligned
        keep = [i for i in range(n) if all(isinstance(v[i], str) and v[i].strip() for v in cols.values())]
        return {c: [v[i] for i in keep] for c, v in cols.items()}

    def available(self) -> dict[str, str]:
        """Return {canonical code: column name} for the language columns found."""
        return match_codes(self._columns)

    def _iter_texts(self, lang):
        """Return the column of `lang` (a list of aligned texts)."""
        return self._columns[self.source_code(lang)]


# A new parallel dataset in the same format only needs its HF id
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
        """Return {canonical code: Flores+ code}, using only variants present in every split."""
        files = HfApi().list_repo_files(self.repo_id, repo_type="dataset")
        # Per split, the set of language codes: "devtest/hin_Deva.jsonl" -> "hin_Deva"
        per_split = [{f[len(s) + 1:-len(".jsonl")] for f in files if f.startswith(s + "/") and f.endswith(".jsonl")}
                     for s in self.splits]
        return match_codes(set.intersection(*per_split))   # variant must exist in every split

    # A generator (uses yield): produces texts one by one instead of building a list
    def _iter_texts(self, lang):
        """Yield the texts of `lang` for each split in turn, sorted by sentence id."""
        for split in self.splits:
            # Download (and cache) one language file of one split
            path = hf_hub_download(self.repo_id, f"{split}/{self.source_code(lang)}.jsonl", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            # Sort by id so sentence i is the same sentence in every language
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
        """The dataset's own code for a language, e.g. 'nep' or '20231101.ne'.

        Args:
            lang: Language entry from languages.py.
        Returns:
            Code used in the repo's file paths.
        """

    @abstractmethod
    def file_pattern(self, code: str) -> str:
        """Regex (full match) for the repo files that hold this language.

        Args:
            code: the dataset's own language code (from dataset_code()).
        Returns:
            Regex string matched against repo file paths.
        """

    # One API call lists every file in the repo with its size (used for listing and sampling)
    @cached_property
    def _file_sizes(self) -> dict[str, int]:
        """List every file of the repo once (one API call).

        Returns:
            {repo file path: size in bytes}.
        """
        tree = HfApi().list_repo_tree(self.repo_id, repo_type="dataset", recursive=True)
        # Folders have no size attribute: keep files only
        return {e.path: e.size for e in tree if hasattr(e, "size")}

    # Files of one language, found by matching the subclass's regex
    def lang_files(self, lang: str) -> list[str]:
        """Return the sorted repo paths of all files belonging to canonical `lang`."""
        pat = re.compile(self.file_pattern(self.dataset_code(ALL_LANGUAGES[lang])))
        return sorted(f for f in self._file_sizes if pat.fullmatch(f))

    # A language is available if at least one of its files exists in the repo
    def available(self) -> dict[str, str]:
        """Return {canonical code: dataset code} for languages with at least one file in the repo."""
        return {code: self.dataset_code(l) for code, l in ALL_LANGUAGES.items() if self.lang_files(code)}


class HubParquetDataset(HubFilesDataset):
    """Parquet shards, streamed with `datasets` and shuffled with a fixed seed (shard order + buffer),
    so only a few shards are read."""
    text_field: ClassVar[str] = "text"
    max_shuffle_buffer: ClassVar[int] = 10_000

    def _iter_texts(self, lang):
        """Stream the parquet shards of `lang` in a seeded shuffled order.

        Args:
            lang: canonical language code.
        Returns:
            Iterator over the text column; load() stops it after max_texts.
        """
        # hf:// URLs let `datasets` stream files directly from the Hub
        urls = [f"hf://datasets/{self.repo_id}/{f}" for f in self.lang_files(lang)]
        # streaming=True: read lazily, never download the whole corpus; read only the text column
        ds = load_dataset("parquet", data_files=urls, split="train", streaming=True, columns=[self.text_field])
        # A buffer of ~4x the texts we need is random enough and keeps reading fast
        buffer = min(self.max_shuffle_buffer, 4 * self.max_texts) if self.max_texts else self.max_shuffle_buffer
        for ex in ds.shuffle(seed=self.seed, buffer_size=buffer):
            yield ex[self.text_field]


class HubTextDataset(HubFilesDataset):
    """Large plain-text files, sampled by seeking to random byte offsets (HTTP range requests).
    This gives a uniform sample over the whole corpus, not just its first lines, and downloads only a
    few MB. Subclasses implement `_records`, which parses texts from a binary file handle positioned
    at an arbitrary byte."""
    records_per_seek: ClassVar[int] = 20     # texts read after each random jump
    block_size: ClassVar[int] = 64 * 1024    # bytes fetched per HTTP request

    @abstractmethod
    def _records(self, fh) -> Iterable[str]:
        """Yield texts from `fh`, starting at an arbitrary byte (skip the partial first record).

        Args:
            fh: binary file handle already positioned by seek().
        Returns:
            Iterator of decoded texts.
        """

    def _iter_texts(self, lang):
        """Yield texts of `lang` read at random byte offsets of its files.

        Offsets are drawn with a seed from (seed, lang), weighted by file size, so every run reads the same
        texts. Duplicate texts from overlapping reads are dropped. Requires max_texts.

        Args:
            lang: canonical language code.
        Returns:
            Iterator of unique texts.
        """
        if not self.max_texts:
            raise ValueError(f"{self.name} is sampled by random seeks and needs max_texts")
        files = self.lang_files(lang)
        sizes = np.array([self._file_sizes[f] for f in files], dtype=np.float64)
        # Stable per-language seed, so the same texts are picked on every run
        rng = np.random.default_rng([self.seed, zlib.crc32(lang.encode())])
        # 1.5x more jumps than strictly needed, to cover duplicates and short records
        n_seeks = int(np.ceil(1.5 * self.max_texts / self.records_per_seek))
        # Which file each jump lands in; bigger files get proportionally more jumps
        which = rng.choice(len(files), size=n_seeks, p=sizes / sizes.sum())
        fs, seen = HfFileSystem(), set()
        for i in np.unique(which):
            # Random byte positions inside file i, sorted so the file is read front to back
            offsets = np.sort(rng.integers(0, int(sizes[i]), size=int((which == i).sum())))
            # Open the remote file like a local one; only the requested blocks are downloaded
            with fs.open(f"datasets/{self.repo_id}/{files[i]}", "rb", block_size=self.block_size) as fh:
                for off in offsets:
                    fh.seek(int(off))
                    # Take up to records_per_seek texts from this position
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
    # Sangraha's folder names differ from ISO codes for Nepali and Odia
    _codes = {"npi": "nep", "ory": "ori"}

    def dataset_code(self, lang):
        """Return Sangraha's folder name: 3-letter code, with 'nep'/'ori' for Nepali/Odia."""
        return self._codes.get(lang.iso3, lang.iso3)

    def file_pattern(self, code):
        """Match the parquet shards under verified/<code>/."""
        return rf"verified/{code}/.+\.parquet"


@DATASETS.register("wikipedia")
class Wikipedia(HubParquetDataset):
    repo_id = "wikimedia/wikipedia"
    snapshot = "20231101"
    granularity = "document"

    _codes = {
        "asm": "as",
        "ben": "bn",
        "guj": "gu",
        "hin": "hi",
        "kan": "kn",
        "kas": "ks",
        "gom": "gom",
        "mal": "ml",
        "mni": "mni",
        "mar": "mr",
        "npi": "ne",
        "ory": "or",
        "pan": "pa",
        "san": "sa",
        "snd": "sd",
        "tam": "ta",
        "tel": "te",
        "urd": "ur",
        "brx": "brx", # bodo is not in Wikipedia
        "sat": "sat",
        "mai": "mai",
        "doi": "doi", # dogri is not in Wikipedia
        "eng": "en",
    }

    def dataset_code(self, lang):
        return f"{self.snapshot}.{self._codes[lang.iso3]}"

    def file_pattern(self, code):
        return rf"{re.escape(code)}/.+\.parquet"


# IndicCorp v2 and IITB IndicMonoDoc share a short-code scheme with two non-standard codes.
_INDICCORP_CODES = {"brx": "bd", "doi": "dg"}


@DATASETS.register("indiccorp-v2")
class IndicCorpV2(HubTextDataset):
    """One text file per language (Hindi split in hi-1/2/3), one sentence per line."""
    repo_id = "ai4bharat/IndicCorpV2"

    def dataset_code(self, lang):
        """Return IndicCorp's short code, e.g. 'hi' ('bd' for Bodo, 'dg' for Dogri)."""
        return _INDICCORP_CODES.get(lang.iso1, lang.iso1)

    def file_pattern(self, code):
        """Match data/<code>.txt and split files like data/hi-1.txt."""
        # Matches "data/hi.txt" and "data/hi-1.txt", "data/hi-2.txt", ...
        return rf"data/{code}(-\d+)?\.txt"

    # One record = one non-empty line
    def _records(self, fh):
        """Yield non-empty lines (one sentence each) after skipping the partial first line."""
        fh.readline()                                  # partial line
        for raw in fh:
            # errors="replace": a broken byte becomes "?" instead of crashing
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                yield line


# IITB repository uses non-standard directory names for Bodo and Dogri.
# English (en) is listed on the dataset card but there is currently no
# `en/` directory in the repository, so the test will report it as skipped.
_IITB_CODES = {
    "brx": "bd",  # Bodo
    "doi": "dg",  # Dogri
}


@DATASETS.register("iitb-indicmonodoc")
class IITBIndicMonoDoc(HubTextDataset):
    """`{code}/shard-N.txt`, documents between <DOC_START> and <DOC_END> lines."""
    repo_id = "cfilt/IITB-IndicMonoDoc"
    granularity = "document"
    records_per_seek = 4

    def dataset_code(self, lang):
        """Return IITB's repository code for the language."""
        return _IITB_CODES.get(lang.iso1, lang.iso1)

    def file_pattern(self, code):
        """Match the shard files <code>/shard-N.txt."""
        return rf"{code}/shard-\d+\.txt"

    # One record = the lines between a <DOC_START> and the next <DOC_END>
    def _records(self, fh):
        """Yield complete documents between <DOC_START> and <DOC_END>."""
        doc = None

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