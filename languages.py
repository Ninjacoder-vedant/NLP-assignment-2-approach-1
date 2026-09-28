"""Canonical language table. Canonical codes are the IN22 column names (FLORES-style `xxx_Scrp`).

Each dataset maps a canonical code to its own naming (`BaseDataset.available()`); nothing else in the
code base needs to know dataset-specific codes.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str   # canonical, e.g. "hin_Deva"; used for CLI args and result files
    name: str
    iso1: str   # short code used by Wikipedia (ISO 639-1 where it exists, else 639-3)

    # 3-letter language part of the code: "hin_Deva" -> "hin"
    @property
    def iso3(self) -> str:
        return self.code.split("_")[0]


# The 22 scheduled Indian languages, keyed by canonical code
LANGUAGES: dict[str, Language] = {l.code: l for l in [
    Language("asm_Beng", "Assamese", "as"),
    Language("ben_Beng", "Bengali", "bn"),
    Language("brx_Deva", "Bodo", "brx"),
    Language("doi_Deva", "Dogri", "doi"),
    Language("gom_Deva", "Konkani", "gom"),
    Language("guj_Gujr", "Gujarati", "gu"),
    Language("hin_Deva", "Hindi", "hi"),
    Language("kan_Knda", "Kannada", "kn"),
    Language("kas_Arab", "Kashmiri", "ks"),
    Language("mai_Deva", "Maithili", "mai"),
    Language("mal_Mlym", "Malayalam", "ml"),
    Language("mar_Deva", "Marathi", "mr"),
    Language("mni_Mtei", "Manipuri", "mni"),
    Language("npi_Deva", "Nepali", "ne"),
    Language("ory_Orya", "Odia", "or"),
    Language("pan_Guru", "Punjabi", "pa"),
    Language("san_Deva", "Sanskrit", "sa"),
    Language("sat_Olck", "Santali", "sat"),
    Language("snd_Deva", "Sindhi", "sd"),
    Language("tam_Taml", "Tamil", "ta"),
    Language("tel_Telu", "Telugu", "te"),
    Language("urd_Arab", "Urdu", "ur"),
]}

# Optional high-resource baseline; not part of the default 22.
ENGLISH = Language("eng_Latn", "English", "en")
ALL_LANGUAGES: dict[str, Language] = {**LANGUAGES, ENGLISH.code: ENGLISH}
