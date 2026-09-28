# Layer-wise isotropy pipeline: design

## Context
[dataset.py](Code/dataset.py) is one script that hard-codes one model (embeddinggemma), one dataset (IN22-Gen) and the metrics. The goal is to run **6 models × 7 datasets × 22 Indic languages**, with layer-wise IsoScore, AvgCos, ID and MEV for each (model, dataset, language). It should be modular and resumable, and adding a model, dataset or metric should mean adding one small class or config entry without touching the rest.

The core logic in `dataset.py` already works and is kept: sampling real tokens (no special or pad tokens), picking the same N for every language, length bucketing, gathering hidden states into a preallocated GPU buffer, MLE ID and AvgCos over pairs from different sentences. It is moved into the modules below.

## Layout (flat, uses your file names)
```
Code/
  languages.py         # 22 canonical languages + code conversions
  registry.py          # tiny Registry (name -> class/spec), about 20 lines
  dataset_loader.py    # BaseDataset + 3 generic bases + 7 concrete datasets
  model_loader.py      # ModelSpec table + 2 wrappers behind one interface
  inference.py         # tokenization, token sampling, hidden-state extraction
  isotropy_metrics.py  # LayerStats (shared cache) + 4 metric classes
  results.py           # ResultStore (per-language files, resume, aggregate)
  run.py               # RunConfig + Experiment loop + argparse CLI
  tests/test_metrics.py, tests/test_smoke.py
```
Data flow: `Dataset.load(lang) -> list[str]` → `Extractor.tokenize` → `Extractor.extract -> LayerPoints [L+1, N, d]` → for each layer, `LayerStats` → `metrics` → `ResultStore.save`.

The 4 abstractions are Dataset, ModelWrapper, Metric and ResultStore. They only communicate through plain types: `list[str]`, tensors and `dict[str, float]`. That is why adding one never breaks the others.

---

## 1. `languages.py`
```python
@dataclass(frozen=True)
class Language:
    code: str            # canonical FLORES-style: "hin_Deva" (used everywhere: files, CLI)
    iso1: str | None     # "hi" (Wikipedia, IndicCorp)
    name: str
LANGUAGES: dict[str, Language]   # the 22: asm_Beng ben_Beng brx_Deva doi_Deva gom_Deva guj_Gujr
                                 # hin_Deva kan_Knda kas_Arab mai_Deva mal_Mlym mar_Deva mni_Mtei
                                 # npi_Deva ory_Orya pan_Guru san_Deva sat_Olck snd_Deva tam_Taml
                                 # tel_Telu urd_Arab   (eng_Latn optional as a baseline)
```
Each dataset converts the canonical code into its own naming. The rest of the code only ever sees canonical codes.

## 2. `registry.py`
```python
class Registry:
    def register(self, name) -> decorator     # @DATASETS.register("in22-gen")
    def get(self, name) ; def names()
DATASETS, MODEL_FAMILIES, METRICS = Registry(), Registry(), Registry()
```

## 3. `dataset_loader.py`
```python
class BaseDataset(ABC):
    name: ClassVar[str]; granularity: ClassVar[str]   # "sentence" | "document"
    def __init__(self, max_texts: int | None, cache_dir: Path, seed: int = 0)
    @abstractmethod
    def languages(self) -> list[str]                  # canonical codes this dataset HAS
    @abstractmethod
    def _iter_texts(self, lang: str) -> Iterable[str] # raw, dataset-specific
    def load(self, lang) -> list[str]:                # FINAL (never overridden):
        # cache hit -> read cache/texts/{name}/{lang}.jsonl
        # else: strip, drop empty, cap at max_texts, write cache
```
The text cache means every model sees exactly the same texts, and each corpus is downloaded or streamed only once.

Three generic bases do the heavy lifting. Concrete datasets are about 5 to 10 lines each:

| Base | Mechanism | Concrete classes |
|---|---|---|
| `WideParallelDataset` | one HF split, one column per language; keeps rows that are non-empty in **all** languages (computed once, cached) | `IN22Gen`, `IN22Conv` |
| `PerLangConfigParallelDataset` | one HF config per language, aligned by `id` | `FloresPlus` (`openlanguagedata/flores_plus`, gated) |
| `StreamingMonolingualDataset` | `load_dataset(..., streaming=True)`, `.shuffle(seed, buffer)`, `.take(max_texts)`. Subclass implements `hf_kwargs(lang)` (`name` / `data_dir` / `data_files`) and `text_field` | `SangrahaVerified`, `IndicCorpV2`, `Wikipedia` (`wikimedia/wikipedia`, `20231101.{iso1}`), `IITBIndicMonoDoc` |

`languages()` returns only what exists. For example, Wikipedia has no Bodo or Dogri edition, and the runner skips those.

## 4. `model_loader.py`
```python
@dataclass(frozen=True)
class ModelSpec:
    key: str                 # "embeddinggemma-300m" (folder name in results)
    hf_id: str
    family: str              # "sentence_transformer" | "hf"
    auto_class: str = "AutoModel"   # "AutoModelForMultimodalLM" for Qwen3.5
    dtype: str = "bfloat16"  # embeddinggemma: no fp16

MODEL_SPECS = {s.key: s for s in [
  ModelSpec("embeddinggemma-300m", "google/embeddinggemma-300m", "sentence_transformer"),
  ModelSpec("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", "sentence_transformer"),
  ModelSpec("harrier-oss-v1-0.6b", "microsoft/harrier-oss-v1-0.6b", "sentence_transformer"),
  ModelSpec("gemma-3-1b-pt", "google/gemma-3-1b-pt", "hf"),
  ModelSpec("llama-3.2-1b", "meta-llama/Llama-3.2-1B", "hf"),
  ModelSpec("qwen3.5-0.8b-base", "Qwen/Qwen3.5-0.8B-Base", "hf", auto_class="AutoModelForMultimodalLM"),
]}

class ModelWrapper(ABC):          # the ONLY interface inference.py uses
    tokenizer; device; max_length: int
    special_ids: set[int]; pad_id: int
    @abstractmethod
    def hidden_states(self, input_ids, attention_mask) -> tuple[Tensor, ...]  # (L+1) x [B,T,d]
    def unload(self): del model; gc.collect(); torch.cuda.empty_cache()

@MODEL_FAMILIES.register("sentence_transformer")
class SentenceTransformerWrapper(ModelWrapper):   # backbone = st[0].auto_model, st.tokenizer, st.max_seq_length
@MODEL_FAMILIES.register("hf")
class HFTransformerWrapper(ModelWrapper):         # getattr(transformers, spec.auto_class); AutoTokenizer
def load_model(spec, device) -> ModelWrapper
```
Design decisions:
- **Base `AutoModel`, not `...ForCausalLM`**, for gemma and llama. No LM head is loaded, which saves memory (the LM head is large at 256k vocab for Gemma), and the hidden states are identical.
- **Always right padding** (set in the collate, ignoring `tokenizer.padding_side`). Real-token positions are then 0..L-1 for every model. Left padding shifts position ids in some models, and Qwen3-Embedding defaults to left.
- **Raw text, no instruction prompts** (no `task: ... | query:`). This keeps the models comparable.
- The number of layers and `d` are read from the output shapes (`len(hidden_states)`, `hs.shape[-1]`), not from the config. Multimodal configs nest these values (`text_config`), so reading shapes is more robust.
- HF token: `os.environ["HF_TOKEN"]`. A `utils.get_hf_token()` tries Kaggle secrets and falls back to env, so the code runs outside Kaggle.

## 5. `inference.py`
```python
@dataclass
class TokenizedCorpus:
    ids: list[list[int]]; real: list[np.ndarray]   # bool mask: not special
    n_real: int; n_words: int                      # -> fertility stats

@dataclass
class LayerPoints:
    points: Tensor     # [L+1, N, d] float32 on device (preallocated)
    text_ids: Tensor   # [N] which text each token came from (for AvgCos)

class HiddenStateExtractor:
    def __init__(self, model: ModelWrapper, max_length: int, max_tokens_per_batch: int)
    def tokenize(self, texts) -> TokenizedCorpus
    def extract(self, corpus, n_tokens: int, rng) -> LayerPoints
```
Efficiency changes compared with `dataset.py`:
1. **Token-budget batching** (`max_tokens_per_batch`, e.g. 16k) instead of a fixed `BATCH_SIZE`. Sentences (about 30 tokens) and documents (512 tokens) then both use the GPU well.
2. **Skip texts with zero sampled tokens.** They never reach the model. This matters a lot for native corpora, where N ≪ total tokens.
3. Only the selected tokens are gathered from each layer, straight into the preallocated buffer, under `torch.inference_mode()`.
4. Documents and articles are truncated at `max_length` (config, default 512).

## 6. `isotropy_metrics.py`
```python
class LayerStats:                       # lazy, cached: each is computed at most once per layer
    def __init__(self, X: Tensor, text_ids: Tensor, gen: torch.Generator)
    @cached_property eigvals            # of cov(X), float64, eigvalsh (on GPU)
    @cached_property X_unit             # L2-normalized rows
    @cached_property knn_dists          # chunked cdist + topk

class IsotropyMetric(ABC):
    name: ClassVar[str]
    @abstractmethod
    def compute(self, s: LayerStats) -> dict[str, float]   # dict -> one metric may emit >1 column

@METRICS.register("isoscore") class IsoScore        # from s.eigvals (Rudman et al. 2022 closed form)
@METRICS.register("avgcos")   class AvgRandomCosine # n_pairs random pairs, different texts only -> {"avg_cos"}
@METRICS.register("id")       class IntrinsicDim    # Levina–Bickel MLE, k=20 -> {"id_mle", "id_score": id_mle/d}
@METRICS.register("mev")      class MEV             # s.eigvals[-1] / s.eigvals.sum()
```
IsoScore and MEV both come from the eigenvalues of the covariance (IsoScore's PCA-reoriented variance vector *is* the eigenvalue vector). One `eigvalsh` on the GPU therefore serves both. It replaces the CPU float64 numpy call to the `IsoScore` library, which is the current bottleneck. The official library is kept as a **test oracle** to confirm the results match.

AvgCos stores the raw mean cosine. Transforms such as `1 - |cos|` are left to analysis so the stored value has one clear meaning.

## 7. `results.py`
```
results/{model_key}/{dataset}/{lang}.csv        # rows = layers; cols = model,dataset,lang,layer,n_tokens,<metric cols>
results/{model_key}/{dataset}/token_stats.csv   # per-lang real tokens, words, fertility
results/{model_key}/{dataset}/run_config.json   # N, max_length, seed, k, n_pairs, lib versions
```
```python
class ResultStore:
    def __init__(self, root)
    def exists(self, model, dataset, lang) -> bool      # resume / skip
    def save_lang(self, model, dataset, lang, rows)      # write tmp, then os.replace (atomic)
    def save_meta(self, model, dataset, token_stats, config)
    def load_all(self) -> pd.DataFrame                   # glob + concat -> one tidy table for plots
```
Each language gets its own file, so a crash (a Kaggle timeout, for example) loses at most one language. A rerun then continues where it stopped.

## 8. `run.py`
```python
@dataclass
class RunConfig:
    models: list[str]; datasets: list[str]; langs: list[str] = ALL_22
    n_tokens_cap: int = 20_000        # >= 10*d even for d=2048 (Llama)
    max_length: int = 512; max_tokens_per_batch: int = 16_384
    max_texts: dict[str, int]         # e.g. sentence corpora 5000, document corpora 1000
    n_pairs: int = 200_000; id_k: int = 20; seed: int = 0
    out_dir: str = "results"; save_points: bool = False

class Experiment:
    def run(self):
        for spec in models:                       # outer: models are the expensive thing to load
            model = load_model(spec); ext = HiddenStateExtractor(model, ...)
            for ds in datasets:
                langs = [l for l in ds.languages() if l in cfg.langs]
                corpora = {l: ext.tokenize(ds.load(l)) for l in langs}    # pass 1 (cheap)
                N = min(cfg.n_tokens_cap, min(c.n_real for c in corpora.values()))  # same N for all langs
                store.save_meta(...)
                for l in langs:
                    if store.exists(spec.key, ds.name, l): continue
                    rng = np.random.default_rng([cfg.seed, zlib.crc32(f"{ds.name}/{l}".encode())])
                    lp = ext.extract(corpora[l], N, rng)                    # pass 2
                    rows = [{"layer": i, **merge(m.compute(LayerStats(lp.points[i], ...)) for m in metrics)}
                            for i in range(lp.points.shape[0])]
                    store.save_lang(...); del lp; free_cuda()
            model.unload()
```
CLI: `python run.py --models all --datasets in22-gen flores-plus --langs hin_Deva tam_Taml --n-tokens 20000`.

The per-(dataset, lang) seed comes from a **stable hash**, so a resumed run produces the same numbers as an uninterrupted one. Pass 1 always tokenizes all languages, so N does not change on resume.

## Extending (what is touched)
| Add | Change |
|---|---|
| Model from an existing family | 1 `ModelSpec` line |
| New model family (e.g. T5 encoder) | 1 `ModelWrapper` subclass + `@MODEL_FAMILIES.register` |
| Dataset in an existing format | about 8-line subclass of one of the 3 bases + `@DATASETS.register` |
| Metric | 1 `IsotropyMetric` subclass + `@METRICS.register` (can reuse `LayerStats` caches) |

## Things to verify during implementation
- Exact HF ids, configs and dirs, and the per-language code conventions, for Sangraha (`verified/<code>`), IndicCorpV2, and IITB-IndicMonoDoc. Map these in each subclass's `hf_kwargs`.
- Qwen3.5-0.8B-Base: text-only forward via `AutoModelForMultimodalLM` returns `hidden_states` for the language model.
- Gated models and datasets (Gemma, Llama, Flores+): accept their terms on HF with the token's account.

## Verification
1. `tests/test_metrics.py`, on synthetic data:
   - Isotropic Gaussian (n ≫ d): IsoScore ≈ 1, MEV ≈ 1/d, avg_cos ≈ 0.
   - Rank-1 data: IsoScore ≈ 0, MEV ≈ 1.
   - Data on a 5-dimensional subspace: id_mle ≈ 5.
   - Torch IsoScore equals `IsoScore.IsoScore` (the official library) to 1e-6.
2. `tests/test_smoke.py`: a `FakeDataset` (3 languages, 50 sentences) and a tiny model (`hf-internal-testing/tiny-random-LlamaForCausalLM` via the `hf` family) run end-to-end on CPU. Checks: the output files exist, there are L+1 rows per language, and a rerun skips everything (resume works).
3. Parity: `python run.py --models embeddinggemma-300m --datasets in22-gen` gives the same numbers as the old [dataset.py](Code/dataset.py), within sampling noise, for a few languages.
4. The full run, on GPU and per model, is then resumable across Kaggle sessions.
