"""End-to-end on CPU with tiny random models and an in-memory dataset."""
import numpy as np
import pandas as pd
import pytest
import torch

from dataset_loader import BaseDataset
from inference import HiddenStateExtractor
from model_loader import ModelSpec, load_model
from registry import DATASETS
from results import ConfigMismatch
import run as runner

TINY = {
    "tiny-llama": ModelSpec("tiny-llama", "hf-internal-testing/tiny-random-LlamaForCausalLM", "hf"),
    "tiny-st": ModelSpec("tiny-st", "sentence-transformers-testing/stsb-bert-tiny-safetensors", "sentence_transformer"),
}
WORDS = "the a cat dog runs sleeps quickly under over tree house blue green big small".split()


@DATASETS.register("fake")
class FakeDataset(BaseDataset):
    parallel = True

    def available(self):
        return {"hin_Deva": "hi", "tam_Taml": "ta", "tel_Telu": "te"}

    def _iter_texts(self, lang):
        rng = np.random.default_rng(len(lang) + ord(lang[0]))
        for _ in range(60):
            yield " ".join(rng.choice(WORDS, size=rng.integers(3, 25)))


@pytest.fixture(autouse=True)
def tiny_models(monkeypatch):
    for k, v in TINY.items():
        monkeypatch.setitem(runner.MODEL_SPECS, k, v)


@pytest.mark.parametrize("key", list(TINY))
def test_extracted_points_match_unbatched_forward(key):
    model = load_model(TINY[key], "cpu")
    ext = HiddenStateExtractor(model, max_length=64, max_tokens_per_batch=64)   # forces many batches
    texts = list(FakeDataset(cache_dir="/nonexistent")._iter_texts("hin_Deva"))[:20]
    corpus = ext.tokenize(texts)
    points = ext.extract(corpus, 50, np.random.default_rng(0))
    # Unbatched hidden states of every text, to search for each extracted point
    with torch.inference_mode():
        per_text = [model.hidden_states(ids, torch.ones_like(ids))
                    for ids in (torch.from_numpy(x)[None] for x in corpus.ids)]
    for p in range(0, 50, 7):
        # the point must equal the hidden state of one sampled position of some text, at every layer
        found = [(s, q) for s, hs in enumerate(per_text) for q in range(hs[-1].shape[1])
                 if torch.allclose(hs[-1][0, q], points[-1, p], atol=1e-4)]
        assert found, f"point {p} not found in any text"
        s, q = found[0]
        assert corpus.real[s][q]
        for layer, h in enumerate(per_text[s]):
            torch.testing.assert_close(points[layer, p], h[0, q].float(), atol=1e-4, rtol=1e-4)


def test_first_real_token_is_never_sampled():
    model = load_model(TINY["tiny-llama"], "cpu")
    corpus = HiddenStateExtractor(model).tokenize(["the cat runs", "a dog sleeps under the tree"])
    for ids, real in zip(corpus.ids, corpus.real):
        first = next(i for i, t in enumerate(ids) if int(t) not in model.special_ids)
        assert not real[first] and real[first + 1:].all()


def test_run_resume_and_config_guard(tmp_path):
    cfg = runner.RunConfig(models=list(TINY), datasets=["fake"], langs=["hin_Deva", "tam_Taml", "tel_Telu", "urd_Arab"],
                           n_tokens=300, device="cpu", out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    df = runner.ResultStore(cfg.out_dir).load_all()
    assert set(df.model) == set(TINY) and set(df.lang) == {"hin_Deva", "tam_Taml", "tel_Telu"}
    for (m, l), g in df.groupby(["model", "lang"]):
        assert sorted(g.layer) == list(range(len(g)))                   # embeddings + every layer
        assert g[["isoscore", "mev", "avg_cos"]].notna().all().all()
        # ID is NaN only where there are too few distinct points (layer 0 of a 15-word vocabulary)
        assert (g.id_mle.notna() | (g.id_n_unique <= 20)).all() and g.id_mle.iloc[1:].notna().all()
        assert (g.n_tokens == 300).all()
    stats = pd.read_csv(tmp_path / "res" / "tiny-llama" / "fake" / "token_stats.csv")
    assert (stats.status == "done").all()

    files = sorted((tmp_path / "res").glob("*/*/*_*.csv"))
    mtimes = [f.stat().st_mtime_ns for f in files]
    runner.Experiment(cfg).run()                                        # resume: nothing recomputed
    assert [f.stat().st_mtime_ns for f in files] == mtimes

    runner.Experiment(runner.dataclasses.replace(cfg, overwrite=True)).run()
    assert [f.stat().st_mtime_ns for f in files] == mtimes          # same settings: overwrite keeps results

    changed = runner.dataclasses.replace(cfg, n_tokens=200)
    with pytest.raises(ConfigMismatch):
        runner.Experiment(changed).run()
    runner.Experiment(runner.dataclasses.replace(changed, overwrite=True)).run()
    df = runner.ResultStore(cfg.out_dir).load_all()
    assert (df.n_tokens == 200).all() and set(df.lang) == {"hin_Deva", "tam_Taml", "tel_Telu"}


def test_language_with_too_few_tokens_is_skipped(tmp_path):
    cfg = runner.RunConfig(models=["tiny-llama"], datasets=["fake"], langs=["hin_Deva"], n_tokens=10**6,
                           device="cpu", out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    stats = pd.read_csv(tmp_path / "res" / "tiny-llama" / "fake" / "token_stats.csv")
    assert stats.status[0].startswith("skipped")
    assert runner.ResultStore(cfg.out_dir).load_all().empty
