"""End-to-end on CPU with tiny random models and an in-memory dataset."""
import numpy as np
import pandas as pd
import pytest
import torch
from safetensors import safe_open

from dataset_loader import BaseDataset
from inference import VIEWS, HiddenStateExtractor
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


def unbatched_reference(model, texts):
    """Per text and layer: [n_kept, d] hidden states of its non-special tokens, from one unpadded forward each."""
    special = torch.tensor(sorted(model.special_ids))
    ref = []
    with torch.inference_mode():
        for t in texts:
            ids = model.tokenizer(t, return_tensors="pt")["input_ids"]
            keep = ~torch.isin(ids[0], special)
            ref.append([h[0, keep].float() for h in model.hidden_states(ids, torch.ones_like(ids))])
    return ref


@pytest.mark.parametrize("key", list(TINY))
def test_extracted_views_match_unbatched_forward(key):
    model = load_model(TINY[key], "cpu")
    ext = HiddenStateExtractor(model, batch_size=3)   # several padded batches
    texts = list(FakeDataset(cache_dir="/nonexistent")._iter_texts("hin_Deva"))[:20]
    ref = unbatched_reference(model, texts)
    n_layers = len(ref[0])
    # count_tokens (tokenizer only) predicts exactly how many tokens extract() sees per text
    assert ext.count_tokens(texts).tolist() == [len(r[0]) for r in ref]
    # Token view without indices = every token, in text order
    views = ext.extract(texts)
    assert set(views) == set(VIEWS)
    for layer in range(n_layers):
        all_tok = torch.cat([r[layer] for r in ref])
        torch.testing.assert_close(views["token"][layer].float(), all_tok, atol=1e-4, rtol=1e-4)
        # Sentence views: mean of each text's tokens, and its last token
        torch.testing.assert_close(views["sentence-mean"][layer], torch.stack([r[layer].mean(0) for r in ref]),
                                   atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(views["sentence-last"][layer], torch.stack([r[layer][-1] for r in ref]),
                                   atol=1e-4, rtol=1e-4)
    # Token view with indices = exactly those rows of the combined token matrix, the same at every layer
    idx = np.sort(np.random.default_rng(0).choice(len(views["token"][0]), 50, replace=False))
    picked = ext.extract(texts, idx)
    assert picked["token"].shape == (n_layers, 50, views["token"].shape[-1])
    torch.testing.assert_close(picked["token"], views["token"][:, idx])
    torch.testing.assert_close(picked["sentence-mean"], views["sentence-mean"])
    # Indices beyond the combined matrix are an error, not silently fewer tokens
    with pytest.raises(RuntimeError):
        ext.extract(texts, np.array([len(views["token"][0])]))


def test_run_resume_and_config_guard(tmp_path):
    cfg = runner.RunConfig(models=list(TINY), datasets=["fake"], langs=["hin_Deva", "tam_Taml", "tel_Telu", "urd_Arab"],
                           n_tokens=300, device="cpu", out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    df = runner.ResultStore(cfg.out_dir).load_all()
    assert set(df.model) == set(TINY) and set(df.lang) == {"hin_Deva", "tam_Taml", "tel_Telu"}
    assert set(df.view) == set(VIEWS)
    for (m, l, v), g in df.groupby(["model", "lang", "view"]):
        assert sorted(g.layer) == list(range(len(g)))                   # embeddings + every layer
        assert g.sort_values("layer").layer_name.iloc[0] == "embeddings"
        assert g[["isoscore", "mev", "avg_cos"]].notna().all().all()
        # Token view: N tokens; sentence views: one point per text (FakeDataset has 60)
        assert (g.n_points == (300 if v == "token" else 60)).all()
    tok = df[df.view == "token"]
    # ID is NaN only where there are too few distinct points (layer 0 of a 15-word vocabulary)
    assert (tok.id_mle.notna() | (tok.id_n_unique <= 20)).all() and tok[tok.layer > 0].id_mle.notna().all()
    stats = pd.read_csv(tmp_path / "res" / "tiny-llama" / "fake" / "token_stats.csv")
    assert (stats.status == "done").all() and (stats.n_tokens_used == 300).all()

    files = sorted((tmp_path / "res").glob("*/*/*/metrics.csv"))
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
    assert (df[df.view == "token"].n_points == 200).all() and set(df.lang) == {"hin_Deva", "tam_Taml", "tel_Telu"}


def test_language_with_too_few_tokens_is_skipped(tmp_path):
    cfg = runner.RunConfig(models=["tiny-llama"], datasets=["fake"], langs=["hin_Deva"], n_tokens=10**6,
                           device="cpu", out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    stats = pd.read_csv(tmp_path / "res" / "tiny-llama" / "fake" / "token_stats.csv")
    assert stats.status[0].startswith("skipped")
    assert runner.ResultStore(cfg.out_dir).load_all().empty


def test_no_n_tokens_uses_all_tokens(tmp_path):
    cfg = runner.RunConfig(models=["tiny-llama"], datasets=["fake"], langs=["hin_Deva"], n_tokens=None,
                           device="cpu", out_dir=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    stats = pd.read_csv(tmp_path / "res" / "tiny-llama" / "fake" / "token_stats.csv")
    df = runner.ResultStore(cfg.out_dir).load_all()
    assert (stats.status == "done").all() and not df.empty
    assert (df[df.view == "token"].n_points == stats.n_tokens[0]).all()


@pytest.mark.parametrize("plot", [False, True])
def test_layout_pca3d_points_and_plots(tmp_path, plot):
    cfg = runner.RunConfig(models=["tiny-llama"], datasets=["fake"], langs=["hin_Deva"], n_tokens=300, plot=plot,
                           save_points=True, device="cpu", out_dir=str(tmp_path / "res"),
                           cache_dir=str(tmp_path / "cache"))
    runner.Experiment(cfg).run()
    folder = tmp_path / "res" / "tiny-llama" / "fake" / "hin_Deva"
    n_layers = len(pd.read_csv(folder / "metrics.csv").query("view == 'token'"))
    for view, n in (("token", 300), ("sentence-mean", 60), ("sentence-last", 60)):
        with np.load(folder / view / "pca3d.npz") as proj:
            assert proj["coords"].shape == (n_layers, n, 3)
            assert proj["var_ratio"].shape == (n_layers, 3) and proj["spread"].shape == (n_layers,)
            assert np.isfinite(proj["coords"]).all() and (proj["var_ratio"].sum(1) <= 1 + 1e-6).all()
            assert proj["layer_names"][-1].endswith("(after final norm)")     # tiny Llama has a final norm
        with safe_open(folder / view / "points.safetensors", "pt") as f:
            assert f.get_slice("points")[n_layers - 1].shape[0] == n      # one layer loads on its own
        assert (folder / view / "metrics.png").exists() == plot
        assert (folder / view / "pca3d.png").exists() == plot


def test_final_norm_split():
    # Llama has a norm after its last layer: L+2 hidden states, the last = norm(the one before it)
    llama = load_model(TINY["tiny-llama"], "cpu")
    assert llama.final_norm == "norm"
    ids = llama.tokenizer("the cat sleeps under the tree", return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        hs = llama.hidden_states(ids, torch.ones_like(ids))
        torch.testing.assert_close(llama.model.norm(hs[-2]), hs[-1])
    n_layers = llama.model.config.num_hidden_layers
    assert len(hs) == n_layers + 2 and not torch.equal(hs[-2], hs[-1])
    names = llama.layer_names(len(hs))
    assert names[0] == "embeddings" and names[-2:] == [f"layer {n_layers} (before final norm)",
                                                         f"layer {n_layers} (after final norm)"]
    # BERT (post-LN inside each layer) has no final norm: L+1 hidden states
    bert = load_model(TINY["tiny-st"], "cpu")
    with torch.inference_mode():
        hs = bert.hidden_states(ids.clamp(max=bert.tokenizer.vocab_size - 1), torch.ones_like(ids))
    assert bert.final_norm is None and len(hs) == bert.model.config.num_hidden_layers + 1
    assert bert.layer_names(len(hs))[-1] == f"layer {len(hs) - 1}"
