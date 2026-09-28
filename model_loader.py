"""Models: every model is wrapped behind `ModelWrapper`, the only interface inference.py uses.

To add a model of an existing family, add one `ModelSpec` to MODEL_SPECS. To add a new family
(e.g. an encoder-decoder), subclass `ModelWrapper`, implement `_load`, and register it with
`@MODEL_FAMILIES.register("family-name")`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import transformers
from transformers import AutoTokenizer

from registry import MODEL_FAMILIES
from utils import free_cuda


@dataclass(frozen=True)
class ModelSpec:
    key: str                        # short name, used as the results folder
    hf_id: str
    family: str                     # a MODEL_FAMILIES name
    auto_class: str = "AutoModel"   # transformers class for the "hf" family


MODEL_SPECS: dict[str, ModelSpec] = {s.key: s for s in [
    ModelSpec("embeddinggemma-300m", "google/embeddinggemma-300m", "sentence_transformer"),
    ModelSpec("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", "sentence_transformer"),
    ModelSpec("harrier-oss-v1-0.6b", "microsoft/harrier-oss-v1-0.6b", "sentence_transformer"),
    # Decoder LMs are loaded as the bare backbone (AutoModel): same hidden states, no LM head in memory.
    ModelSpec("gemma-3-1b-pt", "google/gemma-3-1b-pt", "hf"),
    ModelSpec("llama-3.2-1b", "meta-llama/Llama-3.2-1B", "hf"),
    ModelSpec("qwen3.5-0.8b-base", "Qwen/Qwen3.5-0.8B-Base", "hf"),
]}


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """'auto': bfloat16 on GPUs with native bf16 (Ampere+), else float32. float16 is never chosen
    automatically because Gemma-family activations overflow in it."""
    if name != "auto":
        return getattr(torch, name)
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


class ModelWrapper(ABC):
    def __init__(self, spec: ModelSpec, device: str, dtype: torch.dtype):
        self.spec, self.device, self.dtype = spec, device, dtype
        self.model, self.tokenizer, self.max_length = self._load()
        self.model.eval()
        tok = self.tokenizer
        self.special_ids: set[int] = set(tok.all_special_ids)
        self.pad_id: int = next(i for i in (tok.pad_token_id, tok.eos_token_id, 0) if i is not None)

    @abstractmethod
    def _load(self) -> tuple[torch.nn.Module, "transformers.PreTrainedTokenizerBase", int]:
        """Return (backbone module, tokenizer, max sequence length)."""

    forward_kwargs: dict = {}

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(num_layers + 1) tensors of shape [B, T, d]: embedding output, then after each layer."""
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, **self.forward_kwargs)
        return out.hidden_states

    def unload(self) -> None:
        del self.model
        free_cuda()


@MODEL_FAMILIES.register("sentence_transformer")
class SentenceTransformerWrapper(ModelWrapper):
    """Runs the transformer inside a SentenceTransformer directly (no pooling/dense/normalize head)."""

    def _load(self):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(self.spec.hf_id, device=self.device, model_kwargs={"dtype": self.dtype})
        return st[0].auto_model, st.tokenizer, st.max_seq_length


@MODEL_FAMILIES.register("hf")
class HFTransformerWrapper(ModelWrapper):
    forward_kwargs = {"use_cache": False}

    def _load(self):
        cls = getattr(transformers, self.spec.auto_class)
        model = cls.from_pretrained(self.spec.hf_id, dtype=self.dtype).to(self.device)
        tok = AutoTokenizer.from_pretrained(self.spec.hf_id)
        cfg = getattr(model.config, "text_config", model.config)
        max_len = min(tok.model_max_length, getattr(cfg, "max_position_embeddings", 10**9))
        return model, tok, int(max_len)


def load_model(spec: ModelSpec, device: str, dtype: str = "auto") -> ModelWrapper:
    return MODEL_FAMILIES.get(spec.family)(spec, device, resolve_dtype(dtype, device))
