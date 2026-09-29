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


# Description of one model; frozen=True makes it read-only
@dataclass(frozen=True)
class ModelSpec:
    key: str                        # short name, used as the results folder
    hf_id: str
    family: str                     # a MODEL_FAMILIES name
    auto_class: str = "AutoModel"   # transformers class for the "hf" family


# All models, keyed by short name (dict comprehension: {spec.key: spec})
MODEL_SPECS: dict[str, ModelSpec] = {spec.key: spec for spec in [
    ModelSpec("embeddinggemma-300m", "google/embeddinggemma-300m", "sentence_transformer"),
    ModelSpec("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", "sentence_transformer"),
    ModelSpec("harrier-oss-v1-0.6b", "microsoft/harrier-oss-v1-0.6b", "sentence_transformer"),
    # Decoder LMs are loaded as the bare backbone (AutoModel): same hidden states, no LM head in memory.
    ModelSpec("gemma-3-1b-pt", "google/gemma-3-1b-pt", "hf"),
    ModelSpec("llama-3.2-1b", "meta-llama/Llama-3.2-1B", "hf"),
    ModelSpec("qwen3.5-0.8b-base", "Qwen/Qwen3.5-0.8B-Base", "hf"),
]}


def resolve_dtype(name: str, device: str) -> torch.dtype:
    """Pick the torch dtype to load a model in.

    'auto' gives bfloat16 on GPUs with native bf16 (Ampere+), else float32. float16 is never chosen
    automatically because Gemma-family activations overflow in it.

    Args:
        name: "auto" or a torch dtype name like "float32".
        device: e.g. "cuda:0" or "cpu".
    Returns:
        The torch dtype.
    """
    # Explicit choice, e.g. "float32" -> torch.float32
    if name != "auto":
        return getattr(torch, name)
    # Compute capability 8.x+ = Ampere or newer (A100, L4, ...); a T4 is 7.5
    if device.startswith("cuda") and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


# Common interface for every model family; subclasses only implement _load()
class ModelWrapper(ABC):
    def __init__(self, spec: ModelSpec, device: str, dtype: torch.dtype):
        """Load the model and tokenizer (via _load) and record special and padding token ids.

        Args:
            spec: which model to load.
            device: e.g. "cuda:0".
            dtype: precision to load the weights in.
        """
        self.spec, self.device, self.dtype = spec, device, dtype
        self.model, self.tokenizer, self.max_length = self._load()
        
        # Inference mode: disables dropout
        self.model.eval()
        tok = self.tokenizer

        # Ids of <bos>, <eos>, <pad>, ...: these tokens are never sampled
        self.special_ids: set[int] = set(tok.all_special_ids)

        # Padding id: many models (e.g. GPT-2, Llama) have no pad token,
        # so fall back to the eos token, and to 0 if there is no eos either
        if tok.pad_token_id is not None:
            self.pad_id: int = tok.pad_token_id
        elif tok.eos_token_id is not None:
            self.pad_id = tok.eos_token_id
        else:
            self.pad_id = 0

    @abstractmethod
    def _load(self) -> tuple[torch.nn.Module, "transformers.PreTrainedTokenizerBase", int]:
        """Load the model for this family.

        Returns:
            (backbone module on self.device, tokenizer, max sequence length).
        """

    # Extra arguments passed on every forward call (subclasses may override)
    forward_kwargs: dict = {}

    def hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run one forward pass and return the hidden states of every layer.

        Args:
            input_ids: [B, T] token ids (right-padded).
            attention_mask: [B, T], 1 for real tokens, 0 for padding.
        Returns:
            (num_layers + 1) tensors of shape [B, T, d]: embedding output, then after each layer.
        """
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, **self.forward_kwargs)
        return out.hidden_states

    def unload(self) -> None:
        """Delete the model and free its GPU memory."""
        # Drop the model reference and return its GPU memory
        del self.model
        free_cuda()


# Embedding models published for the sentence-transformers library
@MODEL_FAMILIES.register("sentence_transformer")
class SentenceTransformerWrapper(ModelWrapper):
    """Runs the transformer inside a SentenceTransformer directly (no pooling/dense/normalize head)."""

    def _load(self):
        """Load via sentence-transformers and return its inner HF transformer.

        Returns:
            (transformer module without pooling/dense/normalize head, tokenizer, max_seq_length).
        """
        # Imported here so the library is only needed when such a model is used
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(self.spec.hf_id, device=self.device, model_kwargs={"dtype": self.dtype})
        # st[0] is the Transformer module; .auto_model is the underlying HF model
        return st[0].auto_model, st.tokenizer, st.max_seq_length


# Plain Hugging Face transformers models (decoder LMs)
@MODEL_FAMILIES.register("hf")
class HFTransformerWrapper(ModelWrapper):
    # No KV cache: we only need one forward pass, not generation
    forward_kwargs = {"use_cache": False}

    def _load(self):
        """Load with the transformers class named in spec.auto_class (default AutoModel).

        Returns:
            (model on device, tokenizer, max sequence length from tokenizer/config).
        """
        # Class by name, e.g. "AutoModel" -> transformers.AutoModel
        cls = getattr(transformers, self.spec.auto_class)
        model = cls.from_pretrained(self.spec.hf_id, dtype=self.dtype).to(self.device)
        tok = AutoTokenizer.from_pretrained(self.spec.hf_id)
        # Multimodal models (Qwen3.5) keep text settings in config.text_config
        cfg = getattr(model.config, "text_config", model.config)
        # Longest sequence the model supports (tokenizer or position-embedding limit)
        max_len = min(tok.model_max_length, getattr(cfg, "max_position_embeddings", 10**9))
        return model, tok, int(max_len)


def load_model(spec: ModelSpec, device: str, dtype: str = "auto") -> ModelWrapper:
    """Load a model with the wrapper of its family.

    Args:
        spec: which model to load.
        device: e.g. "cuda:0".
        dtype: "auto" or a torch dtype name (see resolve_dtype).
    Returns:
        The loaded ModelWrapper.
    """
    # Pick the wrapper class of the spec's family and build it
    return MODEL_FAMILIES.get(spec.family)(spec, device, resolve_dtype(dtype, device))
