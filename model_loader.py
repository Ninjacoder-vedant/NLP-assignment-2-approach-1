"""Models: every model is wrapped behind `ModelWrapper`, the only interface inference.py uses.

To add a model of an existing family, add one `ModelSpec` to MODEL_SPECS. To add a new family
(e.g. an encoder-decoder), subclass `ModelWrapper`, implement `_load`, and register it with
`@MODEL_FAMILIES.register("family-name")`.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import getsource
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from registry import MODEL_FAMILIES
from utils import free_cuda

log = logging.getLogger("isotropy.model_loader")


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
    if device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float32


def find_final_norm(model: torch.nn.Module) -> tuple[str | None, torch.nn.Module | None, torch.nn.Module | None]:
    """Find the decoder's final norm: a `norm` child that sits next to a `layers` ModuleList.

    Norms inside a layer or in a vision tower have no `layers` sibling, so they are skipped.

    Args:
        model: the backbone module.
    Returns:
        (dotted path of the norm, the norm module, the decoder module owning it), or (None, None, None).
    """
    for path, module in model.named_modules():
        layers = getattr(module, "layers", None)
        norm = getattr(module, "norm", None)
        if isinstance(layers, torch.nn.ModuleList) and isinstance(norm, torch.nn.Module):
            return (f"{path}.norm" if path else "norm"), norm, module
    return None, None, None


def set_tie_last_hidden_states(model: torch.nn.Module, tie: bool) -> None:
    """Set config.tie_last_hidden_states on every (sub)config, so nested text models (Qwen3.5) follow it.

    True (transformers' default): hidden_states[-1] is replaced by last_hidden_state (after the final norm).
    False: hidden_states[-1] stays the raw output of the last layer (before the final norm).
    """
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is None:
            continue
        cfg.tie_last_hidden_states = tie
        for sub in ("text_config", "vision_config"):
            if getattr(cfg, sub, None) is not None:
                getattr(cfg, sub).tie_last_hidden_states = tie


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

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

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

        # Path of the final norm (e.g. "norm") when the last layer is followed by one, else None
        self.final_norm: str | None = self._split_final_norm()

    @torch.inference_mode()
    def _split_final_norm(self) -> str | None:
        """Detect and capture the input to a final norm, if the model has one.

        A pre-forward hook records the exact tensor passed into the norm. This avoids recomputing
        the norm separately, which can differ numerically from the model's own forward result.

        Returns:
            Dotted path of the final norm module, or None if the model has none.
        """
        path, norm, decoder = find_final_norm(self.model)
        # A norm module that forward never calls does not count
        if norm is None or "self.norm(" not in getsource(type(decoder).forward):
            log.info("[%s] no final norm after the last layer: L+1 hidden states", self.spec.key)
            return None
        set_tie_last_hidden_states(self.model, False)
        self._final_norm_input: torch.Tensor | None = None
        self._final_norm_hook = norm.register_forward_pre_hook(self._capture_final_norm_input)
        log.info("[%s] final norm %s after the last layer: L+2 hidden states (before and after it)",
                 self.spec.key, path)
        return path

    def _capture_final_norm_input(self, module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        """Keep the exact pre-norm activation from the current model forward pass."""
        self._final_norm_input = inputs[0]

    def layer_names(self, n: int) -> list[str]:
        """Names of the n entries hidden_states() returns, e.g. for plots and metrics.csv.

        Args:
            n: number of hidden states (L+1, or L+2 with a final norm).
        Returns:
            ["embeddings", "layer 1", ..., "layer L"], and with a final norm the last two are
            "layer L (before final norm)" and "layer L (after final norm)".
        """
        names = ["embeddings"] + [f"layer {i}" for i in range(1, n)]
        if self.final_norm:
            last = n - 2
            names[-2:] = [f"layer {last} (before final norm)", f"layer {last} (after final norm)"]
        return names

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
            Tensors of shape [B, T, d]: embedding output, then after each layer (L+1). With a final norm,
            the last layer is its raw output and last_hidden_state (after the norm) is appended (L+2).
        """
        if self.final_norm:
            self._final_norm_input = None
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, **self.forward_kwargs)
        if self.final_norm:
            pre_norm = self._final_norm_input
            self._final_norm_input = None
            if pre_norm is None:
                raise RuntimeError(f"[{self.spec.key}] final norm hook did not run")
            return (*out.hidden_states[:-1], pre_norm, out.last_hidden_state)
        return out.hidden_states

    def layer_stages(self, n_states: int) -> list[str]:
        """Name embedding output and transformer outputs without assuming model internals."""
        stages = ["embedding"] + [f"block_{i}" for i in range(1, n_states)]
        if self.final_norm:
            last = n_states - 2
            stages[-2:] = [f"block_{last}_before_final_norm", "final_norm"]
        return stages

    def unload(self) -> None:
        """Delete the model and free its GPU memory."""
        # Drop the model reference and return its GPU memory
        if hasattr(self, "_final_norm_hook"):
            self._final_norm_hook.remove()
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
    # Prefer the repository-local copy when present; otherwise use the Hub id as-is.
    local = Path(spec.hf_id)
    if not local.is_dir():
        local = Path(__file__).resolve().parent / "models" / spec.key
    if (local / "config.json").is_file():
        spec = ModelSpec(spec.key, str(local), spec.family, spec.auto_class)
    # Pick the wrapper class of the spec's family and build it
    return MODEL_FAMILIES.get(spec.family)(spec, device, resolve_dtype(dtype, device))
