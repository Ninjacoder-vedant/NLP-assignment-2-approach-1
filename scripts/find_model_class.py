"""Print each model's classes and compare per-layer hidden states with tie_last_hidden_states=True vs False.

Models are loaded through model_loader.load_model, so the module inspected is exactly the backbone the
pipeline uses. Output for each model is printed and also saved to scripts/model/<key>.txt.

Usage:
    python scripts/find_model_class.py                                   # every model in MODEL_SPECS
    python scripts/find_model_class.py --model qwen3-embedding-0.6b llama-3.2-1b
"""
import argparse
import contextlib
from inspect import getsource
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "model_architecture"
sys.path.insert(0, str(ROOT))

from model_loader import MODEL_SPECS, find_final_norm, load_model, set_tie_last_hidden_states
from utils import setup_hf_token


class Tee:
    """Write to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def run(wrapper, inputs, tie: bool):
    """One forward pass; returns (hidden_states on CPU, last_hidden_state on CPU or None)."""
    set_tie_last_hidden_states(wrapper.model, tie)
    with torch.no_grad():
        out = wrapper.model(**inputs, output_hidden_states=True, return_dict=True, **wrapper.forward_kwargs)
    last = getattr(out, "last_hidden_state", None)
    return [h.detach().cpu() for h in out.hidden_states], (last.detach().cpu() if last is not None else None)


def inspect(key: str, device: str, dtype: str, text: str) -> None:
    spec = MODEL_SPECS[key]
    print(f"===== {key}  ({spec.hf_id}, family={spec.family}) =====\n")

    wrapper = load_model(spec, device, dtype)
    model = wrapper.model
    print(f"Backbone class: {type(model).__module__}.{type(model).__name__}")
    print(f"Config class:   {type(model.config).__name__}")
    print(f"Tokenizer:      {type(wrapper.tokenizer).__name__}")
    print(f"dtype: {wrapper.dtype}, device: {wrapper.device}\n")
    print(model)
    print("\nDistinct module classes:")
    for name in sorted({type(m).__name__ for m in model.modules()}):
        print(f"  {name}")

    inputs = wrapper.tokenizer(text, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    tied, tied_last = run(wrapper, inputs, True)
    untied, _ = run(wrapper, inputs, False)

    print(f"\nInput: {text!r}")
    print(f"hidden states: {len(tied)} (tied) / {len(untied)} (untied), shape per layer {tuple(tied[0].shape)}"
          "  [batch, seq_len, hidden]\n")
    print(f"{'layer':>5} | {'equal':>5} | {'max |diff|':>12} | {'mean |diff|':>12} | {'min cos':>10} | {'mean cos':>10}")
    print("-" * 72)
    for i, (a, b) in enumerate(zip(tied, untied)):
        a32, b32 = a.float(), b.float()
        diff = (a32 - b32).abs()
        cos = F.cosine_similarity(a32, b32, dim=-1)
        print(f"{i:>5} | {str(torch.equal(a, b)):>5} | {diff.max().item():>12.6f} | "
              f"{diff.mean().item():>12.6f} | {cos.min().item():>10.6f} | {cos.mean().item():>10.6f}")

    # last_hidden_state is always the final-norm output; with tying it should be the last hidden state
    if tied_last is not None:
        print(f"\ntied[-1] == last_hidden_state exactly?   {torch.equal(tied[-1], tied_last)}")
        print(f"untied[-1] == last_hidden_state exactly? {torch.equal(untied[-1], tied_last)}")

    # Is there a final norm after the last layer? Check structure, source, and numbers
    print("\n----- Final norm check -----")
    norm_path, norm, decoder = find_final_norm(model)
    if norm is None:
        print("No `norm` module next to `layers` found: no final norm detected structurally.")
    else:
        calls_norm = "self.norm(" in getsource(type(decoder).forward)
        print(f"Final norm module:   {norm_path} -> {norm}")
        print(f"{type(decoder).__name__}.forward calls self.norm(...)? {calls_norm}")
        if tied_last is not None:
            param = next(norm.parameters(), None)
            dev = param.device if param is not None else device
            with torch.no_grad():
                renormed = norm(untied[-1].to(dev)).cpu()
            diff = (renormed.float() - tied_last.float()).abs().max().item()
            print(f"norm(untied[-1]) == last_hidden_state exactly? {torch.equal(renormed, tied_last)}")
            print(f"max |norm(untied[-1]) - last_hidden_state|:    {diff:.6f}")

    wrapper.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS),
                        help="model keys from model_loader.MODEL_SPECS (default: all)")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--text", default="Hello, how are you?")
    args = parser.parse_args()
    setup_hf_token()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for key in args.model:
        out_path = OUT_DIR / f"{key}.txt"
        with out_path.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(Tee(sys.__stdout__, f)):
            try:
                inspect(key, args.device, args.dtype, args.text)
            except Exception:
                traceback.print_exc(file=sys.stdout)
                failed.append(key)
        print(f"-> saved {out_path.relative_to(ROOT)}\n")

    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
