"""Download configured models into models/ and log sample text-to-hidden-state checks."""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
LOG_DIR = ROOT / "cache" / "model-checks"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".hf-home"))

import torch
import numpy as np
from huggingface_hub import snapshot_download

from inference import HiddenStateExtractor
from model_loader import MODEL_SPECS, load_model
from utils import setup_hf_token

SAMPLE_TEXTS = [
    "The weather is lovely today.",
    "He drove to the stadium.",
]


def weights_ready(folder: Path) -> bool:
    """Return true only when all indexed shards, or a single weight file, are present."""
    for index in (folder / "model.safetensors.index.json", folder / "pytorch_model.bin.index.json"):
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            return bool(shards) and all((folder / shard).is_file() for shard in shards)
    return any(folder.glob("*.safetensors")) or any(folder.glob("pytorch_model*.bin"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_SPECS, default=list(MODEL_SPECS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--text", action="append", dest="texts", help="input sentence; repeat to add examples")
    args = parser.parse_args()
    texts = args.texts or SAMPLE_TEXTS
    setup_hf_token()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"model-check-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
    logger = logging.getLogger("model_check")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    failed = []
    with log_path.open("a", encoding="utf-8") as records:
        for key in args.models:
            original = MODEL_SPECS[key]
            local_dir = MODELS_DIR / key
            try:
                if not weights_ready(local_dir):
                    logger.info("%s downloading model files into %s", key, local_dir)
                    snapshot_download(original.hf_id, local_dir=str(local_dir))
                model = load_model(original, args.device, args.dtype)
                try:
                    encoded = model.tokenizer(
                        texts, return_tensors="pt", padding=True, truncation=True, max_length=32
                    )
                    encoded = {name: value.to(model.device) for name, value in encoded.items()}
                    with torch.inference_mode():
                        layers = model.hidden_states(encoded["input_ids"], encoded["attention_mask"])
                    finite = all(bool(torch.isfinite(layer).all()) for layer in layers)
                    if not finite:
                        raise FloatingPointError("hidden states contain NaN or infinity")
                    extractor = HiddenStateExtractor(model, batch_size=1)
                    collected = extractor.extract(texts)
                    extracted_shape = None
                    if len(collected[0]):
                        # sample() raises FloatingPointError on NaN or infinity
                        extracted = extractor.sample(collected, min(2, len(collected[0])), np.random.default_rng(0))
                        extracted_shape = list(extracted.shape)
                    for row, text in enumerate(texts):
                        length = int(encoded["attention_mask"][row].sum())
                        token_ids = encoded["input_ids"][row, :length].tolist()
                        decoded = model.tokenizer.decode(token_ids, skip_special_tokens=False)
                        preview = layers[-1][row, length - 1, :8].float().cpu().tolist()
                        result = {
                            "model": key,
                            "hf_id": original.hf_id,
                            "dtype": str(model.dtype),
                            "input_text": text,
                            "token_count": length,
                            "decoded_input": decoded,
                            "output_kind": "last-token hidden-state vector preview",
                            "all_layer_shapes": [list(layer.shape) for layer in layers],
                            "inference_extract_shape": extracted_shape,
                            "output_vector_first_8_values": preview,
                            "all_hidden_states_finite": finite,
                            "status": "pass",
                        }
                        records.write(json.dumps(result, ensure_ascii=False) + "\n")
                        records.flush()
                        logger.info(
                            "%s INPUT %r | decoded %r | OUTPUT final-layer vector %s first8=%s | sampled hidden-state batch=%s",
                            key, text, decoded, tuple(layers[-1][row, length - 1].shape), preview, extracted_shape,
                        )
                    logger.info("%s PASS layers=%d device=%s dtype=%s", key, len(layers), args.device, model.dtype)
                finally:
                    model.unload()
            except Exception as exc:
                failed.append(key)
                failure = {
                    "model": key,
                    "hf_id": original.hf_id,
                    "status": "fail",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1200],
                }
                records.write(json.dumps(failure, ensure_ascii=False) + "\n")
                records.flush()
                logger.error("%s FAIL (%s): %s", key, type(exc).__name__, exc)
    print(f"Model check log: {log_path}")
    if failed:
        raise SystemExit(f"Model checks failed: {', '.join(failed)}; inspect the log above.")


if __name__ == "__main__":
    main()
