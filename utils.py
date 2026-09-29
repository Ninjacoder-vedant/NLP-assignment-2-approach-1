import gc
import os
from pathlib import Path

import torch


def setup_hf_token() -> str | None:
    """Load HF_TOKEN from the environment, repo .env, or Kaggle secret.

    Returns:
        The token, or None if none was found.
    """
    # Read only HF_TOKEN from this repo's ignored .env file when not already exported.
    if not os.environ.get("HF_TOKEN"):
        env_file = Path(__file__).resolve().parent / ".env"
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "HF_TOKEN":
                    token = value.strip().strip("\"'")
                    if token:
                        os.environ["HF_TOKEN"] = token
                        break
        except OSError:
            pass
    # Only look up the Kaggle secret if the token is still missing.
    if not os.environ.get("HF_TOKEN"):
        # Outside Kaggle the import fails: silently keep going without a token.
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


def free_cuda() -> None:
    """Run garbage collection and release cached GPU memory."""
    # Delete unreferenced Python objects first, then return cached GPU memory to the driver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
