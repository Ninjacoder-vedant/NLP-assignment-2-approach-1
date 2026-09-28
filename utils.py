import gc
import os

import torch


def setup_hf_token() -> str | None:
    """Put HF_TOKEN in the environment: Kaggle secret if available, else whatever is already set."""
    if not os.environ.get("HF_TOKEN"):
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
