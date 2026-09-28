import gc
import os

import torch


def setup_hf_token() -> str | None:
    """Put HF_TOKEN in the environment: Kaggle secret if available, else whatever is already set."""
    # Only look up the Kaggle secret if the token is not already set
    if not os.environ.get("HF_TOKEN"):
        # Outside Kaggle the import fails: silently keep going without a token
        try:
            from kaggle_secrets import UserSecretsClient
            os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            pass
    return os.environ.get("HF_TOKEN")


def free_cuda() -> None:
    # Delete unreferenced Python objects first, then return cached GPU memory to the driver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
