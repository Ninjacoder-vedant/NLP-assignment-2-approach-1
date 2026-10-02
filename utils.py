import gc
import os
from pathlib import Path

import torch


def is_kaggle() -> bool:
    """True inside a Kaggle notebook (Kaggle sets KAGGLE_KERNEL_RUN_TYPE in every kernel)."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def setup_hf_token() -> str:
    """Put HF_TOKEN in the environment: Kaggle secret on Kaggle, else the .env file.

    A token that is already set in the environment is kept as is.

    Returns:
        The token.
    Raises:
        RuntimeError: if no token was found.
    """
    # Only look up a token if it is not already set
    if not os.environ.get("HF_TOKEN"):
        if is_kaggle():
            # The secret may not be attached to this notebook: fall through to the error below
            try:
                from kaggle_secrets import UserSecretsClient
                os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass
        else:
            # Imported here so Kaggle does not need python-dotenv installed
            from dotenv import load_dotenv
            # Finds the nearest .env walking up from this file
            load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        where = "add HF_TOKEN under Add-ons > Secrets" if is_kaggle() else "add HF_TOKEN=... to .env file"
        raise RuntimeError(f"HF_TOKEN not found: {where}")
    return token


def free_cuda() -> None:
    """Run garbage collection and release cached GPU memory."""
    # Delete unreferenced Python objects first, then return cached GPU memory to the driver
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
