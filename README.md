# NLP Assignment 2

This project studies how well different language models represent text in Indian languages. It loads text, converts it into tokens, collects the model's hidden states, calculates isotropy measurements, and saves the results.

## Setup

Open PowerShell in this folder and create the project environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

EmbeddingGemma is already copied into `models/embeddinggemma-300m` for local use. The model files are ignored by Git because they are large. The same model also exists in the local Hugging Face cache, so this checkout does not need to download it again.

Other models may be gated by Hugging Face. If one asks for authentication, copy `.env.example` to `.env`, replace the placeholder with your own token, and keep `.env` private. The runner reads `HF_TOKEN` from that file; do not put a real token in source files or commits.

## Check model loading

The model wrapper is in `model_loader.py`. To test all configured models, run `python scripts/check_models.py`. It downloads weights into `models/<model-name>/`, sends two sample sentences through each model, and logs the original text, tokenized text, output shape, and a short preview of the final hidden-state vector. You can pass your own examples with repeated `--text` arguments, such as `python scripts/check_models.py --models embeddinggemma-300m --text "A sentence to test." --text "A second sentence."`. These models produce vectors for this experiment, rather than answer sentences. The logs go under `cache/model-checks/`; models that need account approval or a token are recorded there with the reason they could not load.

```powershell
$env:HF_HUB_OFFLINE = "1"
python scripts/check_models.py --models embeddinggemma-300m
```

## Run the experiment

To see available models, datasets, and metrics, run `python run.py --list`. A small run can then be started with a selected model, dataset, language, and token budget, for example `python run.py --models embeddinggemma-300m --datasets in22-gen --langs hin_Deva --n-tokens 1000`. Results are written under `cache/` and can be resumed.

## Main files

`model_loader.py` loads models through one common interface. `dataset_loader.py` downloads and prepares datasets. `inference.py` tokenizes text and extracts hidden states. `isotropy_metrics.py` calculates measurements. `results.py` stores resumable output. `run.py` connects these pieces.

## Tests

Run the lightweight tests with `pytest -q`. The focused model check above is separate because real model weights are large and model access can require a Hugging Face token.

