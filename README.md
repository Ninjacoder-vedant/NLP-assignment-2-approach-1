# How Do Language Models Represent Indian Languages?

This project compares the internal representations of six language models across Indian-language text. It asks a simple question: as a model reads a sentence, are its internal representations spread across many directions, or concentrated in a few?

The project is an analysis tool, not a chatbot. It does not translate text or write answers. Instead, it records measurements about the numerical representations a model creates while reading text, layer by layer.

## Why study this?

Language models turn text into vectors—lists of numbers that capture information the model uses internally. If those vectors are spread fairly evenly across many directions, the representations are called *isotropic*. If many vectors point in similar directions, they are more concentrated, or *anisotropic*. Comparing this pattern across layers, languages, datasets, and models can help researchers understand how models organize language internally. These measurements describe representation geometry; by themselves, they do not show which model performs best at translation or another task.

## What the experiment does

The program reads text from a selected dataset, sends it through a selected model, and collects vectors for sampled tokens at each layer. It then calculates four measurements: IsoScore estimates how evenly variance is spread, average cosine similarity checks how similarly vectors from different texts point, maximum explainable variance reports how much variance lies along the strongest direction, and intrinsic dimension estimates how many dimensions are meaningfully used. Results are saved so an interrupted run can continue without repeating finished language-and-dataset combinations.

The project uses seven datasets: IN22-Gen, IN22-Conv, FLORES+, Sangraha Verified, IndicCorp v2, Wikipedia, and IITB IndicMonoDoc. Some align the same content across languages; others contain text in individual languages. The six configured models are EmbeddingGemma, Qwen3-Embedding, Harrier, Gemma 3, Llama 3.2, and Qwen3.5. Language coverage differs by dataset. Use `python run.py --list` to see the exact names accepted by the program.

## Get started

Use Python 3.12 or later. In PowerShell, create an environment and install the project packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start with one model, one dataset, and one language rather than the full experiment:

```powershell
python run.py --models embeddinggemma-300m --datasets in22-gen --langs hin_Deva --n-tokens 1000 --device cpu --dtype float32
```

The first run may download the dataset and model weights, which can take time and disk space. Model files and downloaded data stay on your computer and are excluded from Git. The command above is a small example for trying the pipeline; treat its measurements as exploratory. For comparisons, use the same sufficiently large token count for each language and model. Running `python run.py` without options selects the full set of configured models, datasets, and languages and can require substantial compute time and storage.

## Hugging Face access

Some models or datasets require Hugging Face account approval. Accept the model or dataset terms while signed in, then create a read token. If authentication is needed, copy `.env.example` to `.env` and replace its placeholder with your token. Keep `.env` private; it is excluded from Git. Public models do not need a token.

## Where results go

The experiment writes one CSV for each model, dataset, and language under `results/`. Each CSV contains one row per model layer and the measurements for that layer. Run settings and token counts are saved alongside the CSVs, together with a 3-D PCA projection of every layer's tokens (`pca3d/{lang}.npz`). Add `--plot` to also draw the metrics against depth (`plots/{lang}.png`) and one 3-D scatter per layer (`plots/pca3d/{lang}.png`); `python plotting.py` redraws them from saved results. Re-running the same command skips results already completed with matching settings.

To check that model loading and inference work without starting a dataset experiment, run:

```powershell
python scripts/check_models.py --models embeddinggemma-300m
```

The check sends two example sentences through the model and writes a log under `cache/model-checks/`. It displays the input sentence and a preview of the resulting vector; these numbers are model representations, not generated text.

## Tests

Run the project tests with `pytest -q`.
