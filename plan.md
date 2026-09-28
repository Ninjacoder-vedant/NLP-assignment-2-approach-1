# Code plan

## Datasets

First load the dataset, then load the model, run on this dataset, calculate Isotropy metrics, save the results

1. IN22-Conv (N-way parallel)
2. IN22-Gen (N-way parallel)
3. Flores+ (N-way parallel)

## Native corpus

1. Sangraha Verified (Sentence-wise)
2. IndicCorp v2 (Sentence-wise)
3. Wikipedia (Article-wise)
4. IITB IndicMonoDoc (Doc-wise)

## Models
Models that I want to load:

1. google/embeddinggemma-300m (hf token will be required)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
```

2. Qwen/Qwen3-Embedding-0.6B

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
```

3. Qwen/Qwen3.5-0.8B-Base

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
model = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3.5-0.8B-Base", device_map="auto")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

4. google/gemma-3-1b-pt

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-pt", device_map="auto")
```

5. microsoft/harrier-oss-v1-0.6b

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("microsoft/harrier-oss-v1-0.6b")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
```

6. meta-llama/Llama-3.2-1B

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", device_map="auto")
```

## Isotropy metrics

1. IsoScore
2. Average Random Cosine Similarity
3. Intrinsic Dimensionality (ID) Score
4. Maximum Explainable Variance (MEV)
