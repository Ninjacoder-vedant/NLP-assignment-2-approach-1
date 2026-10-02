# Code understanding plan
Read the files in the order the data moves through them: text → tokens → hidden states → numbers → CSV. The top file, [run.py](run.py), shows the overall flow, and each later file fills in one step.

## Reading order

* [ ] run.py
* [x] registry.py
* [x] languages.py
* [ ] dataset_loader.py
* [ ] model_loader.py
* [ ] inference.py
* [ ] isotropy_metrics.py
* [ ] results.py

**1. [run.py](run.py): the big picture (start here)**
- Read `Experiment.run` and `Experiment.run_dataset` first. These two methods are the whole pipeline in about 50 lines.
- Skip `parse_args` for now; it only turns command-line flags into a `RunConfig`.

**2. [registry.py](registry.py): about 30 lines**
- This is how `"in22-gen"` or `"isoscore"` gets looked up as a class. Read it before the next files, because they all use `@DATASETS.register(...)`.

**3. [languages.py](languages.py): just a table of the 22 languages**

**4. [dataset_loader.py](dataset_loader.py): text in**
- Read `BaseDataset` fully. It sets the rules every dataset follows.
- Then read `IN22Gen`, the simplest dataset: 2 lines on top of `WideParallelDataset`.
- Skip `HubTextDataset` (the random-offset sampling) on a first pass; it's the most complex part.

**5. [model_loader.py](model_loader.py): models in**
- `ModelSpec` is the list of models. Both wrappers just return `(model, tokenizer, max_length)`.

**6. [inference.py](inference.py): the core logic**
- `extract` runs the model over the texts in batches of `batch_size` rows and collects the hidden states of every non-special token; `sample` then picks the same N of them at every layer.

**7. [isotropy_metrics.py](isotropy_metrics.py): the maths**
- Each metric is one small class. Start with `MaxExplainableVariance`, which is 2 lines.

**8. [results.py](results.py): saving CSVs and resuming**

**9. [tests/](tests/)**
- The tests show how each piece is meant to be used on small, understandable inputs.

## Python features you'll meet

| Feature | Where | One-line idea |
|---|---|---|
| `@dataclass` | `RunConfig`, `ModelSpec` | Auto-writes `__init__` for a class that just holds fields |
| `ABC`, `@abstractmethod` | `BaseDataset`, `ModelWrapper` | A "template" class; subclasses *must* fill in these methods |
| Decorators (`@something`) | `@DATASETS.register(...)` | A function that wraps or records a class or function |
| `@cached_property` | `_columns`, `eigvals` | Computed the first time it's used, then remembered |
| `ClassVar` | `hf_id`, `granularity` | A setting shared by the whole class, not per object |
| `yield` | `_iter_texts` | Produces items one at a time (a generator) instead of building a whole list |
| Type hints (`-> list[str]`) | everywhere | Documentation only; Python doesn't enforce them |

## A hands-on way in

Reading goes faster if you poke at real objects. In a notebook cell in `/root/Code`:

```python
from utils import setup_hf_token; setup_hf_token()
from dataset_loader import IN22Conv
ds = IN22Conv()
print(ds.languages())
texts = ds.load("hin_Deva")
print(len(texts), texts[:3])
```

Then load a model and call `HiddenStateExtractor(model, batch_size=2).extract(texts[:5])` with `--debug` logging to see which tokens are kept per batch.

Ask whenever you get stuck, on a Python feature or on why something was done a certain way.

# Registery.py

The registry is a **dictionary from names to classes** (`"in22-conv"` → `IN22Conv`), and the decorator is just a neat way to add each class to that dictionary. Every output below comes from the demo I just ran.

## 1. Why we need it

On the command line you type `--datasets in22-conv`, and the program receives the **string** `"in22-conv"`. It has to turn that string into the **class** `IN22Conv`. Without a registry you'd write something like this:

```python
def make_dataset(name):
    if name == "in22-gen":
        return IN22Gen()
    elif name == "in22-conv":
        return IN22Conv()
    elif name == "flores-plus":
        return FloresPlus()
    # ... and edit this every time you add a dataset
```

With the registry, the name sits next to the class, and nothing else needs editing when you add one:

```python
@DATASETS.register("in22-conv")
class IN22Conv(WideParallelDataset):
    hf_id = "ai4bharat/IN22-Conv"
```

## 2. What a decorator is

A decorator is **a function that receives a class (or function) right after it is defined**. The `@` line is shorthand:

```python
@shout
class Cat:
    pass
```

does exactly the same as

```python
class Cat:
    pass
Cat = shout(Cat)      # pass the class to shout(), keep whatever it returns
```

Demo:

```python
def shout(cls):
    print(f"shout() received {cls.__name__}")
    return cls        # hand the class back unchanged

@shout
class Cat:
    pass
```
Output:
```
shout() received Cat
Cat is still: <class '__main__.Cat'>
```

So a decorator can **record** the class somewhere and return it unchanged, and the class still works normally. That is all the registry does.

## 3. A decorator with an argument: `@DATASETS.register("in22-conv")`

Here there are two calls in a row:

```python
@DATASETS.register("in22-conv")
class IN22Conv: ...
```

is the same as

```python
deco = DATASETS.register("in22-conv")   # step 1: returns the inner function `deco`,
                                        #         which remembers name = "in22-conv"
IN22Conv = deco(IN22Conv)               # step 2: deco stores the class and returns it
```

Now the code in [registry.py](registry.py) should read clearly:

```python
class Registry:
    def __init__(self, kind):
        self.kind = kind
        self._items = {}                  # the dictionary: name -> class

    def register(self, name):             # step 1: called with the name
        def deco(cls):                    # step 2: called with the class
            self._items[name] = cls       #   store it: {"in22-conv": IN22Conv}
            cls.name = name               #   the class also learns its name
            return cls                    #   give the class back unchanged
        return deco

    def get(self, name):                  # lookup: "in22-conv" -> IN22Conv
        return self._items[name]          # (the real code also gives a clearer error)

    def names(self):                      # every registered name
        return list(self._items)
```

## 4. Small example you can run

```python
from registry import Registry

ANIMALS = Registry("animal")

@ANIMALS.register("dog")
class Dog:
    def speak(self):
        return "woof"

print(ANIMALS.names())                  # ['dog']
print(ANIMALS.get("dog"))               # <class '__main__.Dog'>
print(Dog.name)                         # dog
print(ANIMALS.get("dog")().speak())     # woof   <- get the class, () creates an object
ANIMALS.get("cow")                      # KeyError: "Unknown animal 'cow'. Available: ['dog']"
```

## 5. Where the project uses it

There are three registries, created at the bottom of `registry.py`:

| Registry | Filled in (the `@...register` lines) | Looked up in |
|---|---|---|
| `DATASETS` | [dataset_loader.py](dataset_loader.py), e.g. `@DATASETS.register("in22-conv")` | `make_dataset` in [run.py](run.py): `cls = DATASETS.get(name)` |
| `METRICS` | [isotropy_metrics.py](isotropy_metrics.py), e.g. `@METRICS.register("mev")` | `build_metrics`: `METRICS.get(n)(**params)` |
| `MODEL_FAMILIES` | [model_loader.py](model_loader.py): `"sentence_transformer"`, `"hf"` | `load_model`: `MODEL_FAMILIES.get(spec.family)(...)` |

`run.py` also uses `DATASETS.names()` to set the defaults and to reject misspelled names in `--datasets`.

What the registries hold once those modules are imported:
```
DATASETS: ['in22-gen', 'in22-conv', 'flores-plus', 'sangraha-verified', 'wikipedia', 'indiccorp-v2', 'iitb-indicmonodoc']
METRICS: ['isoscore', 'mev', 'avgcos', 'id']
MODEL_FAMILIES: ['sentence_transformer', 'hf']
```

## 6. One subtle point

The `@...register` lines run **when the file is imported**, not when you call something. That is why [run.py](run.py) has this line:

```python
import dataset_loader  # noqa: F401  (registers datasets)
```

It uses nothing from `dataset_loader` directly. Importing the module is what fills `DATASETS`. Remove that line and `DATASETS.names()` would come back empty. (`# noqa: F401` tells code checkers the "unused import" is intentional.)