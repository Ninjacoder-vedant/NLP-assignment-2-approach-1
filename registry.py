class Registry:
    """Name -> class mapping, filled with the `@REGISTRY.register("name")` decorator."""

    def __init__(self, kind: str):
        self.kind = kind
        # name -> registered class
        self._items: dict[str, type] = {}

    # Used as @REGISTRY.register("name") above a class: returns a decorator that records the class
    def register(self, name: str):
        def deco(cls):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            # The class also learns its own name, e.g. IN22Gen.name == "in22-gen"
            cls.name = name
            # Return the class unchanged so it can still be used normally
            return cls
        return deco

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError:
            # Clearer error that lists the valid names; `from None` hides the original KeyError
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}") from None

    def names(self) -> list[str]:
        return list(self._items)


# One registry per kind of pluggable component
DATASETS = Registry("dataset")
MODEL_FAMILIES = Registry("model family")
METRICS = Registry("metric")
