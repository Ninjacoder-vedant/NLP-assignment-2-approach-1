class Registry:
    """Name -> class mapping, filled with the `@REGISTRY.register("name")` decorator."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, type] = {}

    def register(self, name: str):
        def deco(cls):
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            cls.name = name
            return cls
        return deco

    def get(self, name: str) -> type:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}") from None

    def names(self) -> list[str]:
        return list(self._items)


DATASETS = Registry("dataset")
MODEL_FAMILIES = Registry("model family")
METRICS = Registry("metric")
