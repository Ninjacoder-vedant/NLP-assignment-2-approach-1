class Registry:
    """Name -> class mapping, filled with the `@REGISTRY.register("name")` decorator."""

    def __init__(self, kind: str):
        """Create an empty registry.

        Args:
            kind: what is registered, e.g. "dataset" (used in error messages).
        """
        self.kind = kind
        # name -> registered class
        self._items: dict[str, type] = {}

    # Used as @REGISTRY.register("name") above a class: returns a decorator that records the class
    def register(self, name: str):
        """Decorator that registers a class under `name` and sets `cls.name = name`.

        Args:
            name: lookup name, e.g. "in22-gen".
        Returns:
            The decorator; it returns the class unchanged. Raises KeyError if `name` is taken.
        """
        def deco(cls):
            """Record `cls` under the name and return it unchanged."""
            if name in self._items:
                raise KeyError(f"{self.kind} '{name}' is already registered")
            self._items[name] = cls
            # The class also learns its own name, e.g. IN22Gen.name == "in22-gen"
            cls.name = name
            # Return the class unchanged so it can still be used normally
            return cls
        return deco

    def get(self, name: str) -> type:
        """Return the class registered as `name`; KeyError listing valid names otherwise."""
        try:
            return self._items[name]
        except KeyError:
            # Clearer error that lists the valid names; `from None` hides the original KeyError
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {self.names()}") from None

    def names(self) -> list[str]:
        """Return all registered names, in registration order."""
        return list(self._items)


# One registry per kind of pluggable component
DATASETS = Registry("dataset")
MODEL_FAMILIES = Registry("model family")
METRICS = Registry("metric")
