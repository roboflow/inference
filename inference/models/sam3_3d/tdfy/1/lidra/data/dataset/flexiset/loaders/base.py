from lidra.data.dataset.flexiset.args import Args


class Base:
    def __init__(self):
        self._default_loaders = {}
        self._args = Args(self._load)

    @property
    def args(self) -> Args:
        return self._args

    @property
    def default_loaders(self):
        return self._default_loaders

    def register_default_loader(self, name, loader):
        if not isinstance(loader, Base):
            raise ValueError(
                f"The provided loader must be an instance of 'Base', "
                f"but got '{type(loader).__name__}'."
            )
        if name in self._default_loaders:
            raise ValueError(
                f"A default loader with the name '{name}' is already registered."
            )
        self._default_loaders[name] = loader

    def _load(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement the _load method."
        )

    def load(self, **kwargs):
        for name, loader in self._default_loaders.items():
            if name not in kwargs:
                kwargs[name] = loader.load(**kwargs)

        args, kwargs = self._args.bind(kwargs)
        return self._load(*args, **kwargs)
