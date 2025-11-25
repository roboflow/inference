from functools import wraps

from lidra.data.dataset.flexiset.loaders.base import Base


class Lambda(Base):
    def __init__(self, function):
        self._load = function
        super().__init__()
