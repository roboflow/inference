from copy import deepcopy

from lidra.data.dataset.flexiset.loaders.base import Base


class Copy(Base):
    def _load(self, data):
        return deepcopy(data)
