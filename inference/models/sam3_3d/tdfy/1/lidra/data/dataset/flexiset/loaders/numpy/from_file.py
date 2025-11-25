import os
import numpy as np

from lidra.data.dataset.flexiset.loaders.base import Base


class FromFile(Base):
    def _load(self, path):
        return np.load(path)
