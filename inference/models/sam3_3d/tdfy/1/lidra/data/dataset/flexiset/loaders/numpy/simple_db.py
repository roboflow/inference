import os
import numpy as np

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.numpy.from_file import FromFile


class SimpleDB(Base):
    data_loader = FromFile()

    def __init__(self, extension: str = ".npz"):
        super().__init__()
        self.extension = extension

    def _load(self, path, uid):
        filepath = os.path.join(path, f"{uid}{self.extension}")
        return self.data_loader._load(filepath)
