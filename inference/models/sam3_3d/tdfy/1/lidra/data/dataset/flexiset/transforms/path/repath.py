import os

from lidra.data.dataset.flexiset.transforms.base import Base


class Repath(Base):
    def __init__(self, subpath):
        super().__init__()
        self._subpath = subpath

    def _transform(self, path):
        if self._subpath.startswith("/"):
            # overwrite if absolute
            return self._subpath
        # extend if relative
        return os.path.join(path, self._subpath)
