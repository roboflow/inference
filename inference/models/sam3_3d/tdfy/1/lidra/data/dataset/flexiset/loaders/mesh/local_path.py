import os

from lidra.data.dataset.flexiset.loaders.base import Base


# COMMENT(Pierre) : hopefully this will be fixed in the metadata at some point
def _try_fix_buggy_path(path):
    prefix = "/fsx-3dfy/gleize/large_experiments/3dfy/datasets/objaverse-xl/"
    if path.startswith(prefix):
        path = path.replace(prefix, "raw/")
    return path


class LocalPath(Base):
    def _load(self, path, metadata):
        local_path = metadata["local_path"]
        local_path = _try_fix_buggy_path(local_path)
        if local_path.startswith("/"):
            return local_path
        return os.path.join(path, local_path)
