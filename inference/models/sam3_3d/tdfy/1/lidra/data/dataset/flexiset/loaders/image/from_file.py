import torch
import numpy as np
import matplotlib.pyplot as plt

from lidra.data.dataset.flexiset.loaders.base import Base


# TODO(Pierre) probably want to make the dtype, value range and channel order configurable
class FromFile(Base):
    def _load(self, path):
        image = plt.imread(path)  # why use matplotlib ?
        if image.dtype == "uint8":
            image = image / 255
            image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.contiguous()
        return image
