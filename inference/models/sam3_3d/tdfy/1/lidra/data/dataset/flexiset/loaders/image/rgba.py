import torch

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.image.from_file import FromFile


class RGBA(Base):
    def __init__(self):
        super().__init__()
        self.register_default_loader("image", FromFile())

    def _load(self, image):
        assert isinstance(image, torch.Tensor), "image must be a tensor"
        assert image.dim() == 3, "image must be 3D tensor"
        assert image.shape[0] == 4, "RGBA image expected to have 4 channels"
        return image
