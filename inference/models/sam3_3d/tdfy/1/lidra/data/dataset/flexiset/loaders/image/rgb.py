import torch

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.image.from_file import FromFile


class RGB(Base):
    def __init__(self):
        super().__init__()
        self.register_default_loader("image", FromFile())

    def _load(self, image):
        assert isinstance(image, torch.Tensor), "image must be a tensor"
        assert image.dim() == 3, "image must be 3D tensor"
        assert image.shape[0] in {
            3,
            4,
        }, "RGB or RGBA image expected, should have 3 or 4 channels"
        image = image[:3]
        return image
