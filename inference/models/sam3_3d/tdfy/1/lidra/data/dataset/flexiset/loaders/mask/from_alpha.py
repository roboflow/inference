from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.flexiset.loaders.image.rgba import RGBA


class FromAlpha(Base):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.register_default_loader("rgba", RGBA())
        self.threshold = threshold

    def _load(self, rgba):
        mask = rgba[3:4, :, :]

        # ensure the mask is binary
        mask = (mask > self.threshold).float()

        return mask
