from lidra.data.dataset.flexiset.transforms.base import Base
from lidra.data.dataset.tdfy.trellis.dataset import PreProcessor
from typing import Optional, Tuple, Union

# TODO(Pierre) clean, design a bit better ?


class ImageAndMaskMess(Base):
    def __init__(self, preprocessor: PreProcessor):
        super().__init__()
        self._preprocessor = preprocessor

    def _transform(self, image, mask, pointmap=None):
        """
        Apply transforms to image, mask, and optionally pointmap.

        Args:
            image: Image tensor
            mask: Mask tensor
            pointmap: Optional pointmap tensor

        Returns:
            Tuple of (image, mask) or (image, mask, pointmap) depending on input
        """
        # Check if we have pointmap and appropriate joint transforms
        if (
            pointmap is not None
            and hasattr(self._preprocessor, "img_mask_pointmap_joint_transform")
            and (
                self._preprocessor.img_mask_pointmap_joint_transform is not None
                and self._preprocessor.img_mask_pointmap_joint_transform != (None,)
            )
        ):
            # Use triple transforms if available
            for trans in self._preprocessor.img_mask_pointmap_joint_transform:
                image, mask, pointmap = trans(image, mask, pointmap)
        else:
            # Use dual transforms if pointmap is not available
            # Original behavior for image and mask only
            for trans in self._preprocessor.img_mask_joint_transform:
                image, mask = trans(image, mask)

        # Apply individual transforms
        if self._preprocessor.mask_transform is not None:
            mask = self._preprocessor.mask_transform(mask)

        if self._preprocessor.img_transform is not None:
            image = self._preprocessor.img_transform(image)

        if pointmap is not None and hasattr(self._preprocessor, "pointmap_transform"):
            if self._preprocessor.pointmap_transform is not None:
                pointmap = self._preprocessor.pointmap_transform(pointmap)

        # Return based on what was provided
        if pointmap is not None:
            return image, mask, pointmap
        return image, mask  # returned in same order as input
