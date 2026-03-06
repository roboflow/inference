from copy import copy

from lidra.data.dataset.flexiset.flexi.transform import Transform as FlexiTransform
from lidra.data.dataset.flexiset.transforms.trellis.image_and_mask import (
    ImageAndMaskMess,
)


def all_transforms(preprocessor=None):
    transforms = []
    # image and mask transform
    if preprocessor is not None:
        preprocessor_0 = preprocessor
        preprocessor_1 = copy(preprocessor)
        preprocessor_1.img_mask_joint_transform = []  # disable joint transform
        transforms.append(
            FlexiTransform(
                ("transformed_image", "transformed_mask"),
                ImageAndMaskMess(preprocessor_0),
            ),
        )
        transforms.append(
            FlexiTransform(
                ("rgb_image", "rgb_image_mask"),
                ImageAndMaskMess(preprocessor_1),
            ),
        )
    return transforms
