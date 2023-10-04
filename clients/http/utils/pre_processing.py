import numpy as np
from PIL import Image


def resize_opencv_image(
    image: np.ndarray,
    max_height: int,
    max_width: int,
) -> None:
    height, width = image.shape[:2]
    height_scaling_ratio = ...

def resize_pillow_image(
    image: Image.Image,
    max_height: int,
    max_width: int,
) -> None:
    pass


