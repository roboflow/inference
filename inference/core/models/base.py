import base64
import pickle
import re
from io import BytesIO
from time import perf_counter
from typing import Union

import requests
from PIL import Image

from inference.core.env import ALLOW_NUMPY_INPUT
from inference.core.exceptions import InputMethodNotAllowed, InvalidNumpyInput
from inference.core.models.mixins import InferenceMixin


class Model(InferenceMixin):
    """Base Inference Model (Inherits from InferenceMixin to define the needed methods)

    This class provides the foundational methods for inference and logging, and can be extended by specific models.

    Methods:
        log(m): Print the given message.
        clear_cache(): Clears any cache if necessary.
    """

    def log(self, m):
        """Prints the given message.

        Args:
            m (str): The message to print.
        """
        print(m)

    def clear_cache(self):
        """Clears any cache if necessary. This method should be implemented in derived classes as needed."""
        pass


class CvModel(Model):
    """Base CV Model (Defines a method for loading imagery)

    This class extends the base Model class and provides a method for loading images in various formats.

    Methods:
        load_image(type: str, value: Union[bytes, str]) -> Image.Image: Loads an image based on the specified type and value.
    """

    @staticmethod
    def load_image(type: str, value: Union[bytes, str]) -> Image.Image:
        """Loads an image based on the specified type and value.

        Args:
            type (str): The type of image to load. Supported types are 'base64', 'url', 'multipart', 'pil', and 'numpy'.
            value (Union[bytes, str]): The value containing the image data, depending on the type.

        Returns:
            Image.Image: The loaded PIL image, converted to RGB.

        Raises:
            NotImplementedError: If the specified image type is not supported.
            InputMethodNotAllowed: If the numpy input method is used but not allowed.
            InvalidNumpyInput: If the numpy input method is used and the input data is invalid.
        """
        if type == "base64":
            # New routes accept images via json body (str), legacy routes accept bytes which need to be decoded as strings
            if not isinstance(value, str):
                value = value.decode("utf-8")
            # Sometimes base64 strings that were encoded by a browser are padded with extra characters, so we need to remove them
            value = re.sub(r"^data:image\/[a-z]+;base64,", "", value)
            value = base64.b64decode(value)
            pil_image = Image.open(BytesIO(value))
        elif type == "url":
            pil_image = Image.open(requests.get(value, stream=True).raw)
        elif type == "multipart":
            pil_image = Image.open(value)
        elif type == "pil":
            pil_image = value
        elif type == "numpy":
            if ALLOW_NUMPY_INPUT:
                data = pickle.loads(value)
                try:
                    pil_image = Image.fromarray(data)
                except Exception as e:
                    if len(data.shape) != 3:
                        raise InvalidNumpyInput(
                            f"Expected 3 dimensions, got {len(data.shape)} dimensions."
                        )
                    elif data.shape[-1] != 3:
                        raise InvalidNumpyInput(
                            f"Expected 3 channels, got {data.shape[-1]} channels."
                        )
                    elif max(data) > 255 or min(data) < 0:
                        raise InvalidNumpyInput(
                            f"Expected values between 0 and 255, got values between {min(data)} and {max(data)}."
                        )

            else:
                raise InputMethodNotAllowed(
                    "Numpy input is not allowed for this server."
                )
        else:
            raise NotImplementedError(f"Image type '{type}' is not supported.")

        return pil_image.convert("RGB")
