"""Deterministic stand-in for an ``inference_models`` text-image embedding model.

It has the ``embed_images`` / ``embed_text`` API of CLIP and Perception
Encoder; nothing is downloaded. It is NOT model inference: embeddings are a
fixed function of the input, so tests can predict similarities exactly::

    image  -> [mean R, mean G, mean B] / 255, then 0          (4 values)
    "red" / "green" / "blue" -> the matching unit color axis, then 0
    "dark" -> [0, 0, 0, 1]
    other text -> 4 values seeded by the text (stable across runs)

So a pure red image scores 1.0 against "red" and 0.0 against "green" and
"blue"; a gray image scores the three colors equally.

Every fake records each call's inputs and the color format it was told.
"""

import hashlib
from typing import List, Optional, Sequence, Union

import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest

DIMENSIONS = 4
NAMED_TEXTS = {
    "red": (1.0, 0.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0, 0.0),
    "dark": (0.0, 0.0, 0.0, 1.0),
}


class FakeEmbeddingModel:
    """Color-mean image embeddings and named or seeded text embeddings.

    Args:
        request: The request it was loaded for, if any.
        device: Device of the returned embeddings.
        dtype: Floating dtype of the returned embeddings.
    """

    def __init__(
        self,
        request: Optional[ModelRequest] = None,
        *,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.request = request
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_calls: List[int] = []
        self.text_calls: List[List[str]] = []
        self.color_formats: List[Optional[str]] = []

    def embed_images(
        self,
        images: Sequence[torch.Tensor],
        input_color_format: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        self.image_calls.append(len(images))
        self.color_formats.append(input_color_format)
        rows = [
            torch.cat(
                [image.float().mean(dim=(1, 2)) / 255, image.new_zeros(1).float()]
            )
            for image in images
        ]
        embeddings = torch.stack(rows).to(device=self.device, dtype=self.dtype)

        return embeddings

    def embed_text(self, texts: Union[str, List[str]], **kwargs) -> torch.Tensor:
        texts = [texts] if isinstance(texts, str) else list(texts)
        self.text_calls.append(texts)
        rows = [text_embedding(text) for text in texts]
        embeddings = torch.stack(rows).to(device=self.device, dtype=self.dtype)

        return embeddings


def text_embedding(text: str) -> torch.Tensor:
    """The fake's embedding of one text (float32, CPU)."""
    if text in NAMED_TEXTS:
        return torch.tensor(NAMED_TEXTS[text])

    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")
    generator = torch.Generator().manual_seed(seed)
    embedding = torch.rand(DIMENSIONS, generator=generator) + 0.1

    return embedding


def color_image(red: int, green: int, blue: int, *, height: int = 8, width: int = 12):
    """A uniform ``(3, H, W)`` uint8 RGB tensor."""
    pixels = torch.tensor([red, green, blue], dtype=torch.uint8).view(3, 1, 1)
    image = pixels.expand(3, height, width).contiguous()

    return image
