"""Most common color of an image by k-means: ``roboflow_core/dominant_color@v1``.

Port of V1 ``roboflow_core/dominant_color@v1``. As V1::

    step   = max(1, min(width, height) // target_size)
    pixels = every step-th pixel of every step-th row
    k-means over pixel colors: k = color_clusters random pixels as centres;
        assign each pixel to its nearest centre, move each centre to the mean
        of its pixels (an empty cluster restarts at a random pixel); stop when
        the centres no longer move or after max_iterations
    color  = centre of the cluster with the most pixels, rounded

CPU boundary: none. Downsampling and k-means run on the image's device; each
iteration reads back two flags (an empty cluster, convergence), and the
result is three numbers.

Differences from V1:

* Deterministic: starting centres and restarts come from a fixed-seed
  generator, so the same image always gives the same color. V1 drew them from
  NumPy's unseeded global generator.
* An image with fewer sampled pixels than ``color_clusters`` uses one cluster
  per pixel; V1 failed the step.
* A one-channel image gives a gray ``(v, v, v)`` color.
* ``hex_color`` (``#RRGGBB``) is added so the color can feed the color
  parameters of drawing blocks directly.
"""

from typing import Any, Dict, Tuple
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND, Kind
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


SEED = 0

RGB = Tuple[int, int, int]

def _is_rgb_color(payload: Any) -> bool:
    return (
        isinstance(payload, tuple)
        and len(payload) == 3
        and all(type(value) is int and 0 <= value <= 255 for value in payload)
    )

RGB_COLOR_KIND = Kind(
    name="rgb_color",
    description="(red, green, blue) tuple of ints in 0..255.",
    validate=_is_rgb_color,
)

class DominantColorBlock(Block):
    """Find the most common color of an image.

    Example: the dominant color of each player crop separates the two teams'
    jerseys; ``hex_color`` can color that player's box in a drawing block.
    """

    type = "roboflow_core/dominant_color@v1"
    metadata = {
        "name": "Dominant Color",
        "port_of": "roboflow_core/dominant_color@v1",
        "section": "classical_cv",
    }
    outputs = {
        "rgb_color": Output(
            RGB_COLOR_KIND,
            source="image",
            description=(
                "Dominant color as an (r, g, b) tuple; use hex_color for "
                "color fields."
            ),
        ),
        "hex_color": Output(
            STRING_KIND,
            source="image",
            description="Dominant color as #RRGGBB.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to analyse.")
        color_clusters: StrictInt | Ref(INTEGER_KIND) = Field(
            default=4,
            gt=0,
            le=10,
            description="Number of k-means color clusters.",
            examples=[4, "$inputs.color_clusters"],
        )
        max_iterations: StrictInt | Ref(INTEGER_KIND) = Field(
            default=100,
            gt=0,
            le=500,
            description="Most k-means iterations.",
            examples=[100],
        )
        target_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=100,
            gt=0,
            le=250,
            description=(
                "Sampling keeps every step-th pixel, step = shorter image side "
                "// target_size (at least 1)."
            ),
            examples=[100],
        )

    def run(
        self,
        *,
        image: ImageData,
        color_clusters: int,
        max_iterations: int,
        target_size: int,
    ) -> Dict[str, Any]:
        """Cluster the sampled pixel colors and return the largest cluster's.

        Args:
            image: Image to analyse.
            color_clusters: Number of clusters, 1..10.
            max_iterations: Most iterations, 1..500.
            target_size: Sampling target for the shorter side, 1..250.

        Returns:
            ``rgb_color`` and ``hex_color`` of the largest cluster.
        """
        pixels = _sampled_colors(image.tensor_image, target_size=target_size)
        color = _dominant_color(
            pixels, clusters=color_clusters, max_iterations=max_iterations
        )
        result = {"rgb_color": color, "hex_color": "#{:02X}{:02X}{:02X}".format(*color)}

        return result

def _sampled_colors(pixels: torch.Tensor, *, target_size: int) -> torch.Tensor:
    # (P, 3) float32 colors of every step-th pixel, on the image's device.
    height, width = int(pixels.shape[1]), int(pixels.shape[2])
    step = max(1, min(width, height) // target_size)
    sampled = pixels[:, ::step, ::step].reshape(pixels.shape[0], -1).T

    return sampled.to(torch.float32).expand(-1, 3)

def _dominant_color(pixels: torch.Tensor, *, clusters: int, max_iterations: int) -> RGB:
    count = pixels.shape[0]
    clusters = min(clusters, count)
    generator = torch.Generator().manual_seed(SEED)

    def random_pixels(number: int) -> torch.Tensor:
        rows = torch.randint(count, (number,), generator=generator)
        return pixels[rows.to(pixels.device)]

    first = torch.randperm(count, generator=generator)[:clusters]
    centres = pixels[first.to(pixels.device)]
    for _ in range(max_iterations):
        labels = ((pixels[:, None] - centres[None]) ** 2).sum(dim=2).argmin(dim=1)
        sizes = torch.bincount(labels, minlength=clusters)
        sums = torch.zeros_like(centres).index_add_(0, labels, pixels)
        moved = sums / sizes.clamp(min=1).unsqueeze(1).to(sums.dtype)
        empty = sizes == 0
        if empty.any():
            moved[empty] = random_pixels(int(empty.sum()))
        if torch.allclose(centres, moved):
            break
        centres = moved

    largest = moved[torch.argmax(sizes)]
    color = tuple(int(value) for value in largest.round().clamp(0, 255).tolist())

    return color
