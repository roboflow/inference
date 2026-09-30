"""Optional PNG gallery of delivered images, with their engine-resolved PTS.

Pixel export is a host boundary: it copies tensors to the CPU for PIL only.
"""

import html
from pathlib import Path
from typing import List, Tuple

from inspection import media_pts
from PIL import Image
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.data import Batch

SCALE = 3


def _image_leaves(value, index: tuple) -> List[Tuple[tuple, ImageData]]:
    if isinstance(value, ImageData):
        return [(index, value)]
    if not isinstance(value, Batch):
        return []

    leaves = []
    for child_index, child in value.iter_with_indices():
        leaves.extend(_image_leaves(child, child_index))

    return leaves


class Gallery:
    """Save selected image fields of chosen groups and index them in HTML.

    Args:
        directory: Destination, or None to disable the gallery.
        fields: ``(group, field)`` pairs to render.
    """

    def __init__(self, directory: Path | None, *, fields: List[Tuple[str, str]]):
        self.directory = directory
        self.fields = fields
        self.figures: List[Tuple[str, str]] = []

    def collect(self, result, session=None) -> None:
        """Handler hook: render the configured fields of one group result.

        Args:
            result: Delivered ``GroupResult``.
            session: Unused; matches the host callback signature.
        """
        if self.directory is None:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        for group, field in self.fields:
            if group != result.group:
                continue
            (entry,) = result.selections[field].values()
            if result.statuses[entry] != "complete":
                continue
            metadata = result.outputs.metadata[entry]
            for index, image in _image_leaves(result.outputs.data[entry], ()):
                label = "-".join(str(part) for part in index) or "root"
                name = f"{group}-{result.pulse.sequence}-{field}-{label}.png"
                self._save(image, self.directory / name)
                pts = media_pts(metadata, index)
                caption = f"{group} #{result.pulse.sequence} {field} {index} PTS={pts}"
                self.figures.append((name, caption))

    def write_index(self) -> Path | None:
        """Write ``index.html`` listing every saved image with its caption.

        Returns:
            Path of the page, or None when the gallery is disabled.
        """
        if self.directory is None:
            return None

        figures = "\n".join(
            f'<figure><img src="{html.escape(name)}"><figcaption>'
            f"{html.escape(caption)}</figcaption></figure>"
            for name, caption in self.figures
        )
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            "<title>Temporal operators gallery</title>"
            "<style>figure{display:inline-block;margin:6px;font:12px sans-serif}"
            "img{image-rendering:pixelated}</style>"
            f"<body><h1>Temporal operators: rig</h1>{figures}</body></html>\n"
        )
        destination = self.directory / "index.html"
        destination.write_text(page)

        return destination

    @staticmethod
    def _save(image: ImageData, path: Path) -> None:
        pixels = image.tensor_image.detach().cpu()
        if pixels.shape[0] == 1:
            pixels = pixels.expand(3, -1, -1)
        array = pixels.permute(1, 2, 0).numpy()
        picture = Image.fromarray(array)
        picture = picture.resize(
            (picture.width * SCALE, picture.height * SCALE), Image.NEAREST
        )
        picture.save(path)
