"""Optional HTML gallery of classified images and their top classes.

Pixel export is a host boundary: tensors are copied to the CPU for PIL only.
"""

import html
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData

THUMBNAIL_SIZE = 280


class Gallery:
    """Collect figures with captions and write them as one ``index.html``.

    Args:
        directory: Destination, or None to disable the gallery.
        title: Page heading.
    """

    def __init__(self, directory: Optional[Path], *, title: str):
        self.directory = directory
        self.title = title
        self.sections: List[Tuple[str, List[Tuple[str, List[str]]]]] = []

    def section(self, heading: str) -> None:
        """Start a new group of figures.

        Args:
            heading: Section heading.
        """
        self.sections.append((heading, []))

    def add(self, image: ImageData, *, name: str, lines: List[str]) -> None:
        """Save a thumbnail and its caption lines into the current section.

        Args:
            image: Image to show.
            name: File stem, unique within the gallery.
            lines: Caption lines, for example top classes.
        """
        if self.directory is None:
            return

        self.directory.mkdir(parents=True, exist_ok=True)
        pixels = image.tensor_image.detach().cpu()
        if pixels.shape[0] == 1:
            pixels = pixels.expand(3, -1, -1)
        picture = Image.fromarray(pixels.permute(1, 2, 0).numpy())
        picture.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture.save(self.directory / f"{name}.png")
        self.sections[-1][1].append((f"{name}.png", lines))

    def write_index(self) -> Optional[Path]:
        """Write ``index.html`` with every section and figure.

        Returns:
            Path of the page, or None when the gallery is disabled.
        """
        if self.directory is None:
            return None

        blocks = []
        for heading, figures in self.sections:
            items = "".join(
                f'<figure><img src="{html.escape(file)}"><figcaption>'
                + "<br>".join(html.escape(line) for line in lines)
                + "</figcaption></figure>"
                for file, lines in figures
            )
            blocks.append(f"<h2>{html.escape(heading)}</h2>{items}")
        page = (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            f"<title>{html.escape(self.title)}</title>"
            "<style>body{font:13px sans-serif;margin:16px}"
            "figure{display:inline-block;vertical-align:top;margin:6px;width:290px}"
            "figcaption{margin-top:4px}</style>"
            f"<body><h1>{html.escape(self.title)}</h1>{''.join(blocks)}</body></html>\n"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / "index.html"
        destination.write_text(page)

        return destination
