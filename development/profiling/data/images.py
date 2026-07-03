from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from development.profiling.data.base import DataRecord


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageDirectoryDataSource:
    """Local image file data source for profiling runs."""

    paths: tuple[Path, ...]
    decode: bool = True
    limit: int | None = None
    repeat: int | None = None

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any] | None = None
    ) -> "ImageDirectoryDataSource":
        """Build an image data source from config values.

        Args:
            config (Mapping[str, Any] | None): Optional source-specific config.

        Returns:
            Configured local image data source.
        """
        config = config or {}
        paths = _resolve_paths(config)
        limit = config.get("limit")
        repeat = config.get("repeat")
        data_source = cls(
            paths=paths,
            decode=bool(config.get("decode", True)),
            limit=None if limit is None else int(limit),
            repeat=None if repeat is None else int(repeat),
        )

        return data_source

    def iter_records(self):
        """Iterate over selected local image records.

        Returns:
            Generator of image records.
        """
        selected_paths = self.paths
        if self.limit is not None:
            selected_paths = selected_paths[: self.limit]

        emitted_paths = selected_paths
        if self.repeat is not None:
            emitted_paths = tuple(islice(cycle(selected_paths), self.repeat))

        path_counts: dict[Path, int] = {}
        for path in emitted_paths:
            repeat_index = path_counts.get(path, 0)
            path_counts[path] = repeat_index + 1
            image = _decode_image(path) if self.decode else None
            record_id = path.stem
            if repeat_index:
                record_id = f"{path.stem}-repeat-{repeat_index}"

            yield DataRecord(
                id=record_id,
                image=image,
                path=path,
                source={
                    "type": "images",
                    "path": str(path),
                    "repeat_index": repeat_index,
                },
            )

    def describe(self) -> Mapping[str, Any]:
        """Describe this image data source.

        Returns:
            Manifest metadata for the selected image paths.
        """
        description = {
            "type": "images",
            "paths": [str(path) for path in self.paths],
            "decode": self.decode,
            "limit": self.limit,
            "repeat": self.repeat,
            "decoded_format": "rgb_numpy_array" if self.decode else None,
        }

        return description


def _resolve_paths(config: Mapping[str, Any]) -> tuple[Path, ...]:
    configured_paths = config.get("paths")
    if configured_paths is not None:
        paths = tuple(Path(path) for path in _as_sequence(configured_paths))
    else:
        directory = config.get("directory") or config.get("images_dir")
        if directory is None:
            raise ValueError("Image data source requires 'directory' or 'paths'.")
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Image directory does not exist: {directory_path}")

        paths = tuple(
            path
            for path in sorted(directory_path.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not paths:
        raise ValueError("Image data source did not resolve any image paths.")

    return paths


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, Path)):
        return (value,)

    return value


def _decode_image(path: Path) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ImportError:
        Image = None

    if Image is not None:
        with Image.open(path) as image:
            decoded_image = np.asarray(image.convert("RGB").copy())

        return decoded_image

    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")

    decoded_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return decoded_image
