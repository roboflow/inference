"""Colors of the annotation blocks: V1 palettes, color strings, per-row lookup.

Colors are RGB ``uint8`` triples, because V2 images are RGB. V1 painted the
same colors into BGR images, so the pixels match.

Palette by row, as V1 ``ColorableVisualizationBlock`` with
``NegativeSafeColorPalette``::

    axis    palette index of row i
    CLASS   class_id[i]
    INDEX   i
    TRACK   bboxes_metadata[i]["tracker_id"]

    color = GREY                         if index < 0
            palette[index % len(palette)] otherwise

``palette_indices`` applies this rule to host arrays and to device tensors
alike; device indices stay on their device.

Palette data comes from supervision, as in V1. Resolved palettes are cached
by their parameters.
"""

from functools import lru_cache
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import supervision as sv
import torch

RGB = Tuple[int, int, int]

PALETTE_NAMES = (
    "DEFAULT",
    "CUSTOM",
    "ROBOFLOW",
    "Matplotlib Viridis",
    "Matplotlib Plasma",
    "Matplotlib Inferno",
    "Matplotlib Magma",
    "Matplotlib Cividis",
    "Matplotlib Pastel1",
    "Matplotlib Pastel2",
    "Matplotlib Paired",
    "Matplotlib Accent",
    "Matplotlib Dark2",
    "Matplotlib Set1",
    "Matplotlib Set2",
    "Matplotlib Set3",
    "Matplotlib Tab10",
    "Matplotlib Tab20",
    "Matplotlib Tab20b",
    "Matplotlib Tab20c",
)
COLOR_AXES = ("INDEX", "CLASS", "TRACK")
UNKNOWN_COLOR: RGB = sv.Color.GREY.as_rgb()

# V1 capitalizes these Matplotlib names and lower-cases the others.
_CAPITALIZED_MATPLOTLIB = (
    "Greys_R",
    "Purples_R",
    "Blues_R",
    "Greens_R",
    "Oranges_R",
    "Reds_R",
    "Wistia",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
)
_COLOR_FORMATS = (
    "#RRGGBB, rgb(R, G, B), bgr(B, G, R) or a supervision color name "
    "(WHITE, BLACK, RED, GREEN, BLUE, YELLOW, GREY, ROBOFLOW)"
)


def parse_color(text: str, *, field: str) -> RGB:
    """Read a V1 color string.

    Args:
        text: ``#RRGGBB`` (also the short and alpha hex forms supervision
            reads), ``rgb(R, G, B)``, ``bgr(B, G, R)`` or a supervision
            color name in any case.
        field: Parameter name for the error message.

    Returns:
        The color as an RGB triple.

    Raises:
        ValueError: When ``text`` is in none of these formats.
    """
    if not isinstance(text, str):
        raise ValueError(f"{field} must be a color string, got {text!r}")

    try:
        if text.startswith("#"):
            color = sv.Color.from_hex(text)
        elif text.startswith("rgb"):
            red, green, blue = (int(part) for part in text[4:-1].split(","))
            color = sv.Color.from_rgb_tuple((red, green, blue))
        elif text.startswith("bgr"):
            blue, green, red = (int(part) for part in text[4:-1].split(","))
            color = sv.Color.from_bgr_tuple((blue, green, red))
        elif text.upper().isidentifier() and isinstance(
            getattr(sv.Color, text.upper(), None), sv.Color
        ):
            color = getattr(sv.Color, text.upper())
        else:
            raise ValueError(text)
    except ValueError:
        raise ValueError(
            f"{field}: invalid color {text!r}; use {_COLOR_FORMATS}"
        ) from None

    rgb = color.as_rgb()

    return rgb


def resolve_palette(
    name: str, *, palette_size: int, custom_colors: Sequence[str]
) -> np.ndarray:
    """Return the colors of a V1 palette.

    Args:
        name: One of ``PALETTE_NAMES``.
        palette_size: Number of colors sampled from a Matplotlib palette.
        custom_colors: Color strings of the ``CUSTOM`` palette.

    Returns:
        Read-only ``(P, 3)`` ``uint8`` RGB colors, ``P >= 1``.

    Raises:
        ValueError: On an unknown name, a palette size below 1, or a
            ``CUSTOM`` palette without valid colors.
    """
    if name not in PALETTE_NAMES:
        raise ValueError(
            f"color_palette must be one of {list(PALETTE_NAMES)}, got {name!r}"
        )
    checked_palette_size(palette_size)
    if name == "CUSTOM":
        if not custom_colors:
            raise ValueError(
                "color_palette CUSTOM needs custom_colors, e.g. "
                '["#FF0000", "#00FF00"]'
            )
        if isinstance(custom_colors, str):
            raise ValueError(
                f"custom_colors must be a list of colors, got {custom_colors!r}"
            )
        custom = tuple(custom_colors)
    else:
        custom = ()

    palette = _palette(name, palette_size, custom)

    return palette


def checked_palette_size(value: Any) -> int:
    """Return a ``palette_size`` after checking it is an integer >= 1.

    Raises:
        ValueError: For booleans, non-integers and values below 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"palette_size must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"palette_size must be at least 1, got {value}")

    return value


def palette_indices(
    axis: str,
    *,
    class_id: Union[np.ndarray, torch.Tensor],
    tracker_id: Optional[Sequence[int]],
) -> Union[np.ndarray, torch.Tensor]:
    """Return each row's palette index for a color axis.

    A device tensor of class ids is never read on the host: ``CLASS`` and
    ``INDEX`` indices are built on its device, and host tracker ids are
    copied there.

    Args:
        axis: ``CLASS``, ``INDEX`` or ``TRACK``.
        class_id: ``(N,)`` integer class ids, a NumPy array or a tensor.
        tracker_id: ``(N,)`` host integer tracker ids, or ``None`` when the
            rows have none.

    Returns:
        ``(N,)`` ``int64`` indices of ``class_id``'s kind and device;
        negative means unknown.

    Raises:
        ValueError: On an unknown axis, or ``TRACK`` without tracker ids.
    """
    if axis not in COLOR_AXES:
        raise ValueError(f"color_axis must be one of {list(COLOR_AXES)}, got {axis!r}")
    if axis == "TRACK" and tracker_id is None:
        raise ValueError(
            "color_axis TRACK needs a tracker_id in every row's bboxes_metadata; "
            "add a tracker step before this one, or use color_axis CLASS"
        )

    on_device = isinstance(class_id, torch.Tensor)
    if axis == "CLASS":
        indices = (
            class_id.to(torch.int64)
            if on_device
            else np.asarray(class_id, dtype=np.int64)
        )
    elif axis == "INDEX":
        indices = (
            torch.arange(len(class_id), dtype=torch.int64, device=class_id.device)
            if on_device
            else np.arange(len(class_id), dtype=np.int64)
        )
    else:
        indices = np.asarray(tracker_id, dtype=np.int64)
        if on_device:
            indices = torch.from_numpy(indices).to(class_id.device)

    return indices


def colors_of(palette: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Look up row colors; negative indices are grey.

    Args:
        palette: ``(P, 3)`` ``uint8`` RGB colors.
        indices: ``(N,)`` integer palette indices.

    Returns:
        New ``(N, 3)`` ``uint8`` RGB colors.
    """
    indices = np.asarray(indices, dtype=np.int64)
    colors = palette[np.mod(indices, len(palette))]
    colors[indices < 0] = UNKNOWN_COLOR

    return colors


@lru_cache(maxsize=64)
def _palette(name: str, palette_size: int, custom: Tuple[str, ...]) -> np.ndarray:
    if name == "CUSTOM":
        rows = [
            parse_color(color, field=f"custom_colors[{index}]")
            for index, color in enumerate(custom)
        ]
    elif name in ("DEFAULT", "ROBOFLOW"):
        rows = [color.as_rgb() for color in getattr(sv.ColorPalette, name).colors]
    else:
        matplotlib_name = name.replace("Matplotlib ", "")
        if matplotlib_name in _CAPITALIZED_MATPLOTLIB:
            matplotlib_name = matplotlib_name.capitalize()
        else:
            matplotlib_name = matplotlib_name.lower()
        sampled = sv.ColorPalette.from_matplotlib(matplotlib_name, palette_size)
        rows = [color.as_rgb() for color in sampled.colors]

    palette = np.asarray(rows, dtype=np.uint8).reshape(-1, 3)
    palette.flags.writeable = False

    return palette
