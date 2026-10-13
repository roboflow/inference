"""Generate a QR code image from text with the ``qrcode`` library.

The image is drawn on the CPU, as in V1, and returned as an RGB ``uint8``
CPU ``ImageData``: a new workflow root with its own ``image_id``. The QR
version grows with the text; each module is 10 pixels, plus ``border``
modules of quiet zone on every side::

    side = (17 + 4 * version + 2 * border) * 10 pixels

Colors use the V1 grammar: ``#RRGGBB``, ``rgb(R, G, B)``, ``bgr(B, G, R)``
or a supervision color name (``BLACK``, ``WHITE``, ``GREEN`` = ``#00FF00``,
...); any other name falls back to the CSS names Pillow knows (``navy``).

Differences from V1:

* No result cache. V1 handed the same image object to every call with the
  same parameters for an hour; here every call returns a new image, so no
  caller can change another's pixels. Generating a short text takes a few
  milliseconds on a CPU.
* ``border`` must be in ``[0, 100]`` modules; V1 accepted any integer and
  failed inside ``qrcode`` (negative) or could allocate an unbounded image.
* Too much text for the chosen error correction is a clear ``ValueError``
  naming the limit, not ``qrcode``'s "Invalid version".
"""

from typing import Annotated, Any, Dict, Literal, Tuple
from uuid import uuid4

import numpy as np
import qrcode
from PIL import ImageColor
from pydantic import Field
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

ErrorCorrection = Literal[
    "Low (~7% word recovery / highest data capacity)",
    "Medium (~15% word recovery)",
    "Quartile (~25% word recovery)",
    "High (~30% word recovery / lowest data capacity)",
]
ERROR_CORRECTION_LEVELS = {
    "Low (~7% word recovery / highest data capacity)": (
        qrcode.constants.ERROR_CORRECT_L
    ),
    "Medium (~15% word recovery)": qrcode.constants.ERROR_CORRECT_M,
    "Quartile (~25% word recovery)": qrcode.constants.ERROR_CORRECT_Q,
    "High (~30% word recovery / lowest data capacity)": (
        qrcode.constants.ERROR_CORRECT_H
    ),
}
BOX_SIZE = 10
MAX_BORDER = 100
Border = Annotated[int, Field(ge=0, le=MAX_BORDER)]
RGB = Tuple[int, int, int]

__all__ = ["QRCodeGeneratorBlock"]


class QRCodeGeneratorBlock(Block):
    """Draw text as a QR code image.

    Stateless and prunable. Every call returns a new CPU image.
    """

    type = "v2/qr_code_generator@v1"
    prunable = True
    metadata = {
        "name": "QR Code Generator",
        "port_of": "roboflow_core/qr_code_generator@v1",
        "section": "transformation",
    }
    outputs = {
        "qr_code": Output(
            IMAGE_KIND,
            description="RGB QR code image on the CPU, a new workflow root.",
        )
    }

    class Params(BlockParams):
        text: str | Ref(STRING_KIND) = Field(
            description=(
                "Text, URL or identifier to encode. Longer text makes a "
                "larger code; the error correction level limits the length."
            ),
            examples=["https://roboflow.com", "$inputs.url"],
        )
        error_correct: ErrorCorrection = Field(
            default="Medium (~15% word recovery)",
            description=(
                "How much of the code can be damaged and still read. Higher "
                "levels hold less text."
            ),
            examples=["Medium (~15% word recovery)"],
        )
        border: Border | Ref(INTEGER_KIND) = Field(
            default=4,
            description=(
                f"Quiet zone in modules (10 px each), 0 to {MAX_BORDER}. "
                "Scanners need about 4."
            ),
            examples=[4, "$inputs.border"],
        )
        fill_color: str | Ref(STRING_KIND) = Field(
            default="BLACK",
            description=(
                "Color of the dark modules: #RRGGBB, rgb(R, G, B), bgr(B, G, R), "
                "a supervision color name or a CSS color name."
            ),
            examples=["BLACK", "#000000", "rgb(0, 0, 0)", "$inputs.fill_color"],
        )
        back_color: str | Ref(STRING_KIND) = Field(
            default="WHITE",
            description="Color of the light modules, in the same formats.",
            examples=["WHITE", "#FFFFFF", "$inputs.back_color"],
        )

    def run(
        self,
        *,
        text: str,
        error_correct: str,
        border: int,
        fill_color: str,
        back_color: str,
    ) -> Dict[str, Any]:
        """Draw ``text`` as a QR code.

        Args:
            text: Content to encode.
            error_correct: One of the four V1 error correction labels.
            border: Quiet zone in modules, ``0`` to ``MAX_BORDER``.
            fill_color: Color string of the dark modules.
            back_color: Color string of the light modules.

        Returns:
            ``qr_code``: the RGB image as a new CPU ``ImageData``.

        Raises:
            ValueError: On a border outside its range, an unknown color or
                text that does not fit any QR version.
        """
        _check_border(border)
        fill = qr_color(fill_color, field="fill_color")
        back = qr_color(back_color, field="back_color")

        pixels = draw_qr_code(
            text,
            error_correction=ERROR_CORRECTION_LEVELS[error_correct],
            border=border,
            fill=fill,
            back=back,
        )
        image = ImageData.from_numpy_rgb(pixels, image_id=f"qr_code-{uuid4().hex}")
        result = {"qr_code": image}

        return result


def draw_qr_code(
    text: str, *, error_correction: int, border: int, fill: RGB, back: RGB
) -> np.ndarray:
    """Draw ``text`` as QR code pixels at ``BOX_SIZE`` pixels per module.

    Args:
        text: Content to encode.
        error_correction: A ``qrcode.constants.ERROR_CORRECT_*`` level.
        border: Quiet zone in modules.
        fill: RGB color of the dark modules.
        back: RGB color of the light modules.

    Returns:
        ``(side, side, 3)`` ``uint8`` RGB array.

    Raises:
        ValueError: When ``text`` does not fit version 40 at this level.
    """
    code = qrcode.QRCode(
        version=None,
        error_correction=error_correction,
        box_size=BOX_SIZE,
        border=border,
    )
    code.add_data(text)
    try:
        code.make(fit=True)
    except ValueError as error:
        # qrcode reports overflow as "Invalid version (was 41, ...)".
        raise ValueError(
            f"text of {len(text)} characters does not fit in a QR code at this "
            "error correction level; shorten it or lower error_correct"
        ) from error

    drawn = code.make_image(fill_color=fill, back_color=back).convert("RGB")
    pixels = np.asarray(drawn, dtype=np.uint8)

    return pixels


def qr_color(text: str, *, field: str) -> RGB:
    """Read a color in the V1 grammar, then as a CSS color name.

    Args:
        text: Color string.
        field: Parameter name for the error message.

    Returns:
        The color as an RGB triple.

    Raises:
        ValueError: When neither grammar reads ``text``.
    """
    try:
        color = parse_color(text, field=field)
    except ValueError as error:
        try:
            color = ImageColor.getrgb(text)[:3]
        except (ValueError, AttributeError):
            raise ValueError(f"{error}, or a CSS color name") from None

    return color


def _check_border(border: Any) -> None:
    if isinstance(border, bool) or not isinstance(border, int):
        raise ValueError(f"border must be an integer, got {border!r}")
    if not 0 <= border <= MAX_BORDER:
        raise ValueError(f"border must be in [0, {MAX_BORDER}] modules, got {border}")
