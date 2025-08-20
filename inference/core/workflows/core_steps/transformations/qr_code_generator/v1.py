from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
    ImageInputField,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Generate a QR code image from a string input (typically a URL).

This block creates a QR code PNG image from the provided text input. It supports 
various customization options including size control, error correction levels, 
and visual styling. The generated QR code can be used in workflows where you need 
to create QR codes for URLs, text content, or other data that needs to be encoded.

The output is a PNG image that can be passed to other workflow blocks such as 
visualizers or image processing blocks.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "QR Code Generator",
            "version": "v1",
            "short_description": "Generate a QR code image from text input.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fas fa-qrcode",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/qr_code_generator@v1", "QRCodeGenerator"]

    # Main input - the text/URL to encode
    text: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Text or URL to encode in the QR code",
        examples=["https://roboflow.com", "$inputs.url", "Hello World"],
    )

    # Note: version=None and box_size=10 are hardcoded defaults (non-editable per spec)

    # Error correction level - Single-Select (no param linking per spec 9a)
    error_correct: Literal[
        "Low (~7% word recovery / highest data capacity)",
        "Medium (~15% word recovery)",
        "Quartile (~25% word recovery)",
        "High (~30% word recovery / lowest data capacity)",
    ] = Field(
        default="Medium (~15% word recovery)",
        title="Error Correction",
        description="Increased error correction comes at the expense of data capacity (text length). Use higher error correction if the QR code is likely to be transformed or obscured, but use a lower error correction level if the URL is long and the QR code is clearly visible.",
        examples=[
            "Low (~7% word recovery / highest data capacity)",
            "Medium (~15% word recovery)",
            "Quartile (~25% word recovery)",
            "High (~30% word recovery / lowest data capacity)",
        ],
    )

    # Visual styling
    border: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=4,
        title="Border Width",
        description="Border thickness in modules (default: 4)",
        examples=[2, 4, 6, "$inputs.border"],
    )
    fill_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="black",
        title="Fill Color",
        description='Block color, either RGB e.g. (163, 81, 251) or CSS3 color name, e.g. "mediumpurple". Default: black',
        examples=["black", "mediumpurple", "(163, 81, 251)", "$inputs.fill_color"],
    )
    back_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="white",
        title="Background Color",
        description='Background color, either RGB e.g. (255, 255, 255) or CSS3 color name, e.g. "white". Default: white',
        examples=["white", "lightblue", "(255, 255, 255)", "$inputs.back_color"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return []

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="qr_code", kind=[IMAGE_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class QRCodeGeneratorBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        text: str,
        error_correct: Literal[
            "Low (~7% word recovery / highest data capacity)",
            "Medium (~15% word recovery)",
            "Quartile (~25% word recovery)",
            "High (~30% word recovery / lowest data capacity)",
        ] = "Medium (~15% word recovery)",
        border: int = 4,
        fill_color: str = "black",
        back_color: str = "white",
    ) -> BlockResult:

        qr_image = generate_qr_code(
            text=text,
            version=None,  # Auto-sizing as per spec requirement 8a
            box_size=10,  # Fixed default as per spec requirement 8b
            error_correct=error_correct,
            border=border,
            fill_color=fill_color,
            back_color=back_color,
        )

        return {"qr_code": qr_image}


def generate_qr_code(
    text: str,
    version: Optional[int] = None,
    box_size: int = 10,
    error_correct: str = "M",
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
) -> WorkflowImageData:
    """Generate a QR code PNG image from text input."""
    global _ERROR_LEVELS
    try:
        import qrcode
    except ImportError:
        raise ImportError(
            "qrcode library is required for QR code generation. "
            "Install it with: pip install qrcode"
        )
    if _ERROR_LEVELS is None:
        _ERROR_LEVELS = _get_error_levels()

    # Parse colors - handle both color names and RGB tuples
    def parse_color(color_str: str):
        color_str = color_str.strip()
        if color_str and color_str[0] == "(" and color_str[-1] == ")":
            # Parse RGB tuple string like "(255, 0, 0)"
            try:
                rgb_values = color_str[1:-1].split(",")
                return tuple(int(val.strip()) for val in rgb_values)
            except Exception:
                return color_str
        return color_str

    fill = parse_color(fill_color)
    back = parse_color(back_color)

    error_level = _ERROR_LEVELS.get(
        error_correct.upper(), qrcode.constants.ERROR_CORRECT_M
    )

    # Create QR code
    qr = qrcode.QRCode(
        version=version,
        error_correction=error_level,
        box_size=box_size,
        border=border,
    )

    qr.add_data(text)
    qr.make(fit=(version is None))

    # Generate image using default image factory
    img = qr.make_image(
        fill_color=fill,
        back_color=back,
    ).convert(
        "RGB"
    )  # Ensure always RGB

    # Direct conversion from PIL.Image to numpy array (much faster than encode/decode)
    numpy_image = np.array(img)

    # Defensive: numpy_image should never be None; original code checks for None on OpenCV decode failure
    if numpy_image is None or numpy_image.size == 0:
        raise ValueError("Failed to generate QR code image")

    # Create WorkflowImageData
    parent_metadata = ImageParentMetadata(parent_id=f"qr_code.{uuid4()}")
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_image,
    )


def _get_error_levels():
    try:
        import qrcode

        C = qrcode.constants
        return {
            "LOW (~7% WORD RECOVERY / HIGHEST DATA CAPACITY)": C.ERROR_CORRECT_L,
            "MEDIUM (~15% WORD RECOVERY)": C.ERROR_CORRECT_M,
            "QUARTILE (~25% WORD RECOVERY)": C.ERROR_CORRECT_Q,
            "HIGH (~30% WORD RECOVERY / LOWEST DATA CAPACITY)": C.ERROR_CORRECT_H,
            "ERROR_CORRECT_L": C.ERROR_CORRECT_L,
            "ERROR_CORRECT_M": C.ERROR_CORRECT_M,
            "ERROR_CORRECT_Q": C.ERROR_CORRECT_Q,
            "ERROR_CORRECT_H": C.ERROR_CORRECT_H,
            "L": C.ERROR_CORRECT_L,
            "M": C.ERROR_CORRECT_M,
            "Q": C.ERROR_CORRECT_Q,
            "H": C.ERROR_CORRECT_H,
        }
    except ImportError:
        raise ImportError(
            "qrcode library is required for QR code generation. "
            "Install it with: pip install qrcode"
        )


_ERROR_LEVELS = None
