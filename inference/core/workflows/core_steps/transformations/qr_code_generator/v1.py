import threading
import time
from collections import OrderedDict
from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
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
            "access_third_party": False,
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
        default="BLACK",
        title="Fill Color",
        description="QR code block color. Supports hex (#FF0000), rgb(255, 0, 0), standard names (BLACK, WHITE, RED, etc.), or CSS3 color names.",
        examples=["BLACK", "#000000", "rgb(0, 0, 0)", "$inputs.fill_color"],
    )
    back_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="WHITE",
        title="Background Color",
        description="QR code background color. Supports hex (#FFFFFF), rgb(255, 255, 255), standard names (BLACK, WHITE, RED, etc.), or CSS3 color names.",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.back_color"],
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
        fill_color: str = "BLACK",
        back_color: str = "WHITE",
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
    fill_color: str = "BLACK",
    back_color: str = "WHITE",
) -> WorkflowImageData:
    """Generate a QR code PNG image from text input."""
    global _ERROR_LEVELS, _QR_CACHE

    # Check cache first
    cached_result = _QR_CACHE.get(
        text, version, box_size, error_correct, border, fill_color, back_color
    )
    if cached_result is not None:
        return cached_result

    try:
        import qrcode
    except ImportError:
        raise ImportError(
            "qrcode library is required for QR code generation. "
            "Install it with: pip install qrcode"
        )
    if _ERROR_LEVELS is None:
        _ERROR_LEVELS = _get_error_levels()

    # Parse colors using the common utility that handles hex, rgb, bgr, and standard names
    try:
        # Convert to supervision Color object, then to RGB tuple for qrcode library
        fill_sv_color = str_to_color(fill_color)
        fill = fill_sv_color.as_rgb()  # Returns (R, G, B) tuple
    except (ValueError, AttributeError):
        # Fallback to original string if not a recognized format
        # This allows qrcode library to handle CSS3 color names directly
        fill = fill_color

    try:
        back_sv_color = str_to_color(back_color)
        back = back_sv_color.as_rgb()  # Returns (R, G, B) tuple
    except (ValueError, AttributeError):
        # Fallback to original string if not a recognized format
        back = back_color

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

    # Convert from RGB (PIL format) to BGR (OpenCV/WorkflowImageData format)
    # PIL creates RGB images, but WorkflowImageData expects BGR format
    numpy_image = numpy_image[:, :, ::-1]  # RGB -> BGR

    # Defensive: numpy_image should never be None; original code checks for None on OpenCV decode failure
    if numpy_image is None or numpy_image.size == 0:
        raise ValueError("Failed to generate QR code image")

    # Create WorkflowImageData
    parent_metadata = ImageParentMetadata(parent_id=f"qr_code.{uuid4()}")
    result = WorkflowImageData(
        parent_metadata=parent_metadata,
        numpy_image=numpy_image,
    )

    # Store in cache
    _QR_CACHE.put(
        text, version, box_size, error_correct, border, fill_color, back_color, result
    )

    return result


class _QRCodeLRUCache:
    """LRU Cache with TTL for QR code generation results."""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._lock = threading.RLock()

    def _make_key(
        self,
        text: str,
        version: Optional[int],
        box_size: int,
        error_correct: str,
        border: int,
        fill_color: str,
        back_color: str,
    ) -> str:
        """Create cache key from QR code parameters."""
        return f"{text}|{version}|{box_size}|{error_correct}|{border}|{fill_color}|{back_color}"

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds

    def _cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]

    def get(
        self,
        text: str,
        version: Optional[int],
        box_size: int,
        error_correct: str,
        border: int,
        fill_color: str,
        back_color: str,
    ) -> Optional:
        """Get cached QR code result if available and not expired."""
        key = self._make_key(
            text, version, box_size, error_correct, border, fill_color, back_color
        )

        with self._lock:
            if key in self._cache:
                result, timestamp = self._cache[key]
                if not self._is_expired(timestamp):
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return result
                else:
                    # Remove expired entry
                    del self._cache[key]

        return None

    def put(
        self,
        text: str,
        version: Optional[int],
        box_size: int,
        error_correct: str,
        border: int,
        fill_color: str,
        back_color: str,
        result,
    ) -> None:
        """Store QR code result in cache."""
        key = self._make_key(
            text, version, box_size, error_correct, border, fill_color, back_color
        )

        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 10 == 0:  # Every 10th insertion
                self._cleanup_expired()

            # Remove oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest (FIFO when at capacity)

            # Add new entry
            self._cache[key] = (result, time.time())


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
_QR_CACHE = _QRCodeLRUCache(max_size=100, ttl_seconds=3600)
