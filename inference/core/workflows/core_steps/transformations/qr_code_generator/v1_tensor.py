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
Generate QR code images from text input (URLs, text content, or other data) with customizable error correction levels, visual styling (colors, borders), and automatic size optimization, producing PNG images suitable for embedding, printing, or further image processing in workflows.

## How This Block Works

This block creates QR code images from text input using the qrcode library, encoding the provided text into a scannable QR code pattern. The block:

1. Receives text input (URLs, strings, or other data to encode) and QR code configuration parameters
2. Checks an internal LRU cache for previously generated QR codes with the same parameters (caching improves performance for repeated QR code generation)
3. Parses color specifications for fill and background colors (supports hex codes, RGB strings, standard color names, and CSS3 color names) using a common color utility
4. Maps the error correction level string to the corresponding qrcode library constant (Low, Medium, Quartile, or High error correction)
5. Creates a QRCode object with the specified parameters:
   - Auto-determines version (size) based on data length when version is None
   - Sets error correction level based on the selected option
   - Uses fixed box_size of 10 pixels per module
   - Applies the specified border width
6. Encodes the text data into the QR code pattern
7. Generates a PIL Image from the QR code with the specified fill and background colors
8. Converts the image to RGB format, then to a NumPy array
9. Converts from RGB (PIL format) to BGR (OpenCV/WorkflowImageData format) for compatibility with workflow image processing
10. Creates a WorkflowImageData object with the QR code image and metadata
11. Stores the result in the cache for future reuse (cache has 100 entry capacity and 1-hour TTL)
12. Returns the QR code image

The block automatically optimizes QR code size based on the data length (auto-sizing), ensuring the QR code is large enough to encode the data but not unnecessarily large. Error correction levels trade off between data capacity and error recovery: higher error correction allows the QR code to be scanned even if partially damaged or obscured, but reduces the maximum data capacity. The block uses caching to improve performance when generating the same QR codes multiple times, which is common in workflows that generate QR codes for the same URLs or data repeatedly.

## Common Use Cases

- **URL and Link Encoding**: Generate QR codes for URLs and web links (e.g., create QR codes for product pages, generate QR codes for documentation links, encode URLs for easy mobile access), enabling quick access to web resources via QR code scanning
- **Data Encoding and Sharing**: Encode text data, identifiers, or information into QR codes (e.g., generate QR codes for product IDs, encode serial numbers, create QR codes for inventory tracking), enabling machine-readable data encoding and sharing
- **Document and Report Generation**: Include QR codes in generated documents or reports (e.g., add QR codes to PDF reports linking to detailed data, embed QR codes in generated images, include QR codes in formatted outputs), enabling interactive document features with scannable links
- **Workflow Result Sharing**: Generate QR codes linking to workflow results or outputs (e.g., create QR codes pointing to detection results, encode links to analysis reports, generate QR codes for sharing workflow outputs), enabling easy sharing and access to workflow-generated content
- **Label and Tag Generation**: Create QR codes for labeling and identification purposes (e.g., generate QR codes for asset tags, create QR codes for product labels, encode identification information), enabling automated label and tag creation workflows
- **Integration and Automation**: Generate QR codes as part of automated workflows (e.g., create QR codes for automated document processing, generate QR codes for workflow automation triggers, encode data for system integration), enabling QR code generation as part of automated processes

## Connecting to Other Blocks

This block receives text input and produces QR code images:

- **After data processing blocks** (e.g., Expression, Property Definition) that produce text output to encode computed values, URLs, or processed data into QR codes, enabling machine-readable encoding of workflow-generated data
- **Before visualization blocks** that can overlay or combine QR codes with other images (e.g., overlay QR codes on images, combine QR codes with detection visualizations, embed QR codes in composite images), enabling QR code integration into visual outputs
- **Before formatter blocks** (e.g., CSV Formatter) to include QR code references or links in formatted outputs, enabling QR code integration into structured data exports
- **Before sink blocks** (e.g., Local File Sink, Webhook Sink) to save or send generated QR code images, enabling QR code distribution and storage
- **In document generation workflows** where QR codes need to be embedded in documents or reports, enabling interactive document features with scannable codes
- **After detection or analysis blocks** to generate QR codes linking to detection results or analysis outputs, enabling easy sharing and access to workflow results via QR code scanning
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
        description="Text, URL, or data to encode into the QR code. Can be any string content including URLs (e.g., 'https://roboflow.com'), text messages, identifiers, serial numbers, or other data. The QR code will automatically size itself based on the data length. Longer text requires larger QR codes or lower error correction levels to fit. Maximum data capacity depends on the error correction level selected.",
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
        description="Error correction level that determines how much damage or obscuration the QR code can tolerate while still being scannable. Higher error correction allows scanning even if the QR code is partially damaged, obscured, or transformed, but reduces the maximum data capacity (text length). Choose 'Low' for maximum data capacity when QR codes will be clearly visible and undamaged. Choose 'Medium' (default) for balanced capacity and error recovery. Choose 'Quartile' or 'High' when QR codes may be partially obscured, damaged, or need to be scanned from difficult angles. Trade-off: higher error correction = better error recovery but less data capacity.",
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
        description="Border thickness in modules (QR code units). Defaults to 4 modules. The border is a quiet zone around the QR code pattern that helps QR code scanners identify and decode the code. Larger borders improve scanning reliability but increase image size. Minimum recommended border is 4 modules. Border is measured in QR code modules (not pixels - actual pixel border size depends on box_size which is fixed at 10 pixels per module).",
        examples=[2, 4, 6, "$inputs.border"],
    )
    fill_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="BLACK",
        title="Fill Color",
        description="Color of the QR code pattern blocks (the dark squares in the QR code). Defaults to 'BLACK'. Supports multiple color formats: hex codes (e.g., '#FF0000' for red, '#000000' for black), RGB strings (e.g., 'rgb(255, 0, 0)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The fill color should contrast well with the background color for reliable scanning. Traditional black-on-white provides the best scanning reliability.",
        examples=["BLACK", "#000000", "rgb(0, 0, 0)", "$inputs.fill_color"],
    )
    back_color: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="WHITE",
        title="Background Color",
        description="Background color of the QR code (the light areas between pattern blocks). Defaults to 'WHITE'. Supports multiple color formats: hex codes (e.g., '#FFFFFF' for white, '#000000' for black), RGB strings (e.g., 'rgb(255, 255, 255)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The background color should contrast well with the fill color for reliable scanning. Traditional white background with black fill provides the best scanning reliability.",
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
