"""The development YOLOv8 view adapter, imported from its example directory.

The adapter lives in ``development/workflows-2.0/24-segmentation-views``.
Importing this module skips the importing test module when that directory is
absent (for example in an installed wheel).
"""

import sys

import pytest

from tests.v2.support.paths import DEVELOPMENT_EXAMPLES

EXAMPLE_DIR = DEVELOPMENT_EXAMPLES / "24-segmentation-views"
if not EXAMPLE_DIR.is_dir():
    pytest.skip(
        "segmentation view example not in this checkout", allow_module_level=True
    )
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from view_adapter import (  # noqa: E402
    SelectionSettings,
    YoloSegmentationViewAdapter,
    check_selection,
    dense_from_rows,
)

__all__ = [
    "SelectionSettings",
    "YoloSegmentationViewAdapter",
    "check_selection",
    "dense_from_rows",
]
