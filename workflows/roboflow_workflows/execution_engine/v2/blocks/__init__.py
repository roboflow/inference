"""Native V2 CPU image block catalogue.

Callers select these blocks explicitly::

    from roboflow_workflows.execution_engine.v2.blocks import create_catalogue

    catalogue = create_catalogue()

``Catalogue.from_modules(["roboflow_workflows.execution_engine.v2.blocks"])``
reads the same catalogue through ``WORKFLOWS_V2_CATALOGUE``. The generic V2
engine never imports this package, and the legacy V1 loader does not discover
it. Every contract detail lives on the block classes themselves.
"""

from roboflow_workflows.execution_engine.v2.blocks.control import ContinueIfBlock
from roboflow_workflows.execution_engine.v2.blocks.image import (
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue


def create_catalogue() -> Catalogue:
    """Collect the native image blocks and their media kinds.

    Returns:
        Immutable catalogue with ``v2/crop``, ``v2/invert``, ``v2/mosaic``,
        ``v2/has_brightness`` and ``v2/continue_if``, the ``image`` and
        ``crop_summary`` kinds, and the built-in kinds these blocks use.
    """
    catalogue = Catalogue(
        [CropBlock, InvertBlock, MosaicBlock, HasBrightnessBlock, ContinueIfBlock],
        kinds=[IMAGE_KIND, CROP_SUMMARY_KIND],
    )

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "CROP_SUMMARY_KIND",
    "ContinueIfBlock",
    "CropBlock",
    "HasBrightnessBlock",
    "IMAGE_KIND",
    "InvertBlock",
    "MosaicBlock",
    "WORKFLOWS_V2_CATALOGUE",
    "create_catalogue",
]
