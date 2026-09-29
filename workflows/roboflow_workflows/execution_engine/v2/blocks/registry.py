"""Explicit registry of the native V2 CPU image blocks.

Nothing in the generic V2 engine imports this module. A caller that wants the
native catalogue builds it explicitly::

    from roboflow_workflows.execution_engine.v2.blocks import native_registry

    registry = native_registry()

Each call returns a fresh, independent :class:`Registry`, so a caller may add
its own kinds and blocks without affecting other callers. The legacy V1 block
loader does not discover these blocks.
"""

from roboflow_workflows.execution_engine.v2.blocks.image import (
    CROP_BLOCK_NAME,
    CROP_CONTRACT,
    HAS_BRIGHTNESS_BLOCK_NAME,
    HAS_BRIGHTNESS_CONTRACT,
    INVERT_BLOCK_NAME,
    INVERT_CONTRACT,
    MOSAIC_BLOCK_NAME,
    MOSAIC_CONTRACT,
    CropBlock,
    HasBrightnessBlock,
    InvertBlock,
    MosaicBlock,
)
from roboflow_workflows.execution_engine.v2.blocks.kinds import (
    BOOLEAN_KIND,
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    is_boolean_payload,
    is_crop_summary_payload,
    is_image_payload,
    is_integer_payload,
)
from roboflow_workflows.execution_engine.v2.contracts import Registry

NATIVE_KIND_NAMES = (IMAGE_KIND, BOOLEAN_KIND, INTEGER_KIND, CROP_SUMMARY_KIND)
NATIVE_BLOCK_NAMES = (
    CROP_BLOCK_NAME,
    INVERT_BLOCK_NAME,
    MOSAIC_BLOCK_NAME,
    HAS_BRIGHTNESS_BLOCK_NAME,
)


def native_registry() -> Registry:
    """Build a fresh registry holding the native V2 CPU image catalogue.

    Registers the ``image``, ``boolean``, ``integer`` and ``crop_summary``
    kinds with their payload validators and the ``v2/crop``, ``v2/invert``,
    ``v2/mosaic`` and ``v2/has_brightness`` blocks.

    Returns:
        New registry instance; callers may register additional kinds and
        blocks on it before compiling a workflow.
    """
    registry = Registry()

    registry.register_kind(IMAGE_KIND, validator=is_image_payload)
    registry.register_kind(BOOLEAN_KIND, validator=is_boolean_payload)
    registry.register_kind(INTEGER_KIND, validator=is_integer_payload)
    registry.register_kind(CROP_SUMMARY_KIND, validator=is_crop_summary_payload)

    registry.register_block(
        CROP_BLOCK_NAME, contract=CROP_CONTRACT, factory=CropBlock.from_config
    )
    registry.register_block(
        INVERT_BLOCK_NAME, contract=INVERT_CONTRACT, factory=InvertBlock.from_config
    )
    registry.register_block(
        MOSAIC_BLOCK_NAME, contract=MOSAIC_CONTRACT, factory=MosaicBlock.from_config
    )
    registry.register_block(
        HAS_BRIGHTNESS_BLOCK_NAME,
        contract=HAS_BRIGHTNESS_CONTRACT,
        factory=HasBrightnessBlock.from_config,
    )

    return registry
