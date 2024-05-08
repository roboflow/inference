from inference.enterprise.workflows.core_steps.transformations.crop import CropBlock
from inference.enterprise.workflows.execution_engine.introspection.utils import (
    build_human_friendly_block_name,
    get_full_type_name,
)


def test_get_full_type_name() -> None:
    # when
    type_name = get_full_type_name(selected_type=CropBlock)

    # then
    assert (
        type_name
        == "inference.enterprise.workflows.core_steps.transformations.crop.CropBlock"
    )


def test_build_human_friendly_block_name_when_block_suffix_present() -> None:
    # when
    result = build_human_friendly_block_name(
        fully_qualified_name="inference.enterprise.workflows.core_steps.transformations.crop.MyCropBlock"
    )

    # then
    assert result == "My Crop"


def test_build_human_friendly_block_name_when_block_suffix_not_present() -> None:
    # when
    result = build_human_friendly_block_name(
        fully_qualified_name="inference.enterprise.workflows.core_steps.transformations.crop.MyCrop"
    )

    # then
    assert result == "My Crop"
