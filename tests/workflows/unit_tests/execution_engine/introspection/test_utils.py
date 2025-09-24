from inference.core.workflows.core_steps.transformations.dynamic_crop.v1 import (
    DynamicCropBlockV1,
)
from inference.core.workflows.execution_engine.introspection.utils import (
    build_human_friendly_block_name,
    get_full_type_name,
)


def test_get_full_type_name() -> None:
    # when
    type_name = get_full_type_name(selected_type=DynamicCropBlockV1)

    # then
    assert (
        type_name
        == "inference.core.workflows.core_steps.transformations.dynamic_crop.v1.DynamicCropBlockV1"
    )


def test_build_human_friendly_block_name_when_overridden_in_schema() -> None:
    # when
    result = build_human_friendly_block_name(
        fully_qualified_name="inference.core.workflows.core_steps.transformations.crop.MyCropBlock",
        block_schema={"name": "Foo Bar"},
    )

    # then
    assert result == "Foo Bar"


def test_build_human_friendly_block_name_when_block_suffix_present() -> None:
    # when
    result = build_human_friendly_block_name(
        fully_qualified_name="inference.core.workflows.core_steps.transformations.dynamic_crop.MyCropBlock"
    )

    # then
    assert result == "My Crop"


def test_build_human_friendly_block_name_when_block_suffix_not_present() -> None:
    # when
    result = build_human_friendly_block_name(
        fully_qualified_name="inference.core.workflows.core_steps.transformations.dynamic_crop.MyCrop"
    )

    # then
    assert result == "My Crop"
