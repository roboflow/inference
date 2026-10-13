"""Retired identities stay outside the opt-in V2 default catalogue."""

import re

import pytest
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import UnknownBlockError

# These are explicit V2 identities, including never-ported retired families.
# ByteTracker's existing V2 @v1 differs from its latest V1 @v3. Include the
# historical versions and Llama alias so retirement cannot leave an old route.
RETIRED_V2_TYPES = (
    "v2/byte_tracker@v1",
    "v2/byte_tracker@v2",
    "v2/byte_tracker@v3",
    "v2/gaze@v1",
    "v2/llama_vision@v2",
    "v2/llama_3_2_vision@v1",
    "v2/lmm@v1",
    "v2/lmm_for_classification@v1",
    "v2/modbus_tcp@v1",
    "v2/qwen25vl@v1",
    "v2/qwen3_5_openrouter@v1",
    "v2/qwen3_6_openrouter@v1",
    "v2/qwen3vl@v1",
    "v2/sinks@v1",
    "v2/vlm_as_classifier@v2",
    "v2/vlm_as_detector@v2",
    "v2/cog_vlm@v1",
    "v2/yolo_world_model@v1",
)


@pytest.mark.parametrize("block_type", RETIRED_V2_TYPES)
def test_retired_v2_family_is_absent_from_default_catalogue(block_type: str) -> None:
    catalogue = create_catalogue()
    family = block_type.partition("@")[0]

    assert block_type not in catalogue.block_types
    assert all(
        registered.partition("@")[0] != family
        for registered in catalogue.block_types
    )


@pytest.mark.parametrize("block_type", RETIRED_V2_TYPES)
def test_retired_v2_type_fails_compilation_with_named_error(block_type: str) -> None:
    definition = {
        "version": "2.0",
        "inputs": [],
        "steps": [{"type": block_type, "name": "retired_step"}],
        "outputs": [],
    }

    with pytest.raises(UnknownBlockError, match=re.escape(block_type)) as error:
        compile_workflow(definition, catalogue=create_catalogue())

    assert error.value.step_path == ("retired_step",)


@pytest.mark.parametrize(
    "block_type",
    (
        "v2/trackers_bytetrack@v1",
        "v2/trackers_sort@v1",
        "v2/trackers_ocsort@v1",
        "v2/trackers_botsort@v1",
    ),
)
def test_active_trackers_remain_discoverable(block_type: str) -> None:
    assert block_type in create_catalogue().block_types
