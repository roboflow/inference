import numpy as np

from inference.core.workflows.core_steps.formatters.vlm_as_detector.muse_detection_parsing import (
    convert_muse_detection_to_pixel_xyxy,
    extract_flat_object_entries,
    parse_muse_object_detection_response,
)
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _build_image(height: int, width: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def test_parse_named_0_to_1000_fields():
    result = parse_muse_object_detection_response(
        image=_build_image(1000, 1000),
        parsed_data=[
            {"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}
        ],
        classes=["cat", "dog"],
        inference_id="inf",
    )
    assert len(result) == 1
    np.testing.assert_allclose(result.xyxy[0], [100, 200, 300, 400])


def test_parse_single_object_without_array():
    result = parse_muse_object_detection_response(
        image=_build_image(1000, 1000),
        parsed_data={
            "label": "dog",
            "x_min": 10,
            "y_min": 20,
            "x_max": 30,
            "y_max": 40,
        },
        classes=["cat", "dog"],
        inference_id="inf",
    )
    assert len(result) == 1
    assert result.data["class_name"][0] == "dog"


def test_extract_loose_comma_separated_objects():
    raw = (
        '{"label": "cat", "x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}, '
        '{"label": "dog", "x_min": 5, "y_min": 6, "x_max": 7, "y_max": 8}'
    )
    entries = extract_flat_object_entries(raw)
    assert [entry["label"] for entry in entries] == ["cat", "dog"]


def test_detector_run_recovers_glimmer_loose_objects():
    block = VLMAsDetectorBlockV2()
    result = block.run(
        image=_build_image(1000, 1000),
        vlm_output=(
            '{"label": "cat", "x_min": 100, "y_min": 200, "x_max": 300, "y_max": 400}, '
            '{"label": "dog", "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40}'
        ),
        classes=["cat", "dog"],
        model_type="muse",
        task_type="object-detection",
    )
    assert result["error_status"] is False
    assert len(result["predictions"]) == 2


def test_convert_box_clamps_out_of_range_coordinates():
    result = convert_muse_detection_to_pixel_xyxy(
        box=[-50.0, 0.0, 1200.0, 1000.0],
        image_height=480,
        image_width=640,
    )
    assert result == [0.0, 0.0, 640.0, 480.0]
