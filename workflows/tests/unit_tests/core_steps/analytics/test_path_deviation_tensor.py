import datetime

import numpy as np
import pytest
import supervision as sv

torch = pytest.importorskip("torch")

from roboflow_workflows.core_steps.analytics.path_deviation import (  # noqa: E402
    v1_tensor,
    v2_tensor,
)
from roboflow_workflows.environment import (  # noqa: E402
    ENABLE_TENSOR_DATA_REPRESENTATION,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine  # noqa: E402
from roboflow_workflows.execution_engine.entities.base import (  # noqa: E402
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

from inference_models.models.base.object_detection import Detections  # noqa: E402

REFERENCE_PATH = [(0, 0), (100, 100)]
XYXY = np.array([[10.0, 20.0, 30.0, 60.0]])


def _native_detections(xyxy: np.ndarray) -> Detections:
    n = len(xyxy)
    return Detections(
        xyxy=torch.as_tensor(xyxy, dtype=torch.float32).reshape(n, 4),
        class_id=torch.zeros(n, dtype=torch.int64),
        confidence=torch.ones(n, dtype=torch.float32),
        bboxes_metadata=[{"tracker_id": i + 1} for i in range(n)],
    )


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_identifier="vid_1",
        frame_number=0,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )


def _run_v1(detections: Detections, triggering_anchor) -> Detections:
    return v1_tensor.PathDeviationAnalyticsBlockV1().run(
        detections=detections,
        metadata=_metadata(),
        triggering_anchor=triggering_anchor,
        reference_path=REFERENCE_PATH,
    )["path_deviation_detections"]


def _run_v2(detections: Detections, triggering_anchor) -> Detections:
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=_metadata(),
    )
    return v2_tensor.PathDeviationAnalyticsBlockV2().run(
        detections=detections,
        image=image,
        triggering_anchor=triggering_anchor,
        reference_path=REFERENCE_PATH,
    )["path_deviation_detections"]


@pytest.mark.parametrize("run_block", [_run_v1, _run_v2])
# CENTER_OF_MASS is left out because it needs masks, which these detections lack.
@pytest.mark.parametrize(
    "anchor", [a for a in sv.Position.list() if a != "CENTER_OF_MASS"]
)
def test_path_deviation_tensor_accepts_every_anchor_the_manifest_allows(
    run_block, anchor: str
) -> None:
    # when
    from_string = run_block(_native_detections(XYXY), anchor)
    from_enum = run_block(_native_detections(XYXY), sv.Position(anchor))

    # then
    assert (
        from_string.bboxes_metadata[0]["path_deviation"]
        == from_enum.bboxes_metadata[0]["path_deviation"]
    )


@pytest.mark.parametrize("run_block", [_run_v1, _run_v2])
def test_path_deviation_tensor_accepts_string_anchor_on_empty_detections(
    run_block,
) -> None:
    # when
    result = run_block(_native_detections(np.zeros((0, 4))), "CENTER")

    # then
    assert result.xyxy.shape[0] == 0


@pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)
@pytest.mark.parametrize(
    "version, run_block", [("v1", _run_v1), ("v2", _run_v2)], ids=["v1", "v2"]
)
def test_path_deviation_tensor_runs_in_workflow_with_default_anchor(
    version: str, run_block
) -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=_metadata(),
    )
    inputs = [
        {
            "type": "WorkflowBatchInput",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ]
    step = {
        "type": f"roboflow_core/path_deviation_analytics@{version}",
        "name": "path_deviation",
        "detections": "$inputs.predictions",
        "reference_path": [[0, 0], [100, 100]],
    }
    runtime_parameters = {"predictions": [_native_detections(XYXY)]}
    if version == "v1":
        inputs.append({"type": "WorkflowVideoMetadata", "name": "video_metadata"})
        step["metadata"] = "$inputs.video_metadata"
        runtime_parameters["video_metadata"] = [_metadata()]
    else:
        inputs.append({"type": "WorkflowImage", "name": "image"})
        step["image"] = "$inputs.image"
        runtime_parameters["image"] = [image]
    engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.3.0",
            "inputs": inputs,
            "steps": [step],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "result",
                    "selector": "$steps.path_deviation.path_deviation_detections",
                }
            ],
        },
        init_parameters={},
        max_concurrent_steps=1,
        prevent_local_images_loading=True,
    )

    # when
    result = engine.run(runtime_parameters=runtime_parameters)

    # then
    expected = run_block(_native_detections(XYXY), sv.Position.CENTER)
    assert (
        result[0]["result"].bboxes_metadata[0]["path_deviation"]
        == expected.bboxes_metadata[0]["path_deviation"]
    )
