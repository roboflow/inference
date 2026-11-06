import datetime

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.analytics.path_deviation.v2 import (
    PathDeviationAnalyticsBlockV2,
)
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)


def test_path_deviation_exact_path():
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    frames = [0, 1, 2, 3]
    detections_list = []
    images_list = []

    for i in frames:
        # Create detection at reference_path[i]
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [
                        reference_path[i][0] - 1,
                        reference_path[i][1] - 1,
                        reference_path[i][0] + 1,
                        reference_path[i][1] + 1,
                    ]
                ]
            ),
            tracker_id=np.array([1]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
            video_metadata=metadata,
        )
        images_list.append(image)

    # Initialize the block
    path_deviation_block = PathDeviationAnalyticsBlockV2()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            image=images_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    assert len(frechet_distances) == len(frames)
    # Optional: Check that frechet distances decrease over time
    assert all(
        frechet_distances[i] >= frechet_distances[i + 1]
        for i in range(len(frechet_distances) - 1)
    )

    # Since the object follows exactly the reference path,
    # the frechet distance should be zero at the final frame.
    assert frechet_distances[-1] == pytest.approx(0.0, abs=1e-6)


def test_path_deviation_with_deviation():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    object_path = [(0, 0), (1, 1), (2, 2), (4, 4)]  # Deviates at the last point
    frames = [0, 1, 2, 3]
    detections_list = []
    images_list = []

    for i in frames:
        # Create detection at object_path[i]
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [
                        object_path[i][0] - 1,
                        object_path[i][1] - 1,
                        object_path[i][0] + 1,
                        object_path[i][1] + 1,
                    ]
                ]
            ),
            tracker_id=np.array([1]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
            video_metadata=metadata,
        )
        images_list.append(image)

    # Initialize the block
    path_deviation_block = PathDeviationAnalyticsBlockV2()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            image=images_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    # Since the object deviates from the reference path, the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_path_deviation_multiple_objects():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    frames = [0, 1, 2, 3]
    detections_list = []
    images_list = []

    for i in frames:
        # Create detections for two objects
        detections = sv.Detections(
            xyxy=np.array(
                [
                    [i, i, i + 1, i + 1],  # Object 1 following the path
                    [i + 1, i, i + 2, i + 1],  # Object 2 deviating
                ]
            ),
            tracker_id=np.array([1, 2]),
        )
        detections_list.append(detections)

        # Create metadata
        metadata = VideoMetadata(
            video_identifier="vid_1",
            frame_number=i,
            fps=1,  # Assume 1 fps
            frame_timestamp=datetime.datetime.fromtimestamp(1726570875 + i).astimezone(
                tz=datetime.timezone.utc
            ),
            comes_from_video_file=True,
        )
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
            video_metadata=metadata,
        )
        images_list.append(image)

    # Initialize the block
    path_deviation_block = PathDeviationAnalyticsBlockV2()

    # Run the block for each frame and collect frechet_distance
    frechet_distances = []

    for i in range(len(frames)):
        result = path_deviation_block.run(
            detections=detections_list[i],
            image=images_list[i],
            triggering_anchor=sv.Position.CENTER,  # Default
            reference_path=reference_path,
        )
        frechet_distance = result["path_deviation_detections"]["path_deviation"][0]
        frechet_distances.append(frechet_distance)

    # Then
    # Since one object follows the path exactly, and one deviates,
    # the frechet_distance should be determined by the object with the maximum distance
    # In this case, object 2 deviates, so the frechet distance should be greater than zero.
    assert frechet_distances[-1] > 0.0


def test_path_deviation_no_tracker_id():
    # Given
    reference_path = [(0, 0), (1, 1), (2, 2), (3, 3)]
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        tracker_id=None,  # No tracker_id
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=0,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    path_deviation_block = PathDeviationAnalyticsBlockV2()

    # When / Then
    with pytest.raises(
        ValueError,
        match="tracker_id not initialized, PathDeviationAnalyticsBlockV2 requires detections to be tracked",
    ):
        _ = path_deviation_block.run(
            detections=detections,
            image=image,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )


def test_path_deviation_invalid_reference_path():
    # Given
    reference_path = "invalid_reference_path"  # Not a list of points
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]]),
        tracker_id=np.array([1]),
    )
    metadata = VideoMetadata(
        video_identifier="vid_1",
        frame_number=0,
        fps=1,
        frame_timestamp=datetime.datetime.fromtimestamp(1726570875).astimezone(
            tz=datetime.timezone.utc
        ),
        comes_from_video_file=True,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=metadata,
    )
    path_deviation_block = PathDeviationAnalyticsBlockV2()

    # When / Then
    with pytest.raises(TypeError):
        _ = path_deviation_block.run(
            detections=detections,
            image=image,
            triggering_anchor=sv.Position.CENTER,
            reference_path=reference_path,
        )


def _video_frame(frame_number: int = 0) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
        video_metadata=VideoMetadata(
            video_identifier="vid_1",
            frame_number=frame_number,
            fps=1,
            frame_timestamp=datetime.datetime.fromtimestamp(
                1726570875 + frame_number
            ).astimezone(tz=datetime.timezone.utc),
            comes_from_video_file=True,
        ),
    )


# CENTER_OF_MASS is left out because it needs masks, which these detections lack.
@pytest.mark.parametrize(
    "anchor", [a for a in sv.Position.list() if a != "CENTER_OF_MASS"]
)
def test_path_deviation_accepts_every_anchor_the_manifest_allows(anchor):
    # Given
    def run(triggering_anchor):
        return PathDeviationAnalyticsBlockV2().run(
            detections=sv.Detections(
                xyxy=np.array([[10.0, 20.0, 30.0, 60.0]]),
                tracker_id=np.array([1]),
            ),
            image=_video_frame(),
            triggering_anchor=triggering_anchor,
            reference_path=[(0, 0), (100, 100)],
        )["path_deviation_detections"]["path_deviation"][0]

    # When / Then
    assert run(anchor) == run(sv.Position(anchor))


def test_path_deviation_accepts_string_anchor_on_empty_detections():
    # When
    result = PathDeviationAnalyticsBlockV2().run(
        detections=sv.Detections(
            xyxy=np.zeros((0, 4)),
            tracker_id=np.array([], dtype=int),
        ),
        image=_video_frame(),
        triggering_anchor="CENTER",
        reference_path=[(0, 0), (100, 100)],
    )

    # Then
    assert len(result["path_deviation_detections"]) == 0


@pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="passes sv.Detections, which the tensor engine rejects; "
    "test_path_deviation_tensor.py covers that path",
)
def test_path_deviation_runs_in_workflow_with_default_anchor():
    # Given
    workflow_definition = {
        "version": "1.3.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/path_deviation_analytics@v2",
                "name": "path_deviation",
                "image": "$inputs.image",
                "detections": "$inputs.predictions",
                "reference_path": [[0, 0], [100, 100]],
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "result",
                "selector": "$steps.path_deviation.path_deviation_detections",
            }
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters={},
        max_concurrent_steps=1,
        prevent_local_images_loading=True,
    )
    predictions = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 60.0]]),
        tracker_id=np.array([1]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={"class_name": np.array(["a"]), "detection_id": np.array(["x"])},
    )

    # When
    result = engine.run(
        runtime_parameters={"image": [_video_frame()], "predictions": [predictions]}
    )

    # Then
    assert result[0]["result"]["path_deviation"][0] == pytest.approx(
        PathDeviationAnalyticsBlockV2().run(
            detections=predictions,
            image=_video_frame(),
            triggering_anchor=sv.Position.CENTER,
            reference_path=[(0, 0), (100, 100)],
        )["path_deviation_detections"]["path_deviation"][0]
    )
