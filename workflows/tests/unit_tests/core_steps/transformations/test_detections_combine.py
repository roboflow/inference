from copy import deepcopy

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common.deserializers import (
    deserialize_detections_kind,
)
from roboflow_workflows.core_steps.common.serializers import serialise_sv_detections
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
    empty_detections_with_image_metadata,
)
from roboflow_workflows.core_steps.transformations.detections_combine.v1 import (
    BlockManifest,
    DetectionsCombineBlockV1,
)
from roboflow_workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


@pytest.mark.parametrize("type_alias", ["roboflow_core/detections_combine@v1"])
def test_detections_combine_validation_when_valid_manifest_is_given(
    type_alias: str,
) -> None:
    # given
    data = {
        "type": type_alias,
        "name": "detections_combine",
        "prediction_one": "$steps.od_model.predictions",
        "prediction_two": "$steps.od_model.predictions",
        "class_name": "custom_merged",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="detections_combine",
        prediction_one="$steps.od_model.predictions",
        prediction_two="$steps.od_model.predictions",
        class_name="custom_merged",
    )


def test_detections_combine_block() -> None:
    # given
    block = DetectionsCombineBlockV1()
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 1]),
        data={
            "class_name": np.array(["person", "person"]),
        },
    )

    # when
    output = block.run(prediction_one=detections, prediction_two=detections)

    # then
    assert isinstance(output, dict)
    assert "predictions" in output
    assert len(output["predictions"]) == 4
    assert np.allclose(
        output["predictions"].xyxy, np.array([[10, 10, 20, 20], [15, 15, 25, 25]] * 2)
    )
    assert np.allclose(output["predictions"].confidence, np.array([0.9, 0.8] * 2))
    assert np.allclose(output["predictions"].class_id, np.array([[1, 1] * 2]))
    assert all([c == "person" for c in output["predictions"].data["class_name"]])


def _empty_detections_for_image(
    image_height: int = 480,
    image_width: int = 640,
    parent_id: str = "root",
) -> sv.Detections:
    """Zero-row detections produced the way real producers emit them: collection
    metadata (image dimensions and parent/root lineage) on ``.metadata``."""
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id=parent_id,
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0,
                left_top_y=0,
                origin_height=image_height,
                origin_width=image_width,
            ),
        ),
        numpy_image=np.zeros((image_height, image_width, 3), dtype=np.uint8),
    )
    return empty_detections_with_image_metadata(image=image)


def _empty_detections_for_crop(
    crop_identifier: str,
    offset_x: int,
    offset_y: int,
) -> sv.Detections:
    """Zero-row detections over a same-sized crop of one shared 480x640 root, so
    lineage keys agree on the root but conflict on the crop offset."""
    root = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="root",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0,
                left_top_y=0,
                origin_height=480,
                origin_width=640,
            ),
        ),
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    crop = WorkflowImageData.create_crop(
        origin_image_data=root,
        crop_identifier=crop_identifier,
        cropped_image=np.zeros((100, 200, 3), dtype=np.uint8),
        offset_x=offset_x,
        offset_y=offset_y,
    )
    return empty_detections_with_image_metadata(image=crop)


def _populated_detections(data: dict) -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20], [15, 15, 25, 25]]),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([1, 1]),
        data=data,
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="root",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=0,
                left_top_y=0,
                origin_height=480,
                origin_width=640,
            ),
        ),
        numpy_image=np.zeros((480, 640, 3), dtype=np.uint8),
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )


def test_detections_combine_preserves_agreeing_collection_metadata_when_all_inputs_empty() -> (
    None
):
    # given
    block = DetectionsCombineBlockV1()
    prediction_one = _empty_detections_for_image()
    prediction_two = _empty_detections_for_image()
    metadata_snapshot_one = deepcopy(prediction_one.metadata)
    metadata_snapshot_two = deepcopy(prediction_two.metadata)

    # when
    output = block.run(prediction_one=prediction_one, prediction_two=prediction_two)

    # then
    result = output["predictions"]
    assert len(result) == 0
    # the whole producer-emitted collection metadata survives the combine
    assert set(result.metadata.keys()) == {
        IMAGE_DIMENSIONS_KEY,
        PARENT_ID_KEY,
        PARENT_COORDINATES_KEY,
        PARENT_DIMENSIONS_KEY,
        ROOT_PARENT_ID_KEY,
        ROOT_PARENT_COORDINATES_KEY,
        ROOT_PARENT_DIMENSIONS_KEY,
    }
    # producer value types: lists for dimensions/coordinates, scalars for ids
    assert result.metadata[IMAGE_DIMENSIONS_KEY] == [480, 640]
    assert result.metadata[PARENT_COORDINATES_KEY] == [0, 0]
    assert result.metadata[PARENT_DIMENSIONS_KEY] == [480, 640]
    assert result.metadata[ROOT_PARENT_COORDINATES_KEY] == [0, 0]
    assert result.metadata[ROOT_PARENT_DIMENSIONS_KEY] == [480, 640]
    assert result.metadata[PARENT_ID_KEY] == "root"
    assert result.metadata[ROOT_PARENT_ID_KEY] == "root"
    # inputs are not mutated and the output is a fresh detections object
    assert result is not prediction_one
    assert result is not prediction_two
    assert prediction_one.metadata == metadata_snapshot_one
    assert prediction_two.metadata == metadata_snapshot_two


def test_detections_combine_all_empty_result_round_trips_image_dimensions() -> None:
    # given
    block = DetectionsCombineBlockV1()
    prediction_one = _empty_detections_for_image()
    prediction_two = _empty_detections_for_image()

    # when
    output = block.run(prediction_one=prediction_one, prediction_two=prediction_two)
    serialized = serialise_sv_detections(output["predictions"])
    deserialized = deserialize_detections_kind(
        parameter="predictions",
        detections=serialized,
    )

    # then - the wire payload carries real dimensions instead of nulls, and the
    # deserialiser restores them on the empty result's metadata
    assert serialized["image"] == {"width": 640, "height": 480}
    assert serialized["predictions"] == []
    assert len(deserialized) == 0
    assert deserialized.metadata[IMAGE_DIMENSIONS_KEY] == [480, 640]


def test_detections_combine_rejects_conflicting_dimensions_when_all_inputs_empty() -> (
    None
):
    # given - same parent, different image dimensions
    block = DetectionsCombineBlockV1()
    prediction_one = _empty_detections_for_image(image_height=480, image_width=640)
    prediction_two = _empty_detections_for_image(image_height=720, image_width=1280)

    # when / then - conflicts reject with supervision's established error rather
    # than silently picking one side
    with pytest.raises(
        ValueError, match="Conflicting metadata for key: 'image_dimensions'"
    ):
        block.run(prediction_one=prediction_one, prediction_two=prediction_two)


def test_detections_combine_rejects_conflicting_lineage_when_all_inputs_empty() -> None:
    # given - crops of the same root at different offsets: identical
    # image_dimensions and root identity, conflicting crop offsets
    block = DetectionsCombineBlockV1()
    prediction_one = _empty_detections_for_crop(
        crop_identifier="crop-a", offset_x=64, offset_y=32
    )
    prediction_two = _empty_detections_for_crop(
        crop_identifier="crop-b", offset_x=96, offset_y=64
    )

    # when / then
    with pytest.raises(
        ValueError, match="Conflicting metadata for key: 'root_parent_coordinates'"
    ):
        block.run(prediction_one=prediction_one, prediction_two=prediction_two)


def test_detections_combine_rejects_metadata_key_mismatch_when_all_inputs_empty() -> (
    None
):
    # given - two metadata-bearing empties with different key sets
    block = DetectionsCombineBlockV1()
    prediction_one = _empty_detections_for_image()
    prediction_two = sv.Detections.empty()
    prediction_two.metadata[IMAGE_DIMENSIONS_KEY] = [480, 640]

    # when / then - asymmetric metadata key sets reject with supervision's
    # established error instead of padding or dropping fields
    with pytest.raises(
        ValueError, match="All metadata dictionaries must have the same keys"
    ):
        block.run(prediction_one=prediction_one, prediction_two=prediction_two)


@pytest.mark.parametrize("bare_first", [False, True])
def test_detections_combine_bare_empty_preserves_other_empty_metadata(
    bare_first: bool,
) -> None:
    populated_metadata = _empty_detections_for_image()
    bare = sv.Detections.empty()
    prediction_one, prediction_two = (
        (bare, populated_metadata) if bare_first else (populated_metadata, bare)
    )

    result = DetectionsCombineBlockV1().run(
        prediction_one=prediction_one, prediction_two=prediction_two
    )["predictions"]

    assert len(result) == 0
    assert result.metadata == populated_metadata.metadata
    assert serialise_sv_detections(result)["image"] == {"width": 640, "height": 480}
    assert bare.metadata == {}


def test_detections_combine_two_bare_empties_still_succeeds() -> None:
    result = DetectionsCombineBlockV1().run(
        prediction_one=sv.Detections.empty(), prediction_two=sv.Detections.empty()
    )["predictions"]

    assert len(result) == 0
    assert result.metadata == {}


def test_detections_combine_preserves_ndarray_valued_metadata_when_all_inputs_empty() -> (
    None
):
    # given - metadata values stored as numpy arrays (merged under array-equality
    # semantics, not naive ==)
    block = DetectionsCombineBlockV1()
    prediction_one = sv.Detections.empty()
    prediction_one.metadata[IMAGE_DIMENSIONS_KEY] = np.array([480, 640])
    prediction_two = sv.Detections.empty()
    prediction_two.metadata[IMAGE_DIMENSIONS_KEY] = np.array([480, 640])

    # when
    output = block.run(prediction_one=prediction_one, prediction_two=prediction_two)

    # then
    assert np.array_equal(
        output["predictions"].metadata[IMAGE_DIMENSIONS_KEY],
        np.array([480, 640]),
    )


def test_detections_combine_rejects_list_vs_ndarray_metadata_conflict_when_all_inputs_empty() -> (
    None
):
    # given - same logical dimensions carried under different value types:
    # list and ndarray are not equal under metadata_values_equal, so this is a
    # genuine producer conflict and must reject rather than coerce
    block = DetectionsCombineBlockV1()
    prediction_one = sv.Detections.empty()
    prediction_one.metadata[IMAGE_DIMENSIONS_KEY] = [480, 640]
    prediction_two = sv.Detections.empty()
    prediction_two.metadata[IMAGE_DIMENSIONS_KEY] = np.array([480, 640])

    # when / then
    with pytest.raises(
        ValueError, match="Conflicting metadata for key: 'image_dimensions'"
    ):
        block.run(prediction_one=prediction_one, prediction_two=prediction_two)


def test_detections_combine_mixed_empty_and_populated_behaviour_unchanged() -> None:
    # given - control: one populated and one metadata-rich empty input
    block = DetectionsCombineBlockV1()
    prediction_one = _populated_detections(
        data={
            "class_name": np.array(["person", "person"]),
            "detection_id": np.array(["id-1", "id-2"]),
        }
    )
    prediction_two = _empty_detections_for_image()

    # when
    output = block.run(prediction_one=prediction_one, prediction_two=prediction_two)

    # then - the populated rows merge as before; the empty input's metadata was
    # already invisible to sv.Detections.merge and stays so
    result = output["predictions"]
    assert len(result) == 2
    assert np.allclose(result.xyxy, np.array([[10, 10, 20, 20], [15, 15, 25, 25]]))
    assert result.data[PARENT_ID_KEY].tolist() == ["root", "root"]
    assert result.metadata == {}


def test_detections_combine_heterogeneous_populated_fields_still_rejected() -> None:
    # given - control: populated inputs whose per-row data key sets differ
    # (e.g. one producer stamps inference_id, the other does not)
    block = DetectionsCombineBlockV1()
    prediction_one = _populated_detections(
        data={
            "class_name": np.array(["person", "person"]),
            "detection_id": np.array(["id-1", "id-2"]),
            "inference_id": np.array(["inf-1", "inf-1"]),
        }
    )
    prediction_two = _populated_detections(
        data={
            "class_name": np.array(["person", "person"]),
            "detection_id": np.array(["id-3", "id-4"]),
        }
    )

    # when / then - the strict data-key behaviour for non-empty inputs is
    # unchanged: still delegated to sv.Detections.merge
    with pytest.raises(
        ValueError, match="All data dictionaries must have the same keys"
    ):
        block.run(prediction_one=prediction_one, prediction_two=prediction_two)
