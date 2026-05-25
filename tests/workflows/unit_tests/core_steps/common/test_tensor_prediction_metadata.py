import numpy as np
import pytest
import torch

from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections as TensorDetections

from inference.core.workflows.core_steps.common.tensor_prediction_metadata import (
    CLASS_NAMES_KEY,
    MODEL_ID_KEY,
    attach_prediction_metadata,
)
from inference.core.workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)


def _make_image(
    *,
    parent_id: str = "parent-1",
    root_parent_id: str = "root-1",
    parent_left_top_x: int = 0,
    parent_left_top_y: int = 0,
    parent_width: int = 100,
    parent_height: int = 80,
    root_left_top_x: int = 10,
    root_left_top_y: int = 20,
    root_width: int = 640,
    root_height: int = 480,
    image_h: int = 80,
    image_w: int = 100,
) -> WorkflowImageData:
    numpy_image = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    parent_metadata = ImageParentMetadata(
        parent_id=parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=parent_left_top_x,
            left_top_y=parent_left_top_y,
            origin_width=parent_width,
            origin_height=parent_height,
        ),
    )
    root_metadata = ImageParentMetadata(
        parent_id=root_parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_x=root_left_top_x,
            left_top_y=root_left_top_y,
            origin_width=root_width,
            origin_height=root_height,
        ),
    )
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        workflow_root_ancestor_metadata=root_metadata,
        numpy_image=numpy_image,
    )


def _make_detections(n: int = 2, image_metadata: dict | None = None) -> TensorDetections:
    return TensorDetections(
        xyxy=torch.zeros((n, 4), dtype=torch.float32),
        class_id=torch.zeros((n,), dtype=torch.int64),
        confidence=torch.zeros((n,), dtype=torch.float32),
        image_metadata=image_metadata,
    )


def test_attach_metadata_populates_all_expected_keys_for_detections() -> None:
    # given
    image = _make_image(
        parent_id="p1",
        root_parent_id="r1",
        parent_left_top_x=3,
        parent_left_top_y=4,
        parent_width=200,
        parent_height=120,
        root_left_top_x=11,
        root_left_top_y=22,
        root_width=1280,
        root_height=720,
        image_h=120,
        image_w=200,
    )
    prediction = _make_detections(n=3)

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="some-model/1",
        prediction_type="object-detection",
        class_names=["cat", "dog"],
    )

    # then
    metadata = prediction.image_metadata
    assert metadata is not None
    assert metadata[INFERENCE_ID_KEY] == inference_id
    assert metadata[MODEL_ID_KEY] == "some-model/1"
    assert metadata[PREDICTION_TYPE_KEY] == "object-detection"
    assert metadata[CLASS_NAMES_KEY] == ["cat", "dog"]
    assert metadata[IMAGE_DIMENSIONS_KEY] == (120, 200)
    assert metadata[PARENT_ID_KEY] == "p1"
    assert metadata[PARENT_DIMENSIONS_KEY] == (120, 200)
    assert metadata[PARENT_COORDINATES_KEY] == (3, 4)
    assert metadata[ROOT_PARENT_ID_KEY] == "r1"
    assert metadata[ROOT_PARENT_DIMENSIONS_KEY] == (720, 1280)
    assert metadata[ROOT_PARENT_COORDINATES_KEY] == (11, 22)


def test_attach_metadata_returns_minted_uuid_when_no_existing_id() -> None:
    # given
    image = _make_image()
    prediction = _make_detections()

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=[],
    )

    # then
    assert isinstance(inference_id, str)
    assert len(inference_id) > 0
    assert prediction.image_metadata[INFERENCE_ID_KEY] == inference_id


def test_attach_metadata_preserves_inference_id_from_existing_metadata() -> None:
    # given
    image = _make_image()
    prediction = _make_detections(image_metadata={INFERENCE_ID_KEY: "preset-id"})

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=["a"],
    )

    # then
    assert inference_id == "preset-id"
    assert prediction.image_metadata[INFERENCE_ID_KEY] == "preset-id"


def test_attach_metadata_uses_explicit_argument_when_metadata_lacks_id() -> None:
    # given
    image = _make_image()
    prediction = _make_detections()

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=[],
        inference_id="explicit-id",
    )

    # then
    assert inference_id == "explicit-id"


def test_attach_metadata_existing_metadata_id_wins_over_explicit_argument() -> None:
    # given
    image = _make_image()
    prediction = _make_detections(image_metadata={INFERENCE_ID_KEY: "from-metadata"})

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=[],
        inference_id="from-argument",
    )

    # then
    assert inference_id == "from-metadata"


def test_attach_metadata_preserves_unrelated_keys_in_existing_metadata() -> None:
    # given
    image = _make_image()
    prediction = _make_detections(
        image_metadata={"custom_upstream_key": "should-survive"}
    )

    # when
    attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=[],
    )

    # then
    assert prediction.image_metadata["custom_upstream_key"] == "should-survive"


def test_attach_metadata_omits_class_names_key_when_class_names_is_none() -> None:
    # given
    image = _make_image()
    prediction = _make_detections()

    # when
    attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=None,
    )

    # then
    assert CLASS_NAMES_KEY not in prediction.image_metadata


def test_attach_metadata_writes_empty_class_names_list_when_explicit_empty() -> None:
    # given
    image = _make_image()
    prediction = _make_detections()

    # when
    attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=[],
    )

    # then
    assert prediction.image_metadata[CLASS_NAMES_KEY] == []


def test_attach_metadata_handles_empty_detections() -> None:
    # given
    image = _make_image()
    prediction = TensorDetections(
        xyxy=torch.zeros((0, 4), dtype=torch.float32),
        class_id=torch.zeros((0,), dtype=torch.int64),
        confidence=torch.zeros((0,), dtype=torch.float32),
    )

    # when
    inference_id = attach_prediction_metadata(
        prediction,
        image=image,
        model_id="m/1",
        prediction_type="object-detection",
        class_names=["only-class"],
    )

    # then
    assert isinstance(inference_id, str)
    assert prediction.image_metadata[PREDICTION_TYPE_KEY] == "object-detection"
    assert prediction.image_metadata[CLASS_NAMES_KEY] == ["only-class"]


def test_attach_metadata_raises_for_classification_prediction() -> None:
    # given
    image = _make_image()
    prediction = ClassificationPrediction(
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.zeros((1,), dtype=torch.float32),
    )

    # when
    with pytest.raises(TypeError):
        attach_prediction_metadata(
            prediction,
            image=image,
            model_id="m/1",
            prediction_type="classification",
            class_names=["a"],
        )
