import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
)
from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.formatters.property_definition.v1 import (
    PropertyDefinitionBlockV1,
)
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
    CLASS_NAMES_KEY,
)
from inference_models import ClassificationPrediction as NativeClassificationPrediction
from inference_models.models.base.object_detection import Detections as NativeDetections

# PropertyDefinitionBlockV1 runs UQL operation chains internally, so under
# ENABLE_TENSOR_DATA_REPRESENTATION it inherits the native-only behaviour of the
# UQL extractors: the block must be fed native `inference_models` predictions, not
# serialised dicts / sv.Detections. The dict/sv tests below skip when the flag is on;
# each has a `*_tensor_native` parity test (skipped when the flag is off) feeding the
# native equivalent. Geometric anchor properties return rounded-int lists identical
# to the numpy path; per-box scalars (area_px, area_converted) are read from
# `bboxes_metadata`.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="dict / sv.Detections input; the UQL chain inside the block is native-only "
    "under ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


@_NUMPY_ONLY
def test_property_extraction_block() -> None:
    # given
    data = ClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions=[
            ClassificationPrediction(
                **{"class": "cat", "class_id": 0, "confidence": 0.6}
            ),
            ClassificationPrediction(
                **{"class": "dog", "class_id": 1, "confidence": 0.4}
            ),
        ],
        top="cat",
        confidence=0.6,
        parent_id="some",
    ).dict(by_alias=True, exclude_none=True)
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "ClassificationPropertyExtract",
                    "property_name": "top_class",
                },
                {
                    "type": "LookupTable",
                    "lookup_table": {"cat": "cat-mutated"},
                },
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=data, operations=operations)

    # then
    assert result == {"output": "cat-mutated"}


@_NUMPY_ONLY
def test_property_extraction_block_with_center() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={"class": np.array(["car", "dog"])},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "center"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[20, 30], [40, 50]]}


@_NUMPY_ONLY
def test_property_extraction_block_with_top_left() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={"class": np.array(["car", "dog"])},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "top_left"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[10, 20], [30, 40]]}


@_NUMPY_ONLY
def test_property_extraction_block_with_top_right() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={"class": np.array(["car", "dog"])},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "top_right"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[30, 20], [50, 40]]}


@_NUMPY_ONLY
def test_property_extraction_block_with_bottom_left() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={"class": np.array(["car", "dog"])},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "bottom_left"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[10, 40], [30, 60]]}


@_NUMPY_ONLY
def test_property_extraction_block_with_bottom_right() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={"class": np.array(["car", "dog"])},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "bottom_right"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[30, 40], [50, 60]]}


@_NUMPY_ONLY
def test_property_extraction_block_with_area_px() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={AREA_KEY_IN_SV_DETECTIONS: np.array([400.0, 1000.0], dtype=np.float32)},
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "DetectionsPropertyExtract",
                    "property_name": AREA_KEY_IN_SV_DETECTIONS,
                }
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [400.0, 1000.0]}


@_NUMPY_ONLY
def test_property_extraction_block_with_area_converted() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.array([4.0, 10.0], dtype=np.float32)
        },
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "DetectionsPropertyExtract",
                    "property_name": AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
                }
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [4.0, 10.0]}


# ---------------------------------------------------------------------------
# Tensor-native parity variants (run only under ENABLE_TENSOR_DATA_REPRESENTATION).
# Same scenarios as above, but feeding the block native `inference_models`
# predictions instead of dict / sv.Detections.
# ---------------------------------------------------------------------------


@_TENSOR_ONLY
def test_property_extraction_block_tensor_native() -> None:
    # given
    data = NativeClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        images_metadata=[{CLASS_NAMES_KEY: {0: "cat", 1: "dog"}}],
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "ClassificationPropertyExtract",
                    "property_name": "top_class",
                },
                {
                    "type": "LookupTable",
                    "lookup_table": {"cat": "cat-mutated"},
                },
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=data, operations=operations)

    # then
    assert result == {"output": "cat-mutated"}


def _native_detections_for_property_extraction() -> NativeDetections:
    # mirrors the sv.Detections fixture used by the geometric parity tests
    return NativeDetections(
        xyxy=torch.tensor([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )


@_TENSOR_ONLY
def test_property_extraction_block_with_center_tensor_native() -> None:
    # given
    detections = _native_detections_for_property_extraction()
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "center"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then - anchor coords are .round().long().tolist() -> ints, like the numpy path
    assert result == {"output": [[20, 30], [40, 50]]}


@_TENSOR_ONLY
def test_property_extraction_block_with_top_left_tensor_native() -> None:
    # given
    detections = _native_detections_for_property_extraction()
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "top_left"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[10, 20], [30, 40]]}


@_TENSOR_ONLY
def test_property_extraction_block_with_top_right_tensor_native() -> None:
    # given
    detections = _native_detections_for_property_extraction()
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "top_right"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[30, 20], [50, 40]]}


@_TENSOR_ONLY
def test_property_extraction_block_with_bottom_left_tensor_native() -> None:
    # given
    detections = _native_detections_for_property_extraction()
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "bottom_left"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[10, 40], [30, 60]]}


@_TENSOR_ONLY
def test_property_extraction_block_with_bottom_right_tensor_native() -> None:
    # given
    detections = _native_detections_for_property_extraction()
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {"type": "DetectionsPropertyExtract", "property_name": "bottom_right"}
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [[30, 40], [50, 60]]}


@_TENSOR_ONLY
def test_property_extraction_block_with_area_px_tensor_native() -> None:
    # given - non-geometric per-box scalars are read from bboxes_metadata
    detections = NativeDetections(
        xyxy=torch.tensor([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
        bboxes_metadata=[
            {AREA_KEY_IN_SV_DETECTIONS: 400.0},
            {AREA_KEY_IN_SV_DETECTIONS: 1000.0},
        ],
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "DetectionsPropertyExtract",
                    "property_name": AREA_KEY_IN_SV_DETECTIONS,
                }
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [400.0, 1000.0]}


@_TENSOR_ONLY
def test_property_extraction_block_with_area_converted_tensor_native() -> None:
    # given - non-geometric per-box scalars are read from bboxes_metadata
    detections = NativeDetections(
        xyxy=torch.tensor([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=torch.float32),
        class_id=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
        bboxes_metadata=[
            {AREA_CONVERTED_KEY_IN_SV_DETECTIONS: 4.0},
            {AREA_CONVERTED_KEY_IN_SV_DETECTIONS: 10.0},
        ],
    )
    operations = OperationsChain.model_validate(
        {
            "operations": [
                {
                    "type": "DetectionsPropertyExtract",
                    "property_name": AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
                }
            ]
        }
    ).operations
    step = PropertyDefinitionBlockV1()

    # when
    result = step.run(data=detections, operations=operations)

    # then
    assert result == {"output": [4.0, 10.0]}
