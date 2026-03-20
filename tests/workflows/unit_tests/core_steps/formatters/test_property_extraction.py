import numpy as np
import supervision as sv

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
)
from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    OperationsChain,
)
from inference.core.workflows.core_steps.formatters.property_definition.v1 import (
    PropertyDefinitionBlockV1,
)
from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
)


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


def test_property_extraction_block_with_area_converted() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [30, 40, 50, 60]], dtype=np.int32),
        class_id=np.array([0, 1], dtype=np.int32),
        confidence=np.array([0.6, 0.4], dtype=np.float32),
        data={
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.array(
                [4.0, 10.0], dtype=np.float32
            )
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
