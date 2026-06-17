import pytest
import torch

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InferenceResponseImage,
    MultiLabelClassificationInferenceResponse,
    MultiLabelClassificationPrediction,
)
from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference_models import (
    ClassificationPrediction as NativeClassificationPrediction,
    MultiLabelClassificationPrediction as NativeMultiLabelClassificationPrediction,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the UQL classification extractors are
# native-only: they reject the serialised dict form. The dict-based tests below are
# skipped when the flag is on; each has a `*_tensor_native` parity test (skipped when
# the flag is off) exercising the same scenario with a native `inference_models`
# prediction. Multi-class class names live in `images_metadata[i][CLASS_NAMES_KEY]`;
# multi-label in `image_metadata[CLASS_NAMES_KEY]`.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="dict input; UQL classification extractors are native-only under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def test_classification_result_extraction_when_data_is_empty() -> None:
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value=None, operations=operations)


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
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

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == "cat"


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_no_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == []


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_classes_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog"]


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
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

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert abs(result - 0.6) < 1e-5


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_no_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == []


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.4]


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_class_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["cat", "dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == 0.6


@_NUMPY_ONLY
def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_no_classes_detected() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=[],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == 0.0


@_NUMPY_ONLY
def test_classification_result_extraction_of_all_classes_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_classes",
        }
    ]
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

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog"]


@_NUMPY_ONLY
def test_classification_result_extraction_of_all_classes_for_multi_label_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_classes",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
            "animal": MultiLabelClassificationPrediction(class_id=3, confidence=0.0),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == ["cat", "dog", "animal"]


@_NUMPY_ONLY
def test_classification_result_extraction_of_all_confidences_for_multi_class_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_confidences",
        }
    ]
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

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.6, 0.4]


@_NUMPY_ONLY
def test_classification_result_extraction_of_all_confidences_for_multi_label_classification_result() -> (
    None
):
    # given
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "all_confidences",
        }
    ]
    data = MultiLabelClassificationInferenceResponse(
        image=InferenceResponseImage(width=128, height=256),
        predictions={
            "cat": MultiLabelClassificationPrediction(class_id=0, confidence=0.6),
            "animal": MultiLabelClassificationPrediction(class_id=3, confidence=0.0),
            "dog": MultiLabelClassificationPrediction(class_id=1, confidence=0.4),
        },
        predicted_classes=["dog"],
    ).dict(by_alias=True, exclude_none=True)

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert result == [0.6, 0.4, 0.0]


# ---------------------------------------------------------------------------
# Tensor-native parity variants (run only under ENABLE_TENSOR_DATA_REPRESENTATION).
# Same scenarios as the dict-based tests above, but with native `inference_models`
# predictions. Multi-class: class_id (bs,), confidence (bs, num_classes), class names
# in images_metadata[i][CLASS_NAMES_KEY]. Multi-label: class_ids (selected,),
# confidence (num_classes,), class names in image_metadata[CLASS_NAMES_KEY].
# ---------------------------------------------------------------------------


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_for_multi_class_classification_result_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "ClassificationPropertyExtract", "property_name": "top_class"}]
    prediction = NativeClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        images_metadata=[{CLASS_NAMES_KEY: {0: "cat", 1: "dog"}}],
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == "cat"


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_classes_detected_tensor_native() -> (
    None
):
    # given
    operations = [{"type": "ClassificationPropertyExtract", "property_name": "top_class"}]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "cat", 1: "dog"}},
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == ["cat", "dog"]


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_for_multi_label_classification_result_when_no_class_detected_tensor_native() -> (
    None
):
    # given - no labels above threshold -> empty class_ids
    operations = [{"type": "ClassificationPropertyExtract", "property_name": "top_class"}]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {0: "cat", 1: "dog"}},
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == []


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_class_classification_result_tensor_native() -> (
    None
):
    # given - confidence is the full (bs, num_classes) softmax; top class is id 0
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
    ]
    prediction = NativeClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == pytest.approx(0.6)


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_no_class_detected_tensor_native() -> (
    None
):
    # given
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == []


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_confidence_for_multi_label_classification_result_when_class_detected_tensor_native() -> (
    None
):
    # given - only label id 1 is above threshold
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "top_class_confidence"}
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == pytest.approx([0.4])


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_class_detected_tensor_native() -> (
    None
):
    # given - both labels above threshold; single -> max selected confidence
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([0, 1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == pytest.approx(0.6)


@_TENSOR_ONLY
def test_classification_result_extraction_of_top_class_confidence_single_for_multi_label_classification_result_when_no_classes_detected_tensor_native() -> (
    None
):
    # given - nothing above threshold; single -> 0.0
    operations = [
        {
            "type": "ClassificationPropertyExtract",
            "property_name": "top_class_confidence_single",
        }
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == 0.0


@_TENSOR_ONLY
def test_classification_result_extraction_of_all_classes_for_multi_class_classification_result_tensor_native() -> (
    None
):
    # given - all_classes returns the full class map, ordered by class id
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "all_classes"}
    ]
    # class map deliberately NOT in id order, to exercise sorted-by-id output
    prediction = NativeClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
        images_metadata=[{CLASS_NAMES_KEY: {1: "dog", 0: "cat"}}],
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == ["cat", "dog"]


@_TENSOR_ONLY
def test_classification_result_extraction_of_all_classes_for_multi_label_classification_result_tensor_native() -> (
    None
):
    # given - all_classes returns the full class map ordered by id (0, 1, 3),
    # regardless of which were detected; map insertion order is deliberately
    # scrambled so the sorted-by-id output is actually exercised. Class id 3 is
    # sparse, so the (num_classes,) confidence vector has 4 slots (id 2 unnamed).
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "all_classes"}
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4, 0.0, 0.0], dtype=torch.float32),
        image_metadata={CLASS_NAMES_KEY: {3: "animal", 0: "cat", 1: "dog"}},
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == ["cat", "dog", "animal"]


@_TENSOR_ONLY
def test_classification_result_extraction_of_all_confidences_for_multi_class_classification_result_tensor_native() -> (
    None
):
    # given - all_confidences returns the full softmax row
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "all_confidences"}
    ]
    prediction = NativeClassificationPrediction(
        class_id=torch.tensor([0], dtype=torch.long),
        confidence=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == pytest.approx([0.6, 0.4])


@_TENSOR_ONLY
def test_classification_result_extraction_of_all_confidences_for_multi_label_classification_result_tensor_native() -> (
    None
):
    # given - native returns the full sigmoid vector wholesale; the dict counterpart
    # used sparse ids {0, 1, 3} for the same three confidences, mirrored here as a
    # contiguous (num_classes,) tensor.
    operations = [
        {"type": "ClassificationPropertyExtract", "property_name": "all_confidences"}
    ]
    prediction = NativeMultiLabelClassificationPrediction(
        class_ids=torch.tensor([1], dtype=torch.long),
        confidence=torch.tensor([0.6, 0.4, 0.0], dtype=torch.float32),
    )

    # when
    result = execute_operations(value=prediction, operations=operations)

    # then
    assert result == pytest.approx([0.6, 0.4, 0.0])
