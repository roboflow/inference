import pytest
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE,
    INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import resolve_model_class
from inference_models.models.convnext.convnext_classification_torch import (
    ConvNeXtForMultiLabelClassificationTorch,
)
from inference_models.models.resnet.resnet_classification_torch import (
    ResNetForMultiLabelClassificationTorch,
)


@pytest.mark.parametrize(
    "task,backend,expected_class_name",
    [
        ("classification", BackendType.ONNX, "ConvNeXtForClassificationOnnx"),
        (
            "multi-label-classification",
            BackendType.ONNX,
            "ConvNeXtForMultiLabelClassificationOnnx",
        ),
        ("classification", BackendType.TORCH, "ConvNeXtForClassificationTorch"),
        (
            "multi-label-classification",
            BackendType.TORCH,
            "ConvNeXtForMultiLabelClassificationTorch",
        ),
    ],
)
def test_convnext_resolves_to_convnext_classes(task, backend, expected_class_name):
    # when
    model_class = resolve_model_class(
        model_architecture="convnext", task_type=task, backend=backend
    )

    # then
    assert model_class.__name__ == expected_class_name


@pytest.mark.parametrize("backend", [BackendType.ONNX, BackendType.TORCH])
def test_multi_label_default_confidence_comes_from_architecture_setting(backend):
    # when
    convnext_class = resolve_model_class(
        model_architecture="convnext",
        task_type="multi-label-classification",
        backend=backend,
    )
    resnet_class = resolve_model_class(
        model_architecture="resnet",
        task_type="multi-label-classification",
        backend=backend,
    )

    # then
    assert convnext_class._default_confidence == (
        INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE
    )
    assert resnet_class._default_confidence == (
        INFERENCE_MODELS_RESNET_DEFAULT_CONFIDENCE
    )


def test_convnext_and_resnet_multi_label_default_confidences_are_independent(
    monkeypatch,
):
    # given
    monkeypatch.setattr(
        ResNetForMultiLabelClassificationTorch, "_default_confidence", 0.3
    )
    monkeypatch.setattr(
        ConvNeXtForMultiLabelClassificationTorch, "_default_confidence", 0.7
    )
    scores = torch.tensor([[0.2, 0.5, 0.8]])
    resnet = ResNetForMultiLabelClassificationTorch(
        model=None,
        inference_config=None,
        class_names=["a", "b", "c"],
        device=torch.device("cpu"),
    )
    convnext = ConvNeXtForMultiLabelClassificationTorch(
        model=None,
        inference_config=None,
        class_names=["a", "b", "c"],
        device=torch.device("cpu"),
    )

    # when
    resnet_results = resnet.post_process(scores)
    convnext_results = convnext.post_process(scores)

    # then
    assert resnet_results[0].class_ids.tolist() == [1, 2]
    assert convnext_results[0].class_ids.tolist() == [2]
