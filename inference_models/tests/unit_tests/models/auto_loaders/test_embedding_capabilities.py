from unittest.mock import MagicMock

import pytest

from inference_models.errors import NoModelPackagesAvailableError
from inference_models.models.auto_loaders import auto_negotiation, core
from inference_models.models.auto_loaders.auto_negotiation import (
    negotiate_model_packages,
)
from inference_models.models.auto_loaders.capabilities import supports_capabilities
from inference_models.models.auto_loaders.core import load_model_with_capabilities
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.base.image_embeddings import ImageEmbeddingModel
from inference_models.weights_providers.entities import (
    ModelPackageMetadata,
    Quantization,
)


@pytest.mark.parametrize(
    "architecture,backend",
    [
        ("resnet", "torch"),
        ("resnet", "onnx"),
        ("vit", "onnx"),
        ("vit", "hugging-face"),
        ("dinov3_probe", "torch"),
        ("dinov3_probe", "onnx"),
    ],
)
@pytest.mark.parametrize("task", ["classification", "multi-label-classification"])
def test_supported_classifier_capabilities(architecture, backend, task):
    assert supports_capabilities(architecture, task, backend, ["image_embeddings"])
    assert not supports_capabilities(architecture, task, "trt", ["image_embeddings"])


def test_pinned_classifier_engine_is_rejected():
    package = ModelPackageMetadata(
        package_id="engine", backend=BackendType.TRT, package_artefacts=[]
    )
    with pytest.raises(NoModelPackagesAvailableError, match="same model version"):
        negotiate_model_packages(
            "resnet",
            "classification",
            [package],
            requested_model_package_id="engine",
            required_capabilities=["image_embeddings"],
        )


def test_local_incompatible_model_rejected_before_loading():
    class Classifier:
        from_pretrained = MagicMock()

    with pytest.raises(NoModelPackagesAvailableError, match="same model version"):
        load_model_with_capabilities(
            Classifier, "/local/package", required_capabilities=["image_embeddings"]
        )
    Classifier.from_pretrained.assert_not_called()


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_local_package_prepares_features_before_returning(output_type):
    class EmbeddingClassifier(ImageEmbeddingModel):
        prepare_image_embeddings = MagicMock()
        from_pretrained = MagicMock()

    model = EmbeddingClassifier()
    EmbeddingClassifier.from_pretrained.return_value = model
    assert (
        load_model_with_capabilities(
            EmbeddingClassifier,
            "/local/package",
            required_capabilities=["image_embeddings"],
            output_type=output_type,
        )
        is model
    )
    model.prepare_image_embeddings.assert_called_once_with(output_type=output_type)


def test_automatic_negotiation_excludes_classifier_engine(monkeypatch):
    packages = [
        ModelPackageMetadata(
            package_id=backend.value,
            backend=backend,
            package_artefacts=[],
            quantization=Quantization.FP32,
        )
        for backend in (BackendType.TRT, BackendType.ONNX, BackendType.TORCH)
    ]
    monkeypatch.setattr(
        auto_negotiation,
        "filter_model_packages_matching_runtime_environment",
        lambda model_packages, **kwargs: (model_packages, []),
    )
    result = negotiate_model_packages(
        "resnet",
        "classification",
        packages,
        allow_untrusted_packages=True,
        requested_quantization=Quantization.FP32,
        required_capabilities=["image_embeddings"],
    )
    assert {package.backend for package in result} == {
        BackendType.ONNX,
        BackendType.TORCH,
    }


def test_access_manager_cache_separates_classification_and_feature_versions(
    monkeypatch,
):
    class EmbeddingClassifier(ImageEmbeddingModel):
        prepare_image_embeddings = MagicMock()

    manager = MagicMock()
    manager.is_model_access_forbidden.return_value = False
    manager.retrieve_model_instance.return_value = EmbeddingClassifier()
    monkeypatch.setattr(core, "x_ray_runtime_environment", lambda: "environment")
    kwargs = {
        "model_access_manager": manager,
        "auto_resolution_cache": MagicMock(),
        "device": "cpu",
    }
    core.AutoModel.from_pretrained("project/1", **kwargs)
    core.AutoModel.from_pretrained(
        "project/1", required_capabilities=["image_embeddings"], **kwargs
    )
    monkeypatch.setattr(core, "FEATURE_DEFINITION", "classifier-linear-input@v2")
    core.AutoModel.from_pretrained(
        "project/1", required_capabilities=["image_embeddings"], **kwargs
    )
    core.AutoModel.from_pretrained(
        "project/1",
        required_capabilities=["image_embeddings"],
        output_type="logits",
        **kwargs
    )
    digests = [
        call.kwargs["loading_parameter_digest"]
        for call in manager.retrieve_model_instance.call_args_list
    ]
    assert len(set(digests)) == 4
