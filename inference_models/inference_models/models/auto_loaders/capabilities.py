"""Operation requirements are separate from exported package model features."""

from inference_models.errors import InvalidParameterError, NoModelPackagesAvailableError
from inference_models.models.base.image_embeddings import ImageEmbeddingModel

IMAGE_EMBEDDINGS = "image_embeddings"
SUPPORTED_IMAGE_EMBEDDING_BACKENDS = {
    "resnet": {"onnx", "torch"},
    "vit": {"onnx", "hugging-face"},
    "dinov3_probe": {"onnx", "torch"},
}


def validate_capabilities(required_capabilities):
    unknown = set(required_capabilities or ()) - {IMAGE_EMBEDDINGS}
    if unknown:
        raise InvalidParameterError(f"Unknown model capabilities: {sorted(unknown)}")


def supports_capabilities(architecture, task, backend, required_capabilities):
    validate_capabilities(required_capabilities)
    if IMAGE_EMBEDDINGS not in (required_capabilities or ()):
        return True
    return task in {"classification", "multi-label-classification"} and backend in (
        SUPPORTED_IMAGE_EMBEDDING_BACKENDS.get(architecture, set())
    )


def prepare_required_capabilities(
    model, required_capabilities, output_type="feature_vector"
):
    validate_capabilities(required_capabilities)
    if IMAGE_EMBEDDINGS in (required_capabilities or ()):
        if not isinstance(model, ImageEmbeddingModel):
            raise NoModelPackagesAvailableError(
                "This package cannot produce image embeddings. Select a ResNet, ViT "
                "or DINOv3 classifier with an ONNX, PyTorch or Hugging Face package. "
                "A classification-only TensorRT engine needs an alternative package "
                "for the same model version."
            )
        model.prepare_image_embeddings(output_type=output_type)
    return model
