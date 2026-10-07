from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from inference.core.entities.responses.inference import InferenceResponse


class EmbeddingInfo(BaseModel):
    """Describe the model and representation defining an embedding space."""

    model_id: str = Field(
        description="Resolved classification model ID used to produce the vectors.",
        examples=["my-classifier/1"],
    )
    feature_definition: str = Field(
        description="Versioned definition of the feature or logit extraction boundary.",
        examples=["classifier-linear-input@v1", "classifier-linear-output@v1"],
    )
    output_type: Literal["feature_vector", "logits"] = Field(
        default="feature_vector",
        description="Whether vectors contain features before the final linear layer or logits before Softmax/Sigmoid.",
        examples=["feature_vector", "logits"],
    )
    dimension: int = Field(
        description="Number of values in each returned vector.",
        examples=[2048],
    )
    normalization: str = Field(
        default="none",
        description="Normalization applied to the returned vectors; none means raw model output.",
        examples=["none"],
    )
    preprocessing: Dict[str, Any] = Field(
        description="Model preprocessing configuration and request overrides used for these vectors.",
        examples=[{"overrides": {"disable_preproc_grayscale": False}}],
    )
    space_id: str = Field(
        description="SHA-256 identifier of the model, feature definition, dimension, normalization and preprocessing. Compare vectors only when these identifiers match.",
        examples=["a" * 64],
    )
    backend: str = Field(
        description="Model implementation or execution backend that produced the vectors.",
        examples=["ResNetForClassificationOnnx"],
    )
    precision: str = Field(
        description="Model output dtype before conversion to JSON floating-point values.",
        examples=["torch.float32"],
    )
    feature_tensor: Optional[str] = Field(
        default=None,
        description="Recovered ONNX tensor name used as the extraction boundary, when available.",
        examples=["features"],
    )
    source_artifact_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 digest of the source model artifact used for ONNX extraction, when available.",
        examples=["b" * 64],
    )
    transform_version: Optional[int] = Field(
        default=None,
        description="Version of the ONNX extraction transform, when applicable.",
        examples=[1],
    )


class ImageEmbeddingResponse(InferenceResponse):
    """Return one embedding per input image with shared compatibility metadata."""

    embeddings: List[List[float]] = Field(
        description="Feature or logit vectors, one per input image in input order.",
        examples=[[[0.25, -0.5], [0.75, 1.0]]],
    )
    embedding_info: EmbeddingInfo = Field(
        description="Model and representation metadata shared by the returned vectors.",
        examples=[
            {
                "model_id": "my-classifier/1",
                "feature_definition": "classifier-linear-input@v1",
                "output_type": "feature_vector",
                "dimension": 2,
                "normalization": "none",
                "preprocessing": {},
                "space_id": "a" * 64,
                "backend": "ResNetForClassificationOnnx",
                "precision": "torch.float32",
            }
        ],
    )
    time: Optional[float] = Field(
        default=None,
        description="Time in seconds spent preprocessing images and generating embeddings.",
        examples=[0.012],
    )
