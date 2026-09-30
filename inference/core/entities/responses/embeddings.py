from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from inference.core.entities.responses.inference import InferenceResponse


class EmbeddingInfo(BaseModel):
    model_id: str
    feature_definition: str
    output_type: Literal["feature_vector", "logits"] = "feature_vector"
    dimension: int
    normalization: str = "none"
    preprocessing: Dict[str, Any]
    space_id: str
    backend: str
    precision: str
    feature_tensor: Optional[str] = None
    source_artifact_sha256: Optional[str] = None
    transform_version: Optional[int] = None


class ImageEmbeddingResponse(InferenceResponse):
    embeddings: List[List[float]]
    embedding_info: EmbeddingInfo
    time: Optional[float] = None
