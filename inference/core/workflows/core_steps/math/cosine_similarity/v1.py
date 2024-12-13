from typing import List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.utils.postprocess import cosine_similarity
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    EMBEDDING_KIND,
    FLOAT_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Calculate the cosine similarity between two embeddings.

A cosine similarity of 1 means the two embeddings are identical,
while a cosine similarity of 0 means the two embeddings are orthogonal.
Greater values indicate greater similarity.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Cosine Similarity",
            "version": "v1",
            "short_description": "Calculate the cosine similarity between two embeddings.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "math",
            "ui_manifest": {
                "section": "advanced",
                "icon": "far fa-calculator-simple",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/cosine_similarity@v1"]
    name: str = Field(description="Unique name of step in workflows")
    embedding_1: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding 1",
        examples=["$steps.clip_image.embedding"],
    )
    embedding_2: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding 2",
        examples=["$steps.clip_text.embedding"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="similarity", kind=[FLOAT_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class CosineSimilarityBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, embedding_1: List[float], embedding_2: List[float]) -> BlockResult:
        if len(embedding_1) != len(embedding_2):
            raise RuntimeError(
                f"roboflow_core/cosine_similarity@v1 block feed with different shape of embeddings. "
                f"`embedding_1`: (N, {len(embedding_1)}), `embedding_2`: (N, {len(embedding_2)})"
            )
        similarity = cosine_similarity(embedding_1, embedding_2)
        return {"similarity": similarity}
