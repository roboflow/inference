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
Calculate the cosine similarity between two embedding vectors by computing the cosine of the angle between them, measuring directional similarity regardless of magnitude to enable similarity comparison, semantic matching, embedding-based search, and similarity-based filtering workflows.

## How This Block Works

This block computes cosine similarity, a measure of similarity between two vectors based on the cosine of the angle between them. The block:

1. Receives two embedding vectors from workflow steps (e.g., from CLIP, Perception Encoder, or other embedding models)
2. Validates embedding dimensions:
   - Ensures both embeddings have the same dimensionality (same number of elements)
   - Raises an error if dimensions don't match
3. Computes cosine similarity:
   - Calculates the dot product of the two embedding vectors
   - Computes the L2 norm (magnitude) of each embedding vector
   - Divides the dot product by the product of the two norms: similarity = (a · b) / (||a|| × ||b||)
   - This measures the cosine of the angle between the vectors, indicating directional similarity
4. Returns similarity score:
   - Outputs a similarity value ranging from -1 to 1
   - Value of 1: Vectors point in the same direction (identical or proportional) - maximum similarity
   - Value of 0: Vectors are orthogonal (perpendicular) - no similarity
   - Value of -1: Vectors point in opposite directions - maximum dissimilarity
   - Greater values (closer to 1) indicate greater similarity

Cosine similarity is magnitude-invariant, meaning it measures similarity in direction rather than size. Two vectors that point in the same direction will have high cosine similarity even if they have different magnitudes. This makes it ideal for comparing embeddings where magnitude may vary but semantic meaning (direction) is what matters.

## Common Use Cases

- **Semantic Similarity Comparison**: Compare semantic similarity between images, text, or other data types using embeddings (e.g., compare image embeddings, match text to images, find similar content), enabling similarity comparison workflows
- **Embedding-Based Search**: Use similarity scores for embedding-based search and retrieval (e.g., find similar images, search by embedding similarity, retrieve similar content), enabling embedding search workflows
- **Cross-Modal Matching**: Match embeddings across different modalities (e.g., match images to text, find images matching text descriptions, match text to images), enabling cross-modal matching workflows
- **Similarity-Based Filtering**: Filter data based on similarity thresholds (e.g., filter similar items, find duplicates using similarity, identify near-duplicates), enabling similarity filtering workflows
- **Content Recommendation**: Use similarity scores for content recommendation and matching (e.g., recommend similar content, match related items, suggest similar products), enabling recommendation workflows
- **Quality Control and Validation**: Validate embeddings or compare embeddings for quality control (e.g., validate embedding quality, compare embeddings for consistency, check embedding similarity), enabling quality control workflows

## Connecting to Other Blocks

This block receives embeddings from embedding model blocks and produces similarity scores:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to compare embeddings (e.g., compare image and text embeddings, compare multiple embeddings, compute similarity scores), enabling embedding-to-similarity workflows
- **Before logic blocks** like Continue If to use similarity scores in conditions (e.g., continue if similarity exceeds threshold, filter based on similarity, make decisions using similarity), enabling similarity-based decision workflows
- **Before filtering blocks** to filter based on similarity (e.g., filter by similarity threshold, remove low-similarity items, keep high-similarity matches), enabling similarity-to-filter workflows
- **Before data storage blocks** to store similarity scores (e.g., store similarity metrics, log similarity comparisons, save similarity results), enabling similarity storage workflows
- **Before notification blocks** to send similarity-based alerts (e.g., notify on high similarity matches, alert on similarity changes, send similarity reports), enabling similarity notification workflows
- **In workflow outputs** to provide similarity scores as final output (e.g., similarity comparison outputs, matching results, similarity metrics), enabling similarity output workflows

## Requirements

This block requires two embedding vectors with the same dimensionality (same number of elements). Embeddings can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. The embeddings are passed as lists of floats. The block computes cosine similarity using the dot product divided by the product of L2 norms, producing a similarity score between -1 and 1. Values closer to 1 indicate greater similarity, values closer to 0 indicate orthogonal vectors, and values closer to -1 indicate opposite directions.
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
        description="First embedding vector to compare. Must have the same dimensionality (same number of elements) as embedding_2. Can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. Embedding vectors are lists of floats representing high-dimensional feature representations.",
        examples=[
            "$steps.clip_image.embedding",
            "$steps.perception_encoder.embedding",
            "$steps.clip_text.embedding",
        ],
    )
    embedding_2: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Second embedding vector to compare. Must have the same dimensionality (same number of elements) as embedding_1. Can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. Embedding vectors are lists of floats representing high-dimensional feature representations. The cosine similarity measures the similarity between embedding_1 and embedding_2.",
        examples=[
            "$steps.clip_text.embedding",
            "$steps.clip_image.embedding",
            "$steps.perception_encoder.embedding",
        ],
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
