"""Cosine similarity of two embeddings: ``v2/cosine_similarity@v1``.

Port of V1 ``roboflow_core/cosine_similarity@v1``::

    similarity = (a . b) / (|a| |b|)          in [-1, 1]

The dot product and both norms run on the embeddings' device, in at least
float32; one device-to-host copy of those three numbers produces the float
output. A zero-length vector has no direction, so it fails the step instead
of returning V1's NaN.
"""

from typing import Any, Dict

import torch
from pydantic import Field
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.data.predictions import EMBEDDING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class CosineSimilarityBlock(Block):
    """Compare two embeddings by direction, ignoring their length.

    Example: ``[1, 0]`` and ``[2, 0]`` give 1.0, ``[1, 0]`` and ``[0, 3]``
    give 0.0 and ``[1, 0]`` and ``[-1, 0]`` give -1.0.
    """

    type = "v2/cosine_similarity@v1"
    metadata = {
        "name": "Cosine Similarity",
        "port_of": "roboflow_core/cosine_similarity@v1",
        "section": "advanced",
    }
    outputs = {
        "similarity": Output(
            FLOAT_KIND, description="Cosine similarity, from -1 to 1."
        ),
    }
    prunable = True

    class Params(BlockParams):
        embedding_1: Ref(EMBEDDING_KIND) = Field(
            description="First embedding, e.g. of an image.",
            examples=["$steps.clip_image.embedding"],
        )
        embedding_2: Ref(EMBEDDING_KIND) = Field(
            description=(
                "Second embedding, with as many values as embedding_1 and on "
                "the same device."
            ),
            examples=["$steps.clip_text.embedding"],
        )

    def run(
        self, *, embedding_1: torch.Tensor, embedding_2: torch.Tensor
    ) -> Dict[str, Any]:
        """Return ``{"similarity": float}`` of the two embeddings.

        Raises:
            ValueError: When the embeddings differ in length or device, or one
                of them is a zero vector.
        """
        similarity = cosine_similarity(embedding_1, embedding_2, block=self.type)

        return {"similarity": similarity}


def cosine_similarity(
    embedding_1: torch.Tensor, embedding_2: torch.Tensor, *, block: str
) -> float:
    """Cosine similarity of two 1-D embeddings.

    Args:
        embedding_1: First 1-D floating embedding.
        embedding_2: Second, with the same length and device.
        block: Block type, for error messages.

    Returns:
        The similarity, from -1 to 1.

    Raises:
        ValueError: On a length or device mismatch or a zero vector.
    """
    if embedding_1.shape != embedding_2.shape:
        raise ValueError(
            f"{block}: embeddings have different lengths, "
            f"{embedding_1.shape[0]} and {embedding_2.shape[0]}; compare "
            "embeddings of the same model"
        )
    if embedding_1.device != embedding_2.device:
        raise ValueError(
            f"{block}: embeddings are on different devices, {embedding_1.device} "
            f"and {embedding_2.device}; embed both with models on one device"
        )

    dtype = torch.promote_types(
        torch.promote_types(embedding_1.dtype, embedding_2.dtype), torch.float32
    )
    first = embedding_1.to(dtype)
    second = embedding_2.to(dtype)
    dot, norm_1, norm_2 = torch.stack(
        [torch.dot(first, second), first.norm(), second.norm()]
    ).tolist()
    if norm_1 == 0 or norm_2 == 0:
        which = "embedding_1" if norm_1 == 0 else "embedding_2"
        raise ValueError(
            f"{block}: {which} is a zero vector, which has no direction; "
            "cosine similarity is undefined"
        )

    # Rounding can leave the quotient just outside [-1, 1].
    similarity = min(1.0, max(-1.0, dot / (norm_1 * norm_2)))

    return similarity
