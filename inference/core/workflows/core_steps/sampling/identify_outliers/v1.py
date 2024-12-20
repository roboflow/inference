from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    EMBEDDING_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Identify outlier embeddings compared to prior data.

This block accepts an embedding and compares it to a sample of prior data.
If the embedding is an outlier, the block will return a boolean flag and the
percentile of the embedding.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Identify Outliers",
            "version": "v1",
            "short_description": "Identify outlier embeddings compared to prior data.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "video",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-chart-scatter-bubble",
            },
        }
    )
    type: Literal["roboflow_core/identify_outliers@v1"]
    name: str = Field(description="Unique name of step in workflows")

    embedding: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding of the current data.",
        examples=["$steps.clip.embedding"],
    )

    threshold_percentile: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.05,
        description="The desired sensitivity. A higher value will result in more data points being classified as outliers.",
        examples=["$inputs.sample_rate", 0.01],
        json_schema_extra={
            "always_visible": True,
        },
    )

    warmup: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=3,
        description="The number of data points to use for the initial average calculation. No outliers are identified during this period.",
        examples=[100],
    )

    window_size: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=32,
        description="The number of previous data points to consider in the sliding window algorithm.",
        examples=[5],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="is_outlier", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="percentile", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="warming_up", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class IdentifyOutliersBlockV1(WorkflowBlock):
    def __init__(self):
        self.samples = 0

        # Keep track of all embeddings for vMF parameter estimation:
        self.all_embeddings = []  # Store normalized embeddings

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def _fit_vmf_parameters(self, embeddings: np.ndarray):
        """
        Fit a von Mises-Fisher distribution to the given set of unit-normalized embeddings.
        Returns:
            mu (np.ndarray): Mean direction vector.
            kappa (float): Concentration parameter.
        """
        n, d = embeddings.shape
        if n < 2:
            # Not enough data to fit
            return None, None

        # Sum all embeddings:
        sum_vec = np.sum(embeddings, axis=0)
        R = np.linalg.norm(sum_vec)
        if R == 0:
            # All embeddings canceled out, no direction
            return None, None

        mu = sum_vec / R
        r_bar = R / n

        # Approximate kappa:
        # For d>2, a known approximation:
        # kappa ≈ r_bar*(d - r_bar^2)/(1 - r_bar^2)
        d = float(d)
        if r_bar < 1.0:
            kappa = (r_bar * (d - r_bar**2)) / (1 - r_bar**2)
        else:
            # Degenerate case: all points are identical
            kappa = np.inf

        return mu, kappa

    def run(
        self,
        embedding: List[float],
        threshold_percentile: float,
        warmup: int,
        window_size: int,
    ) -> BlockResult:
        # Convert to np array
        embedding = np.array(embedding, dtype=float)
        # Normalize embedding for vMF
        norm = np.linalg.norm(embedding)
        if norm == 0:
            # If zero vector, skip normalization
            embedding_normed = embedding
        else:
            embedding_normed = embedding / norm

        self.samples += 1
        warming_up = self.samples < warmup

        # Store normalized embedding for vMF:
        self.all_embeddings.append(embedding_normed)
        if len(self.all_embeddings) > window_size:
            # Remove oldest embedding
            self.all_embeddings.pop(0)

        # If we're still in warmup, we cannot decide outliers yet
        if warming_up:
            return {
                "is_outlier": False,
                "percentile": 0.5,
                "warming_up": True,
            }

        # Fit vMF parameters based on all embeddings so far
        all_emb_array = np.array(self.all_embeddings)
        mu, kappa = self._fit_vmf_parameters(all_emb_array)
        if mu is None or kappa is None:
            # Fallback if we cannot fit (e.g. all embeddings identical)
            mu = embedding_normed
            kappa = 0.0

        # Compute alignment score with the current mean direction
        t_new = np.dot(mu, embedding_normed)

        # Compute empirical percentile of t_new relative to historical t_i
        # Using all previous embeddings (excluding the current one if desired)
        t_values = np.einsum("ij,j->i", all_emb_array, mu)
        # Sort t-values to find percentile
        sorted_t = np.sort(t_values)
        rank = np.searchsorted(sorted_t, t_new, side="left")
        percentile = rank / len(sorted_t)

        # Determine outlier based on percentile thresholds
        is_outlier = (percentile < threshold_percentile) or (
            percentile > (1 - threshold_percentile)
        )

        return {
            "is_outlier": bool(is_outlier),
            "percentile": float(percentile),
            "warming_up": warming_up,
        }
