from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from inference.core.utils.postprocess import cosine_similarity
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    EMBEDDING_KIND,
    FLOAT_KIND,
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
Identify changes compared to prior data via embeddings.

This block accepts an embedding and compares it to a prior average
and standard deviation for the rate of change. When things change
faster or slower than they have in the past, the block will flag
the data as an outlier.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Identify Changes",
            "version": "v1",
            "short_description": "Identify changes compared to prior data via embeddings.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "video",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-triangle",
            },
        }
    )
    type: Literal["roboflow_core/identify_changes@v1"]
    name: str = Field(description="Unique name of step in workflows")

    strategy: Literal[
        "Exponential Moving Average (EMA)",
        "Simple Moving Average (SMA)",
        "Sliding Window",
    ] = Field(
        default="Exponential Moving Average (EMA)",
        description="The change identification algorithm to use.",
        examples=["Simple Moving Average (SMA)"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    embedding: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding of the current data.",
        examples=["$steps.clip.embedding"],
    )

    threshold_percentile: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.2,
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

    smoothing_factor: Optional[
        Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float]
    ] = Field(
        default=0.1,
        description="The smoothing factor for the EMA algorithm. The default of 0.25 means the most recent data point will carry 25% weight in the average. Higher values will make the average more responsive to recent data points.",
        examples=[0.1],
        json_schema_extra={
            "relevant_for": {
                "strategy": {
                    "values": {"Exponential Moving Average (EMA)"},
                },
            },
        },
    )

    window_size: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=10,
        description="The number of data points to consider in the sliding window algorithm.",
        examples=[5],
        json_schema_extra={
            "relevant_for": {
                "strategy": {"values": {"Sliding Window"}, "required": True},
            },
        },
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="is_outlier", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="percentile", kind=[FLOAT_ZERO_TO_ONE_KIND]),
            OutputDefinition(name="z_score", kind=[FLOAT_KIND]),
            OutputDefinition(name="average", kind=[EMBEDDING_KIND]),
            OutputDefinition(name="std", kind=[EMBEDDING_KIND]),
            OutputDefinition(name="warming_up", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class IdentifyChangesBlockV1(WorkflowBlock):
    def __init__(self):
        self.average = None
        self.std = None
        self.var = None  # For EMA variance tracking
        self.M2 = None  # For SMA variance tracking
        self.sliding_window = []
        self.samples = 0

        self.cosine_similarity_avg = None
        self.cosine_similarity_std = None
        self.cosine_similarity_var = None
        self.cosine_similarity_m2 = None
        self.cosine_similarity_sliding_window = []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        strategy: str,
        embedding: List[float],
        threshold_percentile: float,
        smoothing_factor: float,
        window_size: int,
        warmup: int,
    ) -> BlockResult:
        is_outlier = False
        percentile = 0.5
        z_score = 0
        warming_up = False

        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm != 0:
            embedding = embedding / norm

        # determine if embedding is an outlier
        if self.average is not None:
            cs = cosine_similarity(embedding, self.average)

            if self.cosine_similarity_avg is None:
                self.cosine_similarity_avg = cs
                self.cosine_similarity_std = 0
                self.cosine_similarity_var = 0
                self.cosine_similarity_m2 = 0
            else:
                if strategy == "Exponential Moving Average (EMA)":
                    # Update EMA average:
                    self.cosine_similarity_avg = (
                        1 - smoothing_factor
                    ) * self.cosine_similarity_avg + smoothing_factor * cs

                    # Update EMA variance:
                    # var_new = (1 - alpha)*var_old + alpha*(x - new_avg)^2
                    diff = cs - self.cosine_similarity_avg
                    self.cosine_similarity_var = (
                        1 - smoothing_factor
                    ) * self.cosine_similarity_var + smoothing_factor * (diff**2)
                    self.cosine_similarity_std = np.sqrt(self.cosine_similarity_var)
                elif strategy == "Simple Moving Average (SMA)":
                    count = self.samples + 1
                    delta = cs - self.cosine_similarity_avg

                    self.cosine_similarity_avg = (
                        cs / count + self.cosine_similarity_avg * self.samples / count
                    )
                    delta2 = cs - self.cosine_similarity_avg

                    self.cosine_similarity_m2 = (
                        self.cosine_similarity_m2 + delta * delta2
                    )
                    var = self.cosine_similarity_m2 / (count - 1)
                    self.cosine_similarity_std = np.sqrt(var)
                elif strategy == "Sliding Window":
                    self.cosine_similarity_sliding_window.append(cs)
                    if len(self.cosine_similarity_sliding_window) > window_size:
                        self.cosine_similarity_sliding_window.pop(0)

                    self.cosine_similarity_avg = np.mean(
                        self.cosine_similarity_sliding_window
                    )
                    self.cosine_similarity_std = np.std(
                        self.cosine_similarity_sliding_window
                    )

            z_score = (cs - self.cosine_similarity_avg) / self.cosine_similarity_std
            percentile = 1 - 0.5 * (1 + np.math.erf(z_score / np.sqrt(2)))

            # print(f"Z-score: {z_score}, Percentile: {percentile}, Cosine Similarity: {cs}, Average: {self.cosine_similarity_avg}, Std: {self.cosine_similarity_std}")

        if self.samples < warmup:
            is_outlier = False
            warming_up = True
        else:
            is_outlier = percentile <= threshold_percentile / 2 or percentile >= (
                1 - threshold_percentile / 2
            )

        # update average and std
        if self.average is None:
            self.average = embedding
            self.std = np.zeros_like(embedding)
            self.var = np.zeros_like(embedding)
            self.M2 = np.zeros_like(embedding)
        else:
            if strategy == "Exponential Moving Average (EMA)":
                # Update EMA average:
                self.average = (
                    1 - smoothing_factor
                ) * self.average + smoothing_factor * embedding

                # Update EMA variance:
                # var_new = (1 - alpha)*var_old + alpha*(x - new_avg)^2
                diff = embedding - self.average
                self.var = (1 - smoothing_factor) * self.var + smoothing_factor * (
                    diff**2
                )
                self.std = np.sqrt(self.var)

            elif strategy == "Simple Moving Average (SMA)":
                # Use Welford's method to update mean and variance
                count = self.samples + 1
                delta = embedding - self.average

                # Update average:
                self.average = self.average + delta / count
                delta2 = embedding - self.average

                # Update M2:
                self.M2 = self.M2 + delta * delta2
                var = self.M2 / (count - 1)
                self.std = np.sqrt(var)

            elif strategy == "Sliding Window":
                self.sliding_window.append(embedding)
                if len(self.sliding_window) > window_size:
                    self.sliding_window.pop(0)

                self.average = np.mean(self.sliding_window, axis=0)
                self.std = np.std(self.sliding_window, axis=0)

        self.samples = self.samples + 1

        return {
            "is_outlier": is_outlier,
            "percentile": percentile,
            "z_score": z_score,
            "average": self.average.tolist(),
            "std": self.std.tolist(),
            "warming_up": warming_up,
        }
