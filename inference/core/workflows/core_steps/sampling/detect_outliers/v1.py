from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field, model_validator

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    EMBEDDING_KIND,
    FLOAT_KIND,
    BOOLEAN_KIND,
    INTEGER_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

import numpy as np

LONG_DESCRIPTION = """
Detect outlier embeddings compared to prior data.

This block accepts an embedding and compares it to an average and standard deviation of prior data.
If the embedding is an outlier, the block will return a boolean flag and the percentile of the embedding
along with other useful statistics about the distribution.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detect Outliers",
            "version": "v1",
            "short_description": "Detect outlier embeddings compared to prior data.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "video",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-chart-scatter-bubble",
            },
        }
    )
    type: Literal["roboflow_core/detect_outliers@v1"]
    name: str = Field(description="Unique name of step in workflows")

    threshold_percentile: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.05,
        description="The desired percentage of outliers to detect. The default of 5% captures data points that are more than 2 standard deviations away from the average.",
        examples=["$inputs.sample_rate", 0.01],
    ) 

    strategy: Literal[
            "Exponential Moving Average (EMA)",
            "Simple Moving Average (SMA)",
            "Sliding Window",
            "Custom",
    ] = Field(
        default="Exponential Moving Average (EMA)",
        description="The outlier detection algorithm to use.",
        examples=["Simple Moving Average (SMA)"],
    )

    warmup: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=3,
        description="The number of data points to use for the initial average calculation. No outliers are detected during this period.",
        examples=[100],
    )

    smoothing_factor: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.25,
        description="The smoothing factor for the EMA algorithm. The default of 0.25 means the most recent data point will carry 25% weight in the average. Higher values will make the average more responsive to recent data points.",
        examples=[0.1],
        json_schema_extra={
            "relevant_for": {
                "strategy": {"values": { "Exponential Moving Average (EMA)" }, "required": True},
            },
        },
    )

    window_size: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=10,
        description="The number of data points to consider in the sliding window algorithm.",
        examples=[5],
        json_schema_extra={
            "relevant_for": {
                "strategy": {"values": { "Sliding Window" }, "required": True},
            },
        },
    )

    custom_average: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Average embedding of the prior data to compare with (for Custom strategy).",
        examples=["$steps.custom_average.embedding"],
        json_schema_extra={
            "relevant_for": {
                "strategy": {"values": { "Custom" }, "required": True},
            },
        },
    )

    custom_std: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Standard deviation of the prior data to compare with (for Custom strategy).",
        examples=["$steps.custom_average.std"],
        json_schema_extra={
            "relevant_for": {
                "strategy": {"values": { "Custom" }, "required": True},
            },
        },
    )

    embedding: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding of the current data.",
        examples=["$steps.clip.embedding"],
    )

    @model_validator(mode="after")
    def validate(self) -> "BlockManifest":
        if self.strategy == "Custom" and self.custom_average is None:
            raise ValueError(
                f"`custom_average` parameter required to be set for strategy `Custom`"
            )
        
        if self.strategy == "Custom" and self.custom_std is None:
            raise ValueError(
                f"`custom_std` parameter required to be set for strategy `Custom`"
            )
        
        if self.strategy == "Custom" and len(self.custom_average) != len(self.custom_std):
            raise ValueError(
                f"`custom_average` and `custom_std` should have the same dimensions"
            )
        
        return self
    
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


class DetectOutliersBlockV1(WorkflowBlock):
    def __init__(
        self
    ):
        self.average = None
        self.std = None
        self.var = None  # For EMA variance tracking
        self.M2 = None   # For SMA variance tracking
        self.sliding_window = []
        self.samples = 0

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        threshold_percentile: float,
        strategy: str,
        smoothing_factor: float,
        window_size: int,
        warmup: int,
        custom_average: List[float],
        custom_std: float,
        embedding: List[float]
    ) -> BlockResult:
        is_outlier = False
        percentile = 0.5
        z_score = 0
        warming_up = False
        
        embedding = np.array(embedding)

        # determine if embedding is an outlier
        if self.samples > 0 and self.std is not None and np.all(self.std != 0):
            z_scores = (embedding - self.average) / self.std
            z_score = np.linalg.norm(z_scores)
            percentile = 1 - 0.5 * (1 + np.math.erf(z_score / np.sqrt(2)))

        if self.samples < warmup:
            is_outlier = False
            warming_up = True
        else:
            is_outlier = percentile <= threshold_percentile/2 or percentile >= (1 - threshold_percentile/2)
        
        # update average and std
        if self.average is None:
            self.average = embedding
            self.std = np.zeros_like(embedding)
            self.var = np.zeros_like(embedding)
            self.M2 = np.zeros_like(embedding)
        else:
            if strategy == "Exponential Moving Average (EMA)":
                # Update EMA average:
                self.average = (1 - smoothing_factor) * self.average + smoothing_factor * embedding
                
                # Update EMA variance:
                # var_new = (1 - alpha)*var_old + alpha*(x - new_avg)^2
                diff = embedding - self.average
                self.var = (1 - smoothing_factor)*self.var + smoothing_factor*(diff**2)
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

            elif strategy == "Custom":
                # Just set provided custom averages/stds
                self.average = np.array(custom_average)
                self.std = np.array(custom_std)

        self.samples = self.samples + 1

        return {
            "is_outlier": is_outlier,
            "percentile": percentile,
            "z_score": z_score,
            "average": self.average.tolist(),
            "std": self.std.tolist(),
            "warming_up": warming_up
        }
