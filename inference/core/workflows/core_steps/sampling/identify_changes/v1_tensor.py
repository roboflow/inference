import math
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
Identify changes and detect when data patterns change at unusual rates compared to historical norms by tracking embedding vectors over time, measuring cosine similarity changes, computing rate-of-change statistics, and flagging anomalies when changes occur faster or slower than expected for change detection, anomaly monitoring, rate-of-change analysis, and temporal pattern detection workflows.

## How This Block Works

This block detects changes by monitoring how quickly embeddings change over time and comparing the current rate of change against historical patterns. The block:

1. Receives an embedding vector representing the current data point's features
2. Normalizes the embedding to unit length:
   - Converts the embedding to a unit vector (length = 1) for cosine similarity calculations
   - Enables comparison using angular similarity rather than distance-based metrics
   - Handles zero vectors gracefully by skipping normalization
3. Tracks sample count and warmup status:
   - Increments sample counter for each processed embedding
   - Determines if still in warmup period (samples < warmup parameter)
   - During warmup, no outliers are identified to allow baseline establishment
4. Maintains running statistics for embedding averages and standard deviations using one of three strategies:

   **For Exponential Moving Average (EMA) strategy:**
   - Updates average and variance using exponential moving average with smoothing_factor
   - More recent embeddings have greater weight (controlled by smoothing_factor)
   - Adapts quickly to recent trends while maintaining historical context
   - Smoothing factor determines responsiveness (higher = more responsive to recent changes)

   **For Simple Moving Average (SMA) strategy:**
   - Uses Welford's method to calculate running mean and variance
   - All historical samples contribute equally to statistics
   - Provides stable, unbiased estimates over time
   - Well-suited for consistent, long-term tracking

   **For Sliding Window strategy:**
   - Maintains a fixed-size window of recent embeddings (window_size)
   - Removes oldest embeddings when window exceeds size (FIFO)
   - Calculates mean and standard deviation from window contents only
   - Adapts quickly to recent trends, discarding older information

5. Calculates cosine similarity between current embedding and running average:
   - Measures how similar the current embedding is to the typical embedding pattern
   - Cosine similarity ranges from -1 (opposite) to 1 (identical)
   - Values close to 1 indicate the embedding is similar to the norm
   - Values further from 1 indicate the embedding differs from the norm
6. Tracks rate of change by monitoring cosine similarity statistics:
   - Maintains running average and standard deviation of cosine similarity values
   - Uses the same strategy (EMA, SMA, or Sliding Window) for cosine similarity tracking
   - Measures how quickly embeddings are changing compared to historical change rates
   - Tracks both the average change rate and variability in change rates
7. Calculates z-score for current cosine similarity:
   - Measures how many standard deviations the current cosine similarity is from the average
   - Z-score = (current_cosine_similarity - average_cosine_similarity) / std_cosine_similarity
   - Positive z-scores indicate faster-than-normal changes
   - Negative z-scores indicate slower-than-normal changes
8. Converts z-score to percentile:
   - Uses error function (erf) to convert z-score to percentile position
   - Percentile represents where the current change rate ranks among historical rates
   - Values near 0.0 indicate unusually slow changes (low percentiles)
   - Values near 1.0 indicate unusually fast changes (high percentiles)
9. Determines outlier status based on percentile thresholds:
   - Flags as outlier if percentile is below threshold_percentile/2 (unusually slow changes)
   - Flags as outlier if percentile is above (1 - threshold_percentile/2) (unusually fast changes)
   - Detects both abnormally fast and abnormally slow rates of change
10. Updates running statistics for next iteration:
    - Updates embedding average and standard deviation using selected strategy
    - Updates cosine similarity average and standard deviation using selected strategy
    - Maintains state across workflow executions
11. Returns six outputs:
    - **is_outlier**: Boolean flag indicating if the rate of change is anomalous
    - **percentile**: Float value (0.0-1.0) representing where the change rate ranks historically
    - **z_score**: Float value representing standard deviations from average change rate
    - **average**: Current average embedding vector (running average of historical embeddings)
    - **std**: Current standard deviation vector for embeddings (variability per dimension)
    - **warming_up**: Boolean flag indicating if still in warmup period

The block monitors the **rate of change** rather than just detecting outliers. It tracks how quickly embeddings are changing and compares this to historical change patterns. When embeddings change much faster or slower than they have in the past, the block flags this as anomalous. This makes it ideal for detecting sudden pattern shifts, unexpected changes in scenes, or unusual behavior patterns. The three strategies (EMA, SMA, Sliding Window) offer different trade-offs between responsiveness and stability.

## Common Use Cases

- **Change Detection**: Detect when scenes, environments, or patterns change unexpectedly (e.g., detect scene changes, identify sudden pattern shifts, flag unexpected environmental changes), enabling change detection workflows
- **Anomaly Monitoring**: Monitor for unusual changes in behavior or patterns (e.g., detect abnormal behavior changes, monitor unusual pattern variations, flag unexpected rate changes), enabling anomaly monitoring workflows
- **Rate-of-Change Analysis**: Analyze and detect unusual rates of change in data streams (e.g., detect unusually fast changes, identify unusually slow changes, monitor change rate patterns), enabling rate-of-change analysis workflows
- **Temporal Pattern Detection**: Identify when temporal patterns deviate from expected change rates (e.g., detect pattern disruptions, identify timeline anomalies, flag temporal inconsistencies), enabling temporal pattern detection workflows
- **Quality Monitoring**: Monitor for unexpected changes in quality or characteristics (e.g., detect quality degradation, identify unexpected quality changes, monitor characteristic variations), enabling quality monitoring workflows
- **Event Detection**: Detect significant events based on unusual change rates (e.g., detect significant events, identify important changes, flag notable pattern shifts), enabling event detection workflows

## Connecting to Other Blocks

This block receives embeddings and produces is_outlier, percentile, z_score, average, std, and warming_up outputs:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to analyze change rates from embeddings (e.g., detect changes from CLIP embeddings, analyze Perception Encoder change rates, monitor embedding-based changes), enabling embedding-to-change workflows
- **After classification or detection blocks** with embeddings to identify unusual change patterns (e.g., detect unusual detection changes, flag anomalous classification changes, monitor prediction pattern changes), enabling prediction-to-change workflows
- **Before logic blocks** like Continue If to make decisions based on change detection (e.g., continue if change detected, filter based on change rate, trigger actions on unusual changes), enabling change-based decision workflows
- **Before notification blocks** to alert on change detection (e.g., alert on significant changes, notify about pattern shifts, trigger alerts on rate anomalies), enabling change-based notification workflows
- **Before data storage blocks** to record change information (e.g., log change data, store change statistics, record rate-of-change metrics), enabling change data logging workflows
- **In monitoring pipelines** where change detection is part of continuous monitoring (e.g., monitor changes in observation systems, track pattern variations, detect anomalies in monitoring workflows), enabling change monitoring workflows

## Requirements

This block requires embeddings as input (typically from embedding model blocks like CLIP or Perception Encoder). The block maintains internal state across workflow executions, tracking running statistics for both embeddings and cosine similarity values. During the warmup period (first `warmup` samples), no outliers are identified and the block returns is_outlier=False, percentile=0.5, and warming_up=True. After warmup, the block uses the selected strategy (EMA, SMA, or Sliding Window) to track statistics and detect rate-of-change anomalies. The threshold_percentile parameter (0.0-1.0) controls sensitivity - lower values detect only extreme rate changes, while higher values detect more moderate rate deviations. The strategy choice affects responsiveness: EMA adapts quickly to recent trends, SMA provides stable long-term tracking, and Sliding Window adapts quickly but discards older information. The block works best with consistent embedding models and may need adjustment of threshold_percentile and strategy based on expected variation and change patterns in your data.
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
        description="Statistical strategy for tracking embedding and change rate statistics. 'Exponential Moving Average (EMA)': Adapts quickly to recent trends, more weight on recent data. 'Simple Moving Average (SMA)': Stable long-term tracking, all data contributes equally. 'Sliding Window': Fast adaptation, uses only recent window_size samples. EMA is best for adaptive monitoring, SMA for stable tracking, Sliding Window for rapid adaptation.",
        examples=[
            "Exponential Moving Average (EMA)",
            "Simple Moving Average (SMA)",
            "Sliding Window",
        ],
        json_schema_extra={
            "always_visible": True,
        },
    )

    embedding: Selector(kind=[EMBEDDING_KIND]) = Field(
        description="Embedding vector representing the current data point's features. Typically from embedding models like CLIP or Perception Encoder. The embedding is normalized to unit length for cosine similarity calculations. The block compares current embedding to running average and tracks rate of change over time.",
        examples=["$steps.clip.embedding", "$steps.perception_encoder.embedding"],
    )

    threshold_percentile: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.2,
        description="Percentile threshold for change rate anomaly detection, range 0.0-1.0. Change rates below threshold_percentile/2 or above (1 - threshold_percentile/2) are flagged as outliers. Lower values (e.g., 0.05) detect only extreme rate changes - very strict. Higher values (e.g., 0.3) detect more moderate rate deviations - more sensitive. Default 0.2 means bottom 10% and top 10% of change rates are outliers. Adjust based on expected variation in change rates.",
        examples=[0.2, 0.05, 0.3, "$inputs.threshold_percentile"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    warmup: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=3,
        description="Number of initial data points required before change detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment for change rates. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay change detection. Lower values enable faster detection but may be less accurate initially.",
        examples=[3, 10, 100],
    )

    smoothing_factor: Optional[
        Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float]
    ] = Field(
        default=0.1,
        description="Smoothing factor (alpha) for Exponential Moving Average strategy, range 0.0-1.0. Controls responsiveness to recent data - higher values make statistics more responsive to recent changes, lower values maintain more historical context. Example: 0.1 means 10% weight on current value, 90% on historical average. Typical range: 0.05-0.3. Only used when strategy is 'Exponential Moving Average (EMA)'.",
        examples=[0.1, 0.05, 0.25],
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
        description="Maximum number of recent embeddings to maintain in sliding window. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to changes. Smaller windows adapt faster but may be less stable. Only used when strategy is 'Sliding Window'. Must be at least 2. Typical range: 5-50 embeddings.",
        examples=[10, 20, 50],
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
            percentile = 1 - 0.5 * (1 + math.erf(z_score / np.sqrt(2)))

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
