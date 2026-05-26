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
Identify outlier embeddings compared to prior data using von Mises-Fisher statistical distribution analysis to detect anomalies, unusual patterns, or deviations from normal behavior by comparing current embedding vectors against a sliding window of historical embeddings for quality control, anomaly detection, and data sampling workflows.

## How This Block Works

This block detects outliers by statistically comparing embedding vectors against historical data using directional statistics. The block:

1. Receives an embedding vector representing the current data point's features
2. Normalizes the embedding to unit length:
   - Converts the embedding to a unit vector (length = 1) for directional analysis
   - Enables comparison using angular/directional statistics rather than distance-based metrics
   - Handles zero vectors gracefully by skipping normalization
3. Tracks sample count and warmup status:
   - Increments sample counter for each processed embedding
   - Determines if still in warmup period (samples < warmup parameter)
   - During warmup, no outliers are identified to allow baseline establishment
4. Maintains a sliding window of historical embeddings:
   - Stores normalized embeddings in a buffer that grows up to window_size
   - When buffer exceeds window_size, removes oldest embeddings (FIFO)
   - Creates a rolling history of recent data for statistical comparison
5. Fits von Mises-Fisher (vMF) distribution parameters during warmup completion:
   - **Mean Direction (mu)**: Calculates the average direction of all historical embeddings
   - **Concentration Parameter (kappa)**: Measures how tightly clustered the embeddings are around the mean
   - Uses statistical estimation to model the distribution of embedding directions
   - vMF distribution is ideal for directional data on a hypersphere (unit vectors)
6. Computes alignment score for current embedding:
   - Calculates dot product between current normalized embedding and mean direction vector
   - Measures how well the current embedding aligns with the typical direction
   - Higher values indicate closer alignment to the norm, lower values indicate deviation
7. Calculates empirical percentile of current embedding:
   - Computes alignment scores for all historical embeddings against the mean direction
   - Ranks the current embedding's alignment score among historical scores
   - Determines percentile position (0.0 = lowest, 1.0 = highest) of current embedding
8. Determines outlier status based on percentile thresholds:
   - Flags as outlier if percentile is below threshold_percentile (e.g., bottom 5%)
   - Flags as outlier if percentile is above (1 - threshold_percentile) (e.g., top 5%)
   - Detects both extreme low and extreme high deviations from the norm
9. Returns three outputs:
   - **is_outlier**: Boolean flag indicating if the current embedding is an outlier
   - **percentile**: Float value (0.0-1.0) representing where the embedding ranks among historical data
   - **warming_up**: Boolean flag indicating if still in warmup period (always False after warmup)

The block uses von Mises-Fisher distribution analysis, which is designed for directional data on a hypersphere (unit vectors). This makes it well-suited for high-dimensional embeddings where direction matters more than magnitude. The sliding window approach ensures the statistical model adapts to recent trends while the percentile-based detection identifies embeddings that are unusually different from the historical pattern. Lower percentiles indicate embeddings that are less aligned with typical patterns, while higher percentiles indicate embeddings that are unusually well-aligned or different in a positive direction.

## Common Use Cases

- **Anomaly Detection**: Detect unusual images, objects, or patterns that deviate from normal data (e.g., identify unusual product variations, detect anomalous behavior, flag unexpected patterns), enabling anomaly detection workflows
- **Quality Control**: Identify defective or unusual items in manufacturing or production (e.g., detect product defects, identify quality issues, flag manufacturing anomalies), enabling quality control workflows
- **Data Sampling**: Identify interesting or unusual data points for manual review or further analysis (e.g., sample unusual images for labeling, identify edge cases for model improvement, select interesting data for analysis), enabling intelligent data sampling workflows
- **Change Detection**: Detect when data patterns change significantly from historical norms (e.g., detect scene changes, identify pattern shifts, flag significant variations), enabling change detection workflows
- **Model Monitoring**: Monitor model performance by detecting when embeddings deviate from training distribution (e.g., detect distribution shift, identify out-of-distribution data, monitor model drift), enabling model monitoring workflows
- **Content Filtering**: Identify unusual or inappropriate content that differs from expected patterns (e.g., detect unusual content, flag inappropriate material, identify content anomalies), enabling content filtering workflows

## Connecting to Other Blocks

This block receives embeddings and produces is_outlier, percentile, and warming_up outputs:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to analyze embedding outliers (e.g., identify outliers from CLIP embeddings, analyze Perception Encoder outliers, detect anomalies from embeddings), enabling embedding-to-outlier workflows
- **After classification or detection blocks** with embeddings to identify unusual predictions (e.g., identify unusual detections, flag anomalous classifications, detect outlier predictions), enabling prediction-to-outlier workflows
- **Before logic blocks** like Continue If to make decisions based on outlier detection (e.g., continue if outlier detected, filter based on outlier status, trigger actions on anomalies), enabling outlier-based decision workflows
- **Before notification blocks** to alert on outlier detection (e.g., alert on anomalies, notify about unusual data, trigger alerts on outliers), enabling outlier-based notification workflows
- **Before data storage blocks** to record outlier information (e.g., log outlier data, store anomaly statistics, record unusual data points), enabling outlier data logging workflows
- **In quality control pipelines** where outlier detection is part of quality assurance (e.g., filter outliers in quality pipelines, identify issues in production workflows, detect problems in processing chains), enabling quality control workflows

## Requirements

This block requires embeddings as input (typically from embedding model blocks like CLIP or Perception Encoder). The block maintains internal state across workflow executions, accumulating a sliding window of historical embeddings. During the warmup period (first `warmup` samples), no outliers are identified and the block returns is_outlier=False and percentile=0.5. After warmup, the block uses at least `warmup` embeddings (up to `window_size` embeddings) to establish statistical baselines. The threshold_percentile parameter (0.0-1.0) controls sensitivity - lower values (e.g., 0.01) detect only extreme outliers, while higher values (e.g., 0.1) detect more moderate deviations. The block works best with consistent embedding models and may need adjustment of threshold_percentile based on expected variation in your data.
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
        description="Embedding vector representing the current data point's features. Typically from embedding models like CLIP or Perception Encoder. The embedding is normalized to unit length for directional statistical analysis using von Mises-Fisher distribution. Must be a numerical vector of any dimension.",
        examples=["$steps.clip.embedding", "$steps.perception_encoder.embedding"],
    )

    threshold_percentile: Union[Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]), float] = Field(
        default=0.05,
        description="Percentile threshold for outlier detection, range 0.0-1.0. Embeddings below this percentile or above (1 - threshold_percentile) are flagged as outliers. Lower values (e.g., 0.01) detect only extreme outliers - very strict. Higher values (e.g., 0.1) detect more moderate deviations - more sensitive. Default 0.05 means bottom 5% and top 5% are outliers. Adjust based on expected variation in your data.",
        examples=[0.05, 0.01, 0.1, "$inputs.threshold_percentile"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    warmup: Union[Selector(kind=[INTEGER_KIND]), int] = Field(
        default=3,
        description="Number of initial data points required before outlier detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay outlier detection. Lower values enable faster detection but may be less accurate initially.",
        examples=[3, 10, 100],
    )

    window_size: Optional[Union[Selector(kind=[INTEGER_KIND]), int]] = Field(
        default=32,
        description="Maximum number of historical embeddings to maintain in sliding window. The block keeps the most recent window_size embeddings for statistical comparison. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to distribution changes. Smaller windows adapt faster but may be less stable. Set to None for unlimited window (uses all historical data). Typical range: 10-100 embeddings.",
        examples=[32, 64, 100, None],
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
