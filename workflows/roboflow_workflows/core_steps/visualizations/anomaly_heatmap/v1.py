from typing import List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.visualizations.common.annotators.anomaly_heatmap import (
    AnomalyHeatmapAnnotator,
)
from roboflow_workflows.core_steps.visualizations.common.base import (
    OUTPUT_IMAGE_KEY,
    VisualizationBlock,
    VisualizationManifest,
)
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    NUMPY_ARRAY_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlockManifest,
)

TYPE: str = "roboflow_core/anomaly_heatmap_visualization@v1"
SHORT_DESCRIPTION = (
    "Overlay an anomaly detection model's anomaly map on the image as a heatmap."
)
LONG_DESCRIPTION = """
Draw the anomaly map of an **Anomaly Detection Model** (PatchCore or FoundAD) over
the image it was computed for, so the parts of the image that depart from what the
model learned as normal light up.

## How This Block Works

1. Takes the image and the model's `anomaly_map` and `anomaly_threshold` as input
2. Stretches the map (served at the network's square input resolution) onto the image
3. Places every map value on a fixed scale anchored to the threshold:
   `t = value / (value + threshold)`, so zero is transparent, the threshold sits at
   the middle of the ramp, and the ramp saturates as the evidence grows
4. Colours the scale with the chosen `color_map` and blends it over the image at
   the chosen `opacity`

The scale depends only on the model's threshold, never on the image, so the same
colour means the same evidence on every frame and normal images show their map too.
The default `ROBOFLOW` ramp is the yellow-to-red ramp the Roboflow app draws in its
model preview, with transparency that fades out where the evidence is low. The
OpenCV ramps (`JET`, `TURBO`, `INFERNO`, `VIRIDIS`, `HOT`) are opaque colour maps
over the same scale; use `opacity` to see the image through them.

Pass the model block's `anomaly_decision_threshold` output so the middle of the ramp
marks the block's decision (its threshold and `confidence`); at the default confidence
of 0.5 it equals `anomaly_threshold`.

## Common Use Cases

- **Defect localisation**: show an operator where an anomalous part differs from normal
- **Model review**: compare what the model reacts to against the labelled defect
- **Reports and alerts**: attach the heatmap to a notification or dataset upload

## Connecting to Other Blocks

The annotated image can be connected to other visualization blocks (for example the
Classification Label Visualization to print the verdict on top), data storage blocks,
webhook and notification blocks, or video outputs.
"""


class AnomalyHeatmapManifest(VisualizationManifest):
    type: Literal[f"{TYPE}"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Anomaly Heatmap Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": [
                "anomaly",
                "anomaly detection",
                "heatmap",
                "defect",
                "PatchCore",
                "FoundAD",
            ],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-fire",
                "blockPriority": 3,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )

    anomaly_map: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(  # type: ignore
        description="The anomaly map from the Anomaly Detection Model block: raw local anomaly evidence at the network input resolution.",
        examples=["$steps.anomaly_model.anomaly_map"],
    )
    anomaly_threshold: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="The raw anomaly score that anchors the colour ramp; it maps to the middle of the ramp. Connect the model block's anomaly_decision_threshold output so the ramp follows the block's decision.",
        examples=["$steps.anomaly_model.anomaly_decision_threshold", 1.25],
    )
    color_map: Union[
        Literal["ROBOFLOW", "JET", "TURBO", "INFERNO", "VIRIDIS", "HOT"],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="ROBOFLOW",
        description="Colour ramp for the heatmap. ROBOFLOW is the Roboflow app's yellow-to-red ramp, transparent where the evidence is low; the others are opaque OpenCV colour maps.",
        examples=["ROBOFLOW", "JET", "$inputs.color_map"],
    )
    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        default=0.75,
        description="Opacity of the heatmap overlay, from 0.0 (invisible) to 1.0 (the ramp's own transparency only).",
        examples=[0.75, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.VISUALIZATION]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class AnomalyHeatmapVisualizationBlockV1(VisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AnomalyHeatmapManifest

    def getAnnotator(
        self,
        color_map: str,
        opacity: float,
    ) -> AnomalyHeatmapAnnotator:
        key = "_".join(map(str, [color_map, opacity]))
        if key not in self.annotatorCache:
            self.annotatorCache[key] = AnomalyHeatmapAnnotator(
                color_map=color_map,
                opacity=opacity,
            )
        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        copy_image: bool,
        anomaly_map: Optional[np.ndarray],
        anomaly_threshold: float,
        color_map: str,
        opacity: float,
    ) -> BlockResult:
        if anomaly_map is None:
            raise ValueError(
                "Anomaly Heatmap Visualization received no anomaly map. Enable "
                "`include_anomaly_map` on the Anomaly Detection Model block that "
                "feeds this step."
            )
        annotator = self.getAnnotator(color_map, opacity)
        scene = image.numpy_image
        if copy_image:
            scene = scene.copy()
        else:
            image.declare_numpy_image_mutated()
        annotated_image = annotator.annotate(
            scene=scene,
            anomaly_map=np.asarray(anomaly_map, dtype=np.float32),
            threshold=float(anomaly_threshold),
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
