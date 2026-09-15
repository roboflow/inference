from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = (
    "Map the most confident detected class to a value, e.g. a scene name."
)

LONG_DESCRIPTION = """
Turn detections into a single routed value: look at the detections whose class appears in
`routes`, take the most confident one, and emit the value mapped to its class. Classes that are
not in `routes` never take part, so a confidently detected `person` or `chair` cannot outrank the
class you actually care about.

## How This Block Works

1. Keeps only detections whose class name is a key in `routes` and whose confidence is at least
   `confidence_threshold`
2. Picks the most confident remaining detection
3. Emits the value mapped to its class as `value`, alongside `matched_class` and `matched`

When nothing routed is present, `value` is `default_value`. Leave `default_value` empty to emit
nothing (`None`): sinks such as OBS Action then skip that execution, so the last state holds
instead of flapping - the right behaviour for live video, where most frames contain none of the
classes you are routing on.

## Common Use Cases

- **Scene switching**: `{"dog": "Dog", "cat": "Cat"}` feeding an OBS Action `set_scene`
- **Alert routing**: map classes to Slack channels, webhook URLs or message templates
- **Any detection-to-choice step** that would otherwise need a filter, a count, a gate, a
  property extraction and an expression chained together

## Connecting to Other Blocks

- **After a detection model**, or after a Detections Filter for extra conditions
- **Before OBS Action, Expression, Continue If**, or any block that consumes a plain value
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Class Router",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "search_keywords": ["route", "switch", "class", "scene", "map", "trigger"],
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-route",
                "blockPriority": 3,
                "popular": False,
            },
        }
    )
    type: Literal["roboflow_core/detections_class_router@v1"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detections to route on.",
        examples=["$steps.model.predictions"],
    )
    routes: Dict[str, Any] = Field(
        description="Class name to value. Only these classes take part in routing.",
        examples=[{"dog": "Dog", "cat": "Cat", "apple": "Apple"}],
        min_length=1,
    )
    default_value: Optional[Any] = Field(
        default=None,
        description="Value emitted when no routed class is detected. Leave empty to emit "
        "nothing, so downstream sinks hold their last state.",
        examples=["Fireworks"],
    )
    confidence_threshold: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.0,
        description="Minimum confidence for a detection to take part in routing.",
        examples=[0.4],
    )
    case_insensitive: bool = Field(
        default=True,
        description="Match class names regardless of capitalisation.",
        examples=[True],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="value", kind=[WILDCARD_KIND]),
            OutputDefinition(name="matched_class", kind=[STRING_KIND]),
            OutputDefinition(name="matched", kind=[BOOLEAN_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


class DetectionsClassRouterBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: sv.Detections,
        routes: Dict[str, Any],
        default_value: Optional[Any],
        confidence_threshold: float,
        case_insensitive: bool,
    ) -> BlockResult:
        normalise = (lambda s: str(s).lower()) if case_insensitive else str
        table = {normalise(k): (k, v) for k, v in routes.items()}
        best_key, best_confidence = None, -1.0
        if predictions is not None and len(predictions) > 0:
            class_names = predictions.data.get("class_name", [])
            confidences = predictions.confidence
            for name, confidence in zip(class_names, confidences):
                key = normalise(name)
                if key not in table or float(confidence) < confidence_threshold:
                    continue
                if float(confidence) > best_confidence:
                    best_key, best_confidence = key, float(confidence)
        if best_key is None:
            return {"value": default_value, "matched_class": None, "matched": False}
        original_class, value = table[best_key]
        return {"value": value, "matched_class": original_class, "matched": True}
