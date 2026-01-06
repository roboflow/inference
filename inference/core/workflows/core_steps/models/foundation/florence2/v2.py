from typing import List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.models.foundation.florence2.v1 import (
    BaseManifest,
    Florence2BlockV1,
    GroundingSelectionMode,
    RELEVANT_TASKS_DOCS_DESCRIPTION,
    TaskType,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    ROBOFLOW_MODEL_ID_KIND,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest


class V2BlockManifest(BaseManifest):
    type: Literal["roboflow_core/florence_2@v2"]
    model_id: Union[WorkflowParameterSelector(kind=[ROBOFLOW_MODEL_ID_KIND]), str] = (
        Field(
            default="florence-2-base",
            description="Model to be used",
            examples=["florence-2-base"],
            json_schema_extra={"always_visible": True},
        )
    )
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Florence-2 Model",
            "version": "v2",
            "short_description": "Run Florence-2 on an image",
            "long_description": f"""
Run Microsoft's Florence-2 model to perform a wide range of computer vision tasks using a single unified model.

## What is Florence-2?

Florence-2 is a **unified Vision Language Model (VLM)** developed by Microsoft Research that can perform many different computer vision tasks using a single model architecture. Unlike traditional models that are specialized for one task (like object detection or OCR), Florence-2:
- **Performs multiple tasks** - object detection, instance segmentation, OCR, captioning, and more, all with the same model
- **Uses task-specific prompts** - each task is triggered by a special prompt token (e.g., `<OD>` for object detection, `<OCR>` for text recognition)
- **Supports detection grounding** - can use bounding boxes from other detection models to focus on specific regions
- **Provides structured outputs** - returns parsed results in consistent formats for easy integration

This makes Florence-2 incredibly versatile - you can switch between tasks just by changing the task type, without needing separate models for each task.

## How This Block Works

This block takes one or more images as input and processes them through Microsoft's Florence-2 model. Based on the **task type** you select, the block:
1. **Prepares the task-specific prompt** - adds the appropriate Florence-2 task token (e.g., `<OD>`, `<OCR>`, `<CAPTION>`) or uses your custom prompt
2. **Handles detection grounding** (if required) - for tasks that need bounding boxes, it extracts coordinates from previous detections or uses provided coordinates
3. **Sends the request to Florence-2** - processes the image with the task-specific prompt
4. **Parses and returns results** - provides both raw JSON output and parsed structured data, making it easy to use in workflows

The block supports many predefined task types, each optimized for specific use cases, or you can use "custom" mode with your own prompts (useful with fine-tuned models).

## Supported Task Types

The block supports the following task types:

{RELEVANT_TASKS_DOCS_DESCRIPTION}

## Inputs and Outputs

**Input:**
- **images**: One or more images to analyze (can be from workflow inputs or previous steps)
- **task_type**: The type of task to perform (determines which Florence-2 task token is used)
- **prompt**: Text prompt (required for "phrase-grounded-object-detection", "phrase-grounded-instance-segmentation", and "custom" tasks)
- **classes**: List of classes for open-vocabulary detection (required for "open-vocabulary-object-detection" task)
- **grounding_detection**: Bounding box coordinates or detections from previous blocks (required for detection-grounded tasks) - can be provided as `[x_min, y_min, x_max, y_max]` or as detection results from other blocks
- **grounding_selection_mode**: How to select a detection when multiple are provided - "first", "last", "biggest", "smallest", "most-confident", or "least-confident" (default: "first")
- **model_id**: Florence-2 model to use - "florence-2-base" (default, faster) or "florence-2-large" (more accurate, slower). Can also use Roboflow model IDs for custom/fine-tuned models

**Output:**
- **raw_output**: Raw JSON string response from Florence-2
- **parsed_output**: Parsed dictionary containing structured results (format depends on task type)
- **classes**: List of detected class labels (for detection tasks that extract labels)

## Key Configuration Options

- **task_type**: Select the task type that matches your use case - this determines which Florence-2 task token is used and what output format to expect
- **model_id**: Choose between "florence-2-base" (default, faster, good for most tasks) or "florence-2-large" (more accurate, slower, better for complex tasks). Can also use Roboflow model IDs for custom/fine-tuned Florence-2 models
- **grounding_detection**: For detection-grounded tasks, provide bounding boxes either as coordinates `[x_min, y_min, x_max, y_max]` or connect detections from previous blocks (e.g., object detection, instance segmentation)
- **grounding_selection_mode**: When multiple detections are provided, choose how to select one - "first" (default), "last", "biggest", "smallest", "most-confident", or "least-confident"
- **prompt**: For phrase-grounded or custom tasks, provide a text description of what to find or your custom prompt

## Common Use Cases

- **Multi-Task Workflows**: Use a single model for multiple tasks in one workflow - detect objects, then caption them, then extract text from regions
- **Detection-Grounded Analysis**: Chain detections from one model to analyze specific regions - detect objects, then classify or caption each detected region
- **Document Processing**: Extract text from documents using OCR, or detect and read text in specific regions
- **Image Understanding**: Generate captions (short, detailed, or very detailed) for images or specific regions
- **Open-Vocabulary Detection**: Detect objects based on text descriptions without training a custom model
- **Region Analysis**: Propose regions of interest, then analyze each region with different tasks (classification, captioning, OCR)

## Requirements

**⚠️ Important: Dedicated Inference Server Required**

This block requires **local execution** (cannot run remotely). A **GPU is highly recommended** for acceptable performance. You may want to use a dedicated deployment for Florence-2 models. The model requires the `transformers` library - install with `pip install inference[transformers]` or `pip install inference-gpu[transformers]` for GPU support.

## Connecting to Other Blocks

The outputs from this block can be connected to:
- **Parser blocks** (e.g., VLM as Detector v1) to convert Florence-2 outputs into standard detection formats
- **Visualization blocks** to draw bounding boxes, masks, or text overlays based on Florence-2 results
- **Filter blocks** to filter detections or results based on Florence-2's analysis
- **Detection-grounded tasks** - use Florence-2's detection outputs as input to other Florence-2 tasks that require grounding (e.g., detect objects, then segment or classify each one)
- **Conditional logic blocks** to route workflow execution based on Florence-2's results
- **Data storage blocks** to log results for analytics or audit trails

## Version Differences (v2 vs v1)

This version (v2) includes the following enhancement over v1:
- **Improved Model ID Handling**: The `model_version` parameter has been renamed to `model_id` for consistency with other workflow blocks. Additionally, v2 supports Roboflow model IDs, allowing you to use custom or fine-tuned Florence-2 models hosted on Roboflow in addition to the standard "florence-2-base" and "florence-2-large" models.
""",
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": ["Florence", "Florence-2", "Microsoft"],
            "is_vlm_block": True,
            "task_type_property": "task_type",
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-atom",
                "blockPriority": 5.5,
            },
        },
        protected_namespaces=(),
    )


class Florence2BlockV2(Florence2BlockV1):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return V2BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        model_id: str,
        task_type: TaskType,
        prompt: Optional[str],
        classes: Optional[List[str]],
        grounding_detection: Optional[
            Union[Batch[sv.Detections], List[int], List[float]]
        ],
        grounding_selection_mode: GroundingSelectionMode,
    ) -> BlockResult:
        return super().run(
            images=images,
            model_version=model_id,
            task_type=task_type,
            prompt=prompt,
            classes=classes,
            grounding_detection=grounding_detection,
            grounding_selection_mode=grounding_selection_mode,
        )
