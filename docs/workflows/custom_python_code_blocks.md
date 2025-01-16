# Dynamic Python blocks

When the syntax for Workflow definitions was [outlined](/workflows/definitions.md), one key 
aspect was not covered: the ability to define blocks directly within the Workflow definition itself. This section can
include the manifest and Python code for blocks defined in-place, which are dynamically interpreted by the 
Execution Engine. These in-place blocks function similarly to those statically defined in 
[plugins](/workflows/blocks_bundling.md), yet provide much more flexibility.


!!! Warning

    Dynamic blocks only work in your local deployment of `inference` and are not supported 
    on the Roboflow hosted platform.

    If you wish to disable the functionality, `export ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS=False`

## Theory

The high-level overview of Dynamic Python blocks functionality:

* user provides definition of dynamic block in JSON

* definition contains information required by Execution Engine to
construct `WorkflowBlockManifest` and `WorkflowBlock` out of the 
document

* in runtime, Compiler turns definition into dynamically created 
Python classes - exactly the same as statically defined blocks

* In Workflow definition, you may declare steps that use dynamic blocks, 
as if dynamic blocks were standard static ones

  
## Example 

Let's take a look and discuss example workflow with dynamic Python blocks.

??? tip "Workflow with dynamic block"
    
    ```json
    {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowImage",
                "name": "image"
            }
        ],
        "dynamic_blocks_definitions": [
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "OverlapMeasurement",
                    "inputs": {
                        "predictions": {
                            "type": "DynamicInputDefinition",
                            "selector_types": [
                                "step_output"
                            ]
                        },
                        "class_x": {
                            "type": "DynamicInputDefinition",
                            "value_types": [
                                "string"
                            ]
                        },
                        "class_y": {
                            "type": "DynamicInputDefinition",
                            "value_types": [
                                "string"
                            ]
                        }
                    },
                    "outputs": {
                        "overlap": {
                            "type": "DynamicOutputDefinition",
                            "kind": []
                        }
                    }
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": "\ndef run(self, predictions: sv.Detections, class_x: str, class_y: str) -> BlockResult:\n    bboxes_class_x = predictions[predictions.data[\"class_name\"] == class_x]\n    bboxes_class_y = predictions[predictions.data[\"class_name\"] == class_y]\n    overlap = []\n    for bbox_x in bboxes_class_x:\n        bbox_x_coords = bbox_x[0]\n        bbox_overlaps = []\n        for bbox_y in bboxes_class_y:\n            if bbox_y[-1][\"detection_id\"] == bbox_x[-1][\"detection_id\"]:\n                continue\n            bbox_y_coords = bbox_y[0]\n            x_min = max(bbox_x_coords[0], bbox_y_coords[0])\n            y_min = max(bbox_x_coords[1], bbox_y_coords[1])\n            x_max = min(bbox_x_coords[2], bbox_y_coords[2])\n            y_max = min(bbox_x_coords[3], bbox_y_coords[3])\n            # compute the area of intersection rectangle\n            intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)\n            box_x_area = (bbox_x_coords[2] - bbox_x_coords[0] + 1) * (bbox_x_coords[3] - bbox_x_coords[1] + 1)\n            local_overlap = intersection_area / (box_x_area + 1e-5)\n            bbox_overlaps.append(local_overlap)\n        overlap.append(bbox_overlaps)\n    return  {\"overlap\": overlap}\n"
                }
            },
            {
                "type": "DynamicBlockDefinition",
                "manifest": {
                    "type": "ManifestDescription",
                    "block_type": "MaximumOverlap",
                    "inputs": {
                        "overlaps": {
                            "type": "DynamicInputDefinition",
                            "selector_types": [
                                "step_output"
                            ]
                        }
                    },
                    "outputs": {
                        "max_value": {
                            "type": "DynamicOutputDefinition",
                            "kind": []
                        }
                    }
                },
                "code": {
                    "type": "PythonCode",
                    "run_function_code": "\ndef run(self, overlaps: List[List[float]]) -> BlockResult:\n    max_value = -1\n    for overlap in overlaps:\n        for overlap_value in overlap:\n            if not max_value:\n                max_value = overlap_value\n            else:\n                max_value = max(max_value, overlap_value)\n    return {\"max_value\": max_value}\n"
                }
            }
        ],
        "steps": [
            {
                "type": "RoboflowObjectDetectionModel",
                "name": "model",
                "image": "$inputs.image",
                "model_id": "yolov8n-640"
            },
            {
                "type": "OverlapMeasurement",
                "name": "overlap_measurement",
                "predictions": "$steps.model.predictions",
                "class_x": "dog",
                "class_y": "dog"
            },
            {
                "type": "ContinueIf",
                "name": "continue_if",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "overlaps",
                                "operations": [
                                    {
                                        "type": "SequenceLength"
                                    }
                                ]
                            },
                            "comparator": {
                                "type": "(Number) >="
                            },
                            "right_operand": {
                                "type": "StaticOperand",
                                "value": 1
                            }
                        }
                    ]
                },
                "evaluation_parameters": {
                    "overlaps": "$steps.overlap_measurement.overlap"
                },
                "next_steps": [
                    "$steps.maximum_overlap"
                ]
            },
            {
                "type": "MaximumOverlap",
                "name": "maximum_overlap",
                "overlaps": "$steps.overlap_measurement.overlap"
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "overlaps",
                "selector": "$steps.overlap_measurement.overlap"
            },
            {
                "type": "JsonField",
                "name": "max_overlap",
                "selector": "$steps.maximum_overlap.max_value"
            }
        ]
    }
    ```

Let's start the analysis from `dynamic_blocks_definitions` - this is the part of 
Workflow Definition that provides a list of dynamic blocks. Each block contains two sections:

* `manifest` - providing JSON representation of `BlockManifest` - refer [blocks development guide](/workflows/create_workflow_block.md)

* `code` - shipping Python code


### Definition of block manifest

Manifest definition contains several fields, including:

* `block_type` - equivalent of `type` field in block manifest - must provide unique block identifier

* `inputs` - dictionary with names and definitions of dynamic inputs

* `outputs` - dictionary with names and definitions of dynamic outputs

* `output_dimensionality_offset` - field specifies output dimensionality

* `accepts_batch_input` - field dictates if input data in runtime is to be provided in batches
by Execution Engine

* `accepts_empty_values` - field deciding if empty inputs will be ignored while 
constructing step inputs

In any doubt, refer to [blocks development guide](/workflows/create_workflow_block.md), as
the dynamic blocks replicates standard blocs capabilities.


### Definition of dynamic input

Dynamic inputs define fields of dynamically created block manifest. In other words, 
this is definition based on which `BlockManifest` class will be created in runtime.

Each input may define the following properties:

* `has_default_value` - flag to decide if dynamic manifest field has default

* `default_value` - default value (used only if `has_default_value=True`

* `is_optional` - flag to decide if dynamic manifest field is optional

* `is_dimensionality_reference` - flag to decide if dynamic manifest field ship
selector to be used in runtime as dimensionality reference

* `dimensionality_offset` - dimensionality offset for configured input property 
of dynamic manifest

* `selector_types` - type of selectors that may be used by property (one of 
`input_image`, `step_output_image`, `input_parameter`, `step_output`). Step may not
hold selector, but then must provide definition of specific type.

* `selector_data_kind` - dictionary with list of selector kinds specific for each selector type

* `value_types` - definition of specific type that is to be placed in manifest - 
this field specifies typing of dynamically created manifest fields w.r.t Python types.
Selection of types: `any`, `integer`, `float`, `boolean`, `dict`, `list`, `strig`

### Definition of dynamic output

Definitions of outputs are quite simple, hold optional list of `kinds` declared
for given output.


### Definition of Python code

Python code is shipped in JSON document with the following fields:

* `run_function_code` - code of `run(...)` method of your dynamic block 

* `run_function_name` - name of run function

* `init_function_code` - optional code for your init function that will 
assemble step state - it is expected to return dictionary, which will be available for `run()`
function under `self._init_results`

* `init_function_name` - name of init function

* `imports` - list of additional imports (you may only use libraries from your environment, no dependencies will be 
automatically installed)


### How to create `run(...)` method?

You must know the following:

* `run(...)` function must be defined, as if that was class instance method - with 
the first argument being `self` and remaining arguments compatible with dynamic block manifest
declared in definition of dynamic block

* you should expect baseline symbols to be provided, including your import statements and
the following:

```python
from typing import Any, List, Dict, Set, Optional
import supervision as sv
import numpy as np
import math
import time
import json
import os
import requests
import cv2
import shapely
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult
```

So example function may look like the following (for clarity, we provide here
Python code formatted nicely, but you must stringify the code to place it in definition):

```python
def run(self, predictions: sv.Detections, class_x: str, class_y: str) -> BlockResult:
    bboxes_class_x = predictions[predictions.data["class_name"] == class_x]
    bboxes_class_y = predictions[predictions.data["class_name"] == class_y]
    overlap = []
    for bbox_x in bboxes_class_x:
        bbox_x_coords = bbox_x[0]
        bbox_overlaps = []
        for bbox_y in bboxes_class_y:
            if bbox_y[-1]["detection_id"] == bbox_x[-1]["detection_id"]:
                continue
            bbox_y_coords = bbox_y[0]
            x_min = max(bbox_x_coords[0], bbox_y_coords[0])
            y_min = max(bbox_x_coords[1], bbox_y_coords[1])
            x_max = min(bbox_x_coords[2], bbox_y_coords[2])
            y_max = min(bbox_x_coords[3], bbox_y_coords[3])
            # compute the area of intersection rectangle
            intersection_area = max(0, x_max - x_min + 1) * max(0, y_max - y_min + 1)
            box_x_area = (bbox_x_coords[2] - bbox_x_coords[0] + 1) * (bbox_x_coords[3] - bbox_x_coords[1] + 1)
            local_overlap = intersection_area / (box_x_area + 1e-5)
            bbox_overlaps.append(local_overlap)
        overlap.append(bbox_overlaps)
    return  {"overlap": overlap}
```

### How to create `init(...)` method?

Init function is supposed to build `self._init_results` dictionary.

Example:

```python

def my_init() -> Dict[str, Any]:
    return {"some": "value"}
```

### Usage of Dynamic Python block as step

As shown in example Workflow definition, you may simply use the block 
as if that was normal block exposed through static plugin:

```json
{
    "type": "OverlapMeasurement",
    "name": "overlap_measurement",
    "predictions": "$steps.model.predictions",
    "class_x": "dog",
    "class_y": "dog"
}
```