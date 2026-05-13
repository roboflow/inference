# Workflows with dynamic Python Blocks

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow measuring bounding boxes overlap

In real world use-cases you may not be able to find all pieces of functionalities required to complete 
your workflow within existing blocks. 

In such cases you may create piece of python code and put it in workflow as a dynamic block. Specifically 
here, we define two dynamic blocks:

- `OverlapMeasurement` which will accept object detection predictions and provide for boxes 
of specific class matrix of overlap with all boxes of another class.

- `MaximumOverlap` that will take overlap matrix produced by `OverlapMeasurement` and calculate maximum overlap.

Dynamic block may be used to create steps, exactly as if those blocks were standard Workflow blocks 
existing in ecosystem. The workflow presented in the example predicts from object detection model and 
calculates overlap matrix. Later, only if more than one object is detected, maximum overlap is calculated.



??? tip "Workflow definition"

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

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>