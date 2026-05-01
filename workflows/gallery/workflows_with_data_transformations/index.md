# Workflows with data transformations

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with detections class remapping

This workflow presents how to use Detections Transformation block that is going to 
change the name of the following classes: `apple`, `banana` into `fruit`.

In this example, we use non-strict mapping, causing new class `fruit` to be added to
pool of classes - you can see that if `banana` or `apple` is detected, the
class name changes to `fruit` and class id is 1024.

You can test the execution submitting image like 
[this](https://www.pexels.com/photo/four-trays-of-varieties-of-fruits-1300975/).

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiS1JyYU5sS2hjZkl2bTQzNlBCYUYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTd9.p5ks40w86AsWHgPQNq0kYnh6Fpulo3GdLmjKEF92q7w?showGraph=true" loading="lazy" title="Roboflow Workflow for detections class remapping" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "ObjectDetectionModel",
	            "name": "model",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "confidence": "$inputs.confidence"
	        },
	        {
	            "type": "DetectionsTransformation",
	            "name": "class_rename",
	            "predictions": "$steps.model.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsRename",
	                    "strict": false,
	                    "class_map": {
	                        "apple": "fruit",
	                        "banana": "fruit"
	                    }
	                }
	            ]
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "original_predictions",
	            "selector": "$steps.model.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "renamed_predictions",
	            "selector": "$steps.class_rename.predictions"
	        }
	    ]
	}
    ```

## Workflow with detections filtering

This example presents how to use Detections Transformation block to build workflow
that is going to filter predictions based on:

- predicted classes

- size of predicted bounding box relative to size of input image

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoicHNXU0I0WkgzTVd5ZkUxaTc5ek8iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTh9.KHGn3WZNjPqUC9xCO3iWZrrnnYIl_MLuBk_G_VWNIYI?showGraph=true" loading="lazy" title="Roboflow Workflow for detections filtering" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.3
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence": "$inputs.confidence"
	        },
	        {
	            "type": "DetectionsTransformation",
	            "name": "filtering",
	            "predictions": "$steps.detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsFilter",
	                    "filter_operation": {
	                        "type": "StatementGroup",
	                        "operator": "and",
	                        "statements": [
	                            {
	                                "type": "BinaryStatement",
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operations": [
	                                        {
	                                            "type": "ExtractDetectionProperty",
	                                            "property_name": "class_name"
	                                        }
	                                    ]
	                                },
	                                "comparator": {
	                                    "type": "in (Sequence)"
	                                },
	                                "right_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "classes"
	                                }
	                            },
	                            {
	                                "type": "BinaryStatement",
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operations": [
	                                        {
	                                            "type": "ExtractDetectionProperty",
	                                            "property_name": "size"
	                                        }
	                                    ]
	                                },
	                                "comparator": {
	                                    "type": "(Number) >="
	                                },
	                                "right_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "image",
	                                    "operations": [
	                                        {
	                                            "type": "ExtractImageProperty",
	                                            "property_name": "size"
	                                        },
	                                        {
	                                            "type": "Multiply",
	                                            "other": 0.02
	                                        }
	                                    ]
	                                }
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {
	                "image": "$inputs.image",
	                "classes": "$inputs.classes"
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.filtering.*"
	        }
	    ]
	}
    ```

## Instance Segmentation results with background subtracted

This example showcases how to extract all instances detected by instance segmentation model
as separate crops without background.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSzJLNnpRNG51aUh0UlpRRk5mZGIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjB9.2pqePqevc7use2IbAd7dXa2Y1MbQK_y7TXZwC-rM8Uk?showGraph=true" loading="lazy" title="Roboflow Workflow for segmentation plus masked crop" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id",
	            "default_value": "yolov8n-seg-640"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	            "name": "segmentation",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "cropping",
	            "image": "$inputs.image",
	            "predictions": "$steps.segmentation.predictions",
	            "mask_opacity": 1.0
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "crops",
	            "selector": "$steps.cropping.crops"
	        },
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.segmentation.predictions"
	        }
	    ]
	}
    ```

## Workflow with detections sorting

This workflow presents how to use Detections Transformation block that is going to 
align predictions from object detection model such that results are sorted 
ascending regarding confidence.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoicDE3T1ZiOThTMEl4bXU2ekdXc3UiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjF9.qjPwl2aLAxJK7kZxXVOMlE7BPCjyKw7x3Yei9wNCHGI?showGraph=true" loading="lazy" title="Roboflow Workflow for detections sorting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.75
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence": "$inputs.confidence"
	        },
	        {
	            "type": "DetectionsTransformation",
	            "name": "sorting",
	            "predictions": "$steps.detection.predictions",
	            "operations": [
	                {
	                    "type": "SortDetections",
	                    "mode": "confidence",
	                    "ascending": true
	                }
	            ],
	            "operations_parameters": {
	                "image": "$inputs.image",
	                "classes": "$inputs.classes"
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.sorting.*"
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