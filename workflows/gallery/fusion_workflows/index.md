# Fusion Workflows

Below you can find example workflows you can use as inspiration to build your apps.

## Detections Roll-Up with All Detection Types

Comprehensive workflow testing detections_rollup with object detection,
keypoint detection, and instance segmentation. Tests the ability to rollup
predictions from dynamic crops back to parent image coordinates.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiaVJDbVEwWjNadGRZNllBQkhDeGQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjJ9.NkeJEqeQdfHyNLEFAq3TxdhmUDXguylTYyQkPQJ7nCM?showGraph=true" loading="lazy" title="Roboflow Workflow for dimension rollup full" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/detection_offset@v1",
	            "name": "detection_offset",
	            "predictions": "$steps.model_1.predictions",
	            "offset_height": 300,
	            "offset_width": 300
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "dynamic_crop",
	            "images": "$inputs.image",
	            "predictions": "$steps.detection_offset.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "people-detection-o4rdr/7"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "obj_detection",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "people-detection-o4rdr/7",
	            "iou_threshold": 0.3
	        },
	        {
	            "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
	            "name": "keypoint_detection",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "yolov8n-pose-640"
	        },
	        {
	            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	            "name": "segmentation",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "yolov8n-seg-640"
	        },
	        {
	            "type": "roboflow_core/detections_filter@v1",
	            "name": "obj_filter",
	            "predictions": "$steps.obj_detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsFilter",
	                    "filter_operation": {
	                        "type": "StatementGroup",
	                        "operator": "and",
	                        "statements": [
	                            {
	                                "type": "BinaryStatement",
	                                "negate": false,
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "_",
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
	                                    "type": "StaticOperand",
	                                    "value": [
	                                        "person"
	                                    ]
	                                }
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {}
	        },
	        {
	            "type": "roboflow_core/detections_filter@v1",
	            "name": "key_filter",
	            "predictions": "$steps.keypoint_detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsFilter",
	                    "filter_operation": {
	                        "type": "StatementGroup",
	                        "operator": "and",
	                        "statements": [
	                            {
	                                "type": "BinaryStatement",
	                                "negate": false,
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "_",
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
	                                    "type": "StaticOperand",
	                                    "value": [
	                                        "person"
	                                    ]
	                                }
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {}
	        },
	        {
	            "type": "roboflow_core/detections_filter@v1",
	            "name": "seg_filter",
	            "predictions": "$steps.segmentation.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsFilter",
	                    "filter_operation": {
	                        "type": "StatementGroup",
	                        "operator": "and",
	                        "statements": [
	                            {
	                                "type": "BinaryStatement",
	                                "negate": false,
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "_",
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
	                                    "type": "StaticOperand",
	                                    "value": [
	                                        "person"
	                                    ]
	                                }
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {}
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "obj_collapse",
	            "data": "$steps.obj_filter.predictions"
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "key_collapse",
	            "data": "$steps.key_filter.predictions"
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "seg_collapse",
	            "data": "$steps.seg_filter.predictions"
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "obj_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.obj_collapse.output",
	            "overlap_threshold": 0,
	            "keypoint_merge_threshold": 0
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "key_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.key_collapse.output",
	            "overlap_threshold": 1,
	            "keypoint_merge_threshold": 0
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "seg_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.seg_collapse.output"
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "obj_bounding",
	            "image": "$inputs.image",
	            "predictions": "$steps.obj_rollup.rolled_up_detections"
	        },
	        {
	            "type": "roboflow_core/keypoint_visualization@v1",
	            "name": "keypoint_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.key_rollup.rolled_up_detections"
	        },
	        {
	            "type": "roboflow_core/mask_visualization@v1",
	            "name": "mask_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.seg_rollup.rolled_up_detections"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "obj_rollup_detections",
	            "selector": "$steps.obj_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "key_rollup_detections",
	            "selector": "$steps.key_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "seg_rollup_detections",
	            "selector": "$steps.seg_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "obj_bounding_image",
	            "coordinates_system": "own",
	            "selector": "$steps.obj_bounding.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "keypoint_visualization_image",
	            "coordinates_system": "own",
	            "selector": "$steps.keypoint_visualization.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "mask_visualization_image",
	            "coordinates_system": "own",
	            "selector": "$steps.mask_visualization.image"
	        }
	    ]
	}
    ```

## Dimension Rollup with Object Detection

Test detections_rollup with object detection predictions. Demonstrates how to
rollup object detection predictions from dynamic crops back to parent
image coordinates with configurable confidence merging strategies and
overlap thresholds.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiT1RNZ0tmbE0ycUhOMldDZmFHZXciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjN9.BgPUA1uCBpvoS-lRmr33XYaheeghQBt-3U6MRbTTlpM?showGraph=true" loading="lazy" title="Roboflow Workflow for dimension rollup object detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/detection_offset@v1",
	            "name": "detection_offset",
	            "predictions": "$steps.model.predictions",
	            "offset_height": 300,
	            "offset_width": 300
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "dynamic_crop",
	            "images": "$inputs.image",
	            "predictions": "$steps.detection_offset.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "people-detection-o4rdr/7"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "obj_detection",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "people-detection-o4rdr/7",
	            "iou_threshold": 0.3
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "obj_collapse",
	            "data": "$steps.obj_detection.predictions"
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "obj_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.obj_collapse.output",
	            "confidence_strategy": "max",
	            "overlap_threshold": 0.0,
	            "keypoint_merge_threshold": 10
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "obj_bounding",
	            "image": "$inputs.image",
	            "predictions": "$steps.obj_rollup.rolled_up_detections"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "rolled_up_detections",
	            "selector": "$steps.obj_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "bounding_box_visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.obj_bounding.image"
	        }
	    ]
	}
    ```

## Dimension Rollup with Keypoint Detection

Test dimension_rollup with keypoint detection predictions. Demonstrates how to
rollup keypoint detections from dynamic crops back to parent image coordinates
with configurable keypoint merge thresholds.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiNWxaRG4zb0ZIRFA3Vk1RZlVwYzgiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjR9.-9Gd5e8MeVxI7D2RsrdLI0yLb31j14PVENETKhArnK8?showGraph=true" loading="lazy" title="Roboflow Workflow for dimension rollup keypoint detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/detection_offset@v1",
	            "name": "detection_offset",
	            "predictions": "$steps.model.predictions",
	            "offset_height": 300,
	            "offset_width": 300
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "dynamic_crop",
	            "images": "$inputs.image",
	            "predictions": "$steps.detection_offset.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "people-detection-o4rdr/7"
	        },
	        {
	            "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
	            "name": "keypoint_detection",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "yolov8n-pose-640"
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "key_collapse",
	            "data": "$steps.keypoint_detection.predictions"
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "key_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.key_collapse.output",
	            "keypoint_merge_threshold": 10
	        },
	        {
	            "type": "roboflow_core/keypoint_visualization@v1",
	            "name": "keypoint_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.key_rollup.rolled_up_detections"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "rolled_up_detections",
	            "selector": "$steps.key_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "keypoint_visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.keypoint_visualization.image"
	        }
	    ]
	}
    ```

## Dimension Rollup with Instance Segmentation

Test dimension_rollup with instance segmentation predictions. Demonstrates how to
rollup segmentation masks from dynamic crops back to parent image coordinates.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZDlITGRqbjZiZnVQajZMSElPVFoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjV9.ix0hG4Plx8qQbi7RRfVkgFYJDAi6Wroop4la605MGrA?showGraph=true" loading="lazy" title="Roboflow Workflow for dimension rollup segmentation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/detection_offset@v1",
	            "name": "detection_offset",
	            "predictions": "$steps.model.predictions",
	            "offset_height": 300,
	            "offset_width": 300
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "dynamic_crop",
	            "images": "$inputs.image",
	            "predictions": "$steps.detection_offset.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "people-detection-o4rdr/7"
	        },
	        {
	            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	            "name": "segmentation",
	            "images": "$steps.dynamic_crop.crops",
	            "model_id": "yolov8n-seg-640"
	        },
	        {
	            "type": "roboflow_core/dimension_collapse@v1",
	            "name": "seg_collapse",
	            "data": "$steps.segmentation.predictions"
	        },
	        {
	            "type": "roboflow_core/detections_list_rollup@v1",
	            "name": "seg_rollup",
	            "parent_detection": "$steps.detection_offset.predictions",
	            "child_detections": "$steps.seg_collapse.output"
	        },
	        {
	            "type": "roboflow_core/mask_visualization@v1",
	            "name": "mask_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.seg_rollup.rolled_up_detections"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "rolled_up_detections",
	            "selector": "$steps.seg_rollup.rolled_up_detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "mask_visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.mask_visualization.image"
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