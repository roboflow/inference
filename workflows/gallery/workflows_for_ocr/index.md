# Workflows for OCR

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with DocTR model

This example showcases quite sophisticated workflows usage scenario that assume the following:

- we have generic object detection model capable of recognising cars

- we have specialised object detection model trained to detect license plates in the images depicting **single car only**

- we have generic OCR model capable of recognising lines of texts from images

Our goal is to read license plates of every car we detect in the picture. We can achieve that goal with 
workflow from this example. In the definition we can see that generic object detection model is applied first, 
to make the job easier for the secondary (plates detection) model we enlarge bounding boxes, slightly 
offsetting its dimensions with Detections Offset block - later we apply cropping to be able to run
license plate detection for every detected car instance (increasing the depth of the batch). Once secondary model
runs and we have bounding boxes for license plates - we crop previously cropped cars images to extract plates.
Once this is done, plates crops are passed to OCR step which turns images of plates into text.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMlRqQ3FIakh3NkxqUUI5cmxvNEciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTB9.O384Q6rUzf0d5AdorN7nWlnAZZWgkSnkFafGK4aLyf8?showGraph=true" loading="lazy" title="Roboflow Workflow for detection plus ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	    "steps": [
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "car"
	            ]
	        },
	        {
	            "type": "DetectionOffset",
	            "name": "offset",
	            "predictions": "$steps.detection.predictions",
	            "image_metadata": "$steps.detection.image",
	            "prediction_type": "$steps.detection.prediction_type",
	            "offset_width": 10,
	            "offset_height": 10
	        },
	        {
	            "type": "Crop",
	            "name": "cars_crops",
	            "image": "$inputs.image",
	            "predictions": "$steps.offset.predictions"
	        },
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "plates_detection",
	            "image": "$steps.cars_crops.crops",
	            "model_id": "vehicle-registration-plates-trudk/2"
	        },
	        {
	            "type": "DetectionOffset",
	            "name": "plates_offset",
	            "predictions": "$steps.plates_detection.predictions",
	            "image_metadata": "$steps.plates_detection.image",
	            "prediction_type": "$steps.plates_detection.prediction_type",
	            "offset_width": 50,
	            "offset_height": 50
	        },
	        {
	            "type": "Crop",
	            "name": "plates_crops",
	            "image": "$steps.cars_crops.crops",
	            "predictions": "$steps.plates_offset.predictions"
	        },
	        {
	            "type": "OCRModel",
	            "name": "ocr",
	            "image": "$steps.plates_crops.crops"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "cars_crops",
	            "selector": "$steps.cars_crops.crops"
	        },
	        {
	            "type": "JsonField",
	            "name": "plates_crops",
	            "selector": "$steps.plates_crops.crops"
	        },
	        {
	            "type": "JsonField",
	            "name": "plates_ocr",
	            "selector": "$steps.ocr.result"
	        }
	    ]
	}
    ```

## Google Vision OCR

In this example, Google Vision OCR is used to extract text from input image.
Additionally, example presents how to combine structured output of Google API
with visualisation blocks.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMkxLMHdHZ2FWS0diaFAxU3QxVEIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTF9.FJkxxRwYiyDIPfKHoCVrqaG6RjaGrjwHSkMrUZiLb_s?showGraph=true" loading="lazy" title="Roboflow Workflow for google vision ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "api_key"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_vision_ocr@v1",
	            "name": "google_vision_ocr",
	            "image": "$inputs.image",
	            "ocr_type": "text_detection",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "bounding_box_visualization",
	            "predictions": "$steps.google_vision_ocr.predictions",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/label_visualization@v1",
	            "name": "label_visualization",
	            "predictions": "$steps.google_vision_ocr.predictions",
	            "image": "$steps.bounding_box_visualization.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "extracted_text",
	            "selector": "$steps.google_vision_ocr.text"
	        },
	        {
	            "type": "JsonField",
	            "name": "text_detections",
	            "selector": "$steps.google_vision_ocr.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "text_visualised",
	            "selector": "$steps.label_visualization.image"
	        }
	    ]
	}
    ```

## Workflow with model detecting individual characters and text stitching

This workflow extracts and organizes text from an image using OCR. It begins by analyzing the image with detection 
model to detect individual characters or words and their positions. 

Then, it groups nearby text into lines based on a specified `tolerance` for spacing and arranges them in 
reading order (`left-to-right`). 

The final output is a JSON field containing the structured text in readable, logical order, accurately reflecting 
the layout of the original image.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidnpKT3FKUFd2bzNQRjFDeEk0aFYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTJ9.j6AKRy3NwEOnlpkMolpyf1GzchkGF8aRDJbXGkn84Hc?showGraph=true" loading="lazy" title="Roboflow Workflow for ocr detections stitch" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "ocr-oy9a7/1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "tolerance",
	            "default_value": 10
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "ocr_detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/stitch_ocr_detections@v2",
	            "name": "detections_stitch",
	            "predictions": "$steps.ocr_detection.predictions",
	            "stitching_algorithm": "tolerance",
	            "reading_direction": "left_to_right",
	            "tolerance": "$inputs.tolerance"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "ocr_text",
	            "selector": "$steps.detections_stitch.ocr_text"
	        }
	    ]
	}
    ```

## Workflow with model detecting individual characters and text stitching (tolerance algorithm)

This workflow extracts and organizes text from an image using OCR with the tolerance-based stitching algorithm.
It detects individual characters or words and their positions, then groups nearby text into lines based on a
specified pixel `tolerance` for spacing and arranges them in reading order (`left-to-right`).

The tolerance algorithm is best for consistent font sizes and well-aligned horizontal/vertical text.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYkhwZ3I0b09QMDMwOEM1UVM1cTIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTN9.PHRuq_UufsFSPq2rq0dZPZHdoRHeBU8u-QH1wEMRwYo?showGraph=true" loading="lazy" title="Roboflow Workflow for ocr detections stitch v2 tolerance" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "ocr-oy9a7/1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "tolerance",
	            "default_value": 10
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "ocr_detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/stitch_ocr_detections@v2",
	            "name": "detections_stitch",
	            "predictions": "$steps.ocr_detection.predictions",
	            "stitching_algorithm": "tolerance",
	            "reading_direction": "left_to_right",
	            "tolerance": "$inputs.tolerance"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "ocr_text",
	            "selector": "$steps.detections_stitch.ocr_text"
	        }
	    ]
	}
    ```

## Workflow with model detecting individual characters and text stitching (Otsu algorithm)

This workflow extracts and organizes text from an image using OCR with the Otsu thresholding algorithm.
It detects individual characters and uses Otsu's method on normalized gap distances to automatically find
the optimal threshold separating character gaps from word gaps.

The Otsu algorithm is resolution-invariant and works well with variable font sizes and automatic word
boundary detection. It detects bimodal distributions to distinguish single words from multi-word text.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiRFV1MG16cHlHcGQ0U1JFMVBRc2UiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTV9.ebWkTFoIO9ATAF7OR522nIS_Qv9t5g0475CpQ_kxn-0?showGraph=true" loading="lazy" title="Roboflow Workflow for ocr detections stitch v2 otsu" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "ocr-oy9a7/1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "threshold_multiplier",
	            "default_value": 1.0
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "ocr_detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/stitch_ocr_detections@v2",
	            "name": "detections_stitch",
	            "predictions": "$steps.ocr_detection.predictions",
	            "stitching_algorithm": "otsu",
	            "reading_direction": "left_to_right",
	            "otsu_threshold_multiplier": "$inputs.threshold_multiplier"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "ocr_text",
	            "selector": "$steps.detections_stitch.ocr_text"
	        }
	    ]
	}
    ```

## Workflow with model detecting individual characters and text stitching (collimate algorithm)

This workflow extracts and organizes text from an image using OCR with the collimate algorithm.
It detects individual characters and uses greedy parent-child traversal to follow text flow,
building lines through traversal rather than bucketing.

The collimate algorithm is best for skewed, curved, or non-axis-aligned text where traditional
bucket-based line grouping may fail.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiejhLdzBXZ1pZenRGTmtGY2k1V2wiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMTZ9.nD38sGBooimeWJrwvnz6LFJV8FgF3BHhDPrF4SNRBsg?showGraph=true" loading="lazy" title="Roboflow Workflow for ocr detections stitch v2 collimate" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "ocr-oy9a7/1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.4
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "collimate_tolerance",
	            "default_value": 10
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "ocr_detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/stitch_ocr_detections@v2",
	            "name": "detections_stitch",
	            "predictions": "$steps.ocr_detection.predictions",
	            "stitching_algorithm": "collimate",
	            "reading_direction": "left_to_right",
	            "collimate_tolerance": "$inputs.collimate_tolerance"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "ocr_text",
	            "selector": "$steps.detections_stitch.ocr_text"
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