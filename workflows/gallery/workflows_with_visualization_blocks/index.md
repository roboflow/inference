# Workflows with visualization blocks

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with multi-label classification label visualization

This workflow demonstrates how to visualize the predictions of a multi-label classification model. 
It is compatable with single-label and multi-label classification tasks. It is also 
compatible with supervision visualization fields like text position, color, scale, etc.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiaUdTYWVMZHhISEo5Sm9YRVhlcDkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDh9.c1oB7hJCy4ltrerMWmbVTecFqeB42tErLomqEUFcSR4?showGraph=true" loading="lazy" title="Roboflow Workflow for multi label classification visualization" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "deepfashion2-1000-items/1"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/classification_label_visualization@v1",
	            "name": "classification_label_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.model.predictions",
	            "text": "Class and Confidence",
	            "text_position": "CENTER"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        },
	        {
	            "type": "JsonField",
	            "name": "classification_label_visualization",
	            "selector": "$steps.classification_label_visualization.image"
	        }
	    ]
	}
    ```

## Workflow with single-label classification label visualization

This workflow demonstrates how to visualize the predictions of a single-label classification model. 
It is compatable with single-label and multi-label classification tasks. It is also 
compatible with supervision visualization fields like text position, color, scale, etc.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoicGFlMmE2bExZdEx1MGhYOEZlQVEiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDl9.lfUugjsP_qqb3j1AyCCZuE2ShlIA9uvF_ohCi1DoX4g?showGraph=true" loading="lazy" title="Roboflow Workflow for single label classification visualization" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "fruit-ee3k2/1"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_classification_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/classification_label_visualization@v1",
	            "name": "classification_label_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.model.predictions",
	            "num_classifications": "$inputs.num_classifications",
	            "text": "Class and Confidence",
	            "color_axis": "INDEX",
	            "color_palette": "ROBOFLOW",
	            "text_scale": 1,
	            "text_color": "BLACK",
	            "text_padding": 28
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "classification_label_visualization",
	            "selector": "$steps.classification_label_visualization.image"
	        }
	    ]
	}
    ```

## Predictions from different models visualised together

This workflow showcases how predictions from different models (even from nested 
batches created from input images) may be visualised together.

Our scenario covers:

- Detecting cars using YOLOv8 model

- Dynamically cropping input images to run secondary model (license plates detector) for each 
car instance

- Stitching together all predictions for licence plates into single prediction

- Fusing cars detections and license plates detections into single prediction

- Visualizing final predictions

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVXlHSk9lMVlxa09tckdodjNSVkIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNTB9.ziAmUZXJ-HbXDrzlgmbsP4BjJgM0OpaefUefsg8-T3M?showGraph=true" loading="lazy" title="Roboflow Workflow for visualisation blocks" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "car_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "car"
	            ]
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "cropping",
	            "image": "$inputs.image",
	            "predictions": "$steps.car_detection.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "plates_detection",
	            "image": "$steps.cropping.crops",
	            "model_id": "vehicle-registration-plates-trudk/2"
	        },
	        {
	            "type": "roboflow_core/detections_stitch@v1",
	            "name": "stitch",
	            "reference_image": "$inputs.image",
	            "predictions": "$steps.plates_detection.predictions",
	            "overlap_filtering_strategy": "nms"
	        },
	        {
	            "type": "DetectionsConsensus",
	            "name": "consensus",
	            "predictions_batches": [
	                "$steps.car_detection.predictions",
	                "$steps.stitch.predictions"
	            ],
	            "required_votes": 1
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "bbox_visualiser",
	            "predictions": "$steps.consensus.predictions",
	            "image": "$inputs.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.consensus.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "visualisation",
	            "selector": "$steps.bbox_visualiser.image"
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