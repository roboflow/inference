# Advanced inference techniques

Below you can find example workflows you can use as inspiration to build your apps.

## SAHI in workflows - object detection

This example illustrates usage of [SAHI](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/) 
technique in workflows.

Workflows implementation requires three blocks:

- Image Slicer - which runs a sliding window over image and for each image prepares batch of crops 

- detection model block (in our scenario Roboflow Object Detection model) - which is responsible 
for making predictions on each crop

- Detections stitch - which combines partial predictions for each slice of the image into a single prediction

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQTZ1TFpmeHQyYm90elV1ZE5Id24iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzV9.Uf4X4HSDBYHLudg_AyJiyfE89zC-akax_CcrU8Dj-Yg?showGraph=true" loading="lazy" title="Roboflow Workflow for sahi detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "overlap_filtering_strategy"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/image_slicer@v1",
	            "name": "image_slicer",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$steps.image_slicer.slices",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/detections_stitch@v1",
	            "name": "stitch",
	            "reference_image": "$inputs.image",
	            "predictions": "$steps.detection.predictions",
	            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy"
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "bbox_visualiser",
	            "predictions": "$steps.stitch.predictions",
	            "image": "$inputs.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.stitch.predictions",
	            "coordinates_system": "own"
	        },
	        {
	            "type": "JsonField",
	            "name": "visualisation",
	            "selector": "$steps.bbox_visualiser.image"
	        }
	    ]
	}
    ```

## SAHI in workflows - instance segmentation

This example illustrates usage of [SAHI](https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/) 
technique in workflows.

Workflows implementation requires three blocks:

- Image Slicer - which runs a sliding window over image and for each image prepares batch of crops 

- detection model block (in our scenario Roboflow Instance Segmentation model) - which is responsible 
for making predictions on each crop

- Detections stitch - which combines partial predictions for each slice of the image into a single prediction

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidEJ5ZFo1OXBUbWpYcERpOHZqQWQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzZ9.FiwQI25GF3GOhSai8jkHyjl0MHUbIvLT2TM6koqAgPs?showGraph=true" loading="lazy" title="Roboflow Workflow for sahi segmentation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "overlap_filtering_strategy"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/image_slicer@v1",
	            "name": "image_slicer",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$steps.image_slicer.slices",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/detections_stitch@v1",
	            "name": "stitch",
	            "reference_image": "$inputs.image",
	            "predictions": "$steps.detection.predictions",
	            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy"
	        },
	        {
	            "type": "roboflow_core/bounding_box_visualization@v1",
	            "name": "bbox_visualiser",
	            "predictions": "$steps.stitch.predictions",
	            "image": "$inputs.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.stitch.predictions",
	            "coordinates_system": "own"
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