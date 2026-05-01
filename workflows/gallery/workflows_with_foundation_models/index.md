# Workflows with foundation models

Below you can find example workflows you can use as inspiration to build your apps.

## Gaze Detection Workflow

This workflow uses L2CS-Net to detect faces and estimate their gaze direction.
The output includes:
- Face detections with facial landmarks
- Gaze angles (yaw and pitch) in degrees
- Visualization of facial landmarks

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoieG9HalBhVzBFTndibzVqVnhHQmQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzB9.mNHcJJqM3ZspSzX2zu5I3K4NZ-MlMCyuDlqQMtn0qfc?showGraph=true" loading="lazy" title="Roboflow Workflow for gaze detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "do_run_face_detection",
	            "default_value": true
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/gaze@v1",
	            "name": "gaze",
	            "images": "$inputs.image",
	            "do_run_face_detection": "$inputs.do_run_face_detection"
	        },
	        {
	            "type": "roboflow_core/keypoint_visualization@v1",
	            "name": "visualization",
	            "predictions": "$steps.gaze.face_predictions",
	            "image": "$inputs.image",
	            "annotator_type": "vertex",
	            "color": "#A351FB",
	            "text_color": "black",
	            "text_scale": 0.5,
	            "text_thickness": 1,
	            "text_padding": 10,
	            "thickness": 2,
	            "radius": 10
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "face_predictions",
	            "selector": "$steps.gaze.face_predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "yaw_degrees",
	            "selector": "$steps.gaze.yaw_degrees"
	        },
	        {
	            "type": "JsonField",
	            "name": "pitch_degrees",
	            "selector": "$steps.gaze.pitch_degrees"
	        },
	        {
	            "type": "JsonField",
	            "name": "visualization",
	            "selector": "$steps.visualization.image"
	        }
	    ]
	}
    ```

## Workflow with Segment Anything 2 model

Meta AI introduced very capable segmentation model called [SAM 2](https://ai.meta.com/sam2/) which
has capabilities of producing segmentation masks for instances of objects. 

**EXAMPLE REQUIRES DEDICATED DEPLOYMENT** and will not run in preview!

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibEUwWDhUMDBkQ25sSGFBWm5kdVoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzF9.h6hk5kunW7zYP46vIs8zIW2IgQkrUPEizzGLVld5jq8?showGraph=true" loading="lazy" title="Roboflow Workflow for simple sam2" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "mask_threshold",
	            "default_value": 0.0
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "version",
	            "default_value": "hiera_tiny"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/segment_anything@v1",
	            "name": "segment_anything",
	            "images": "$inputs.image",
	            "threshold": "$inputs.mask_threshold",
	            "version": "$inputs.version"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.segment_anything.predictions"
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