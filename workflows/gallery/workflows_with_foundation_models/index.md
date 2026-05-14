# Workflows with foundation models

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with Segment Anything 2 model

Meta AI introduced very capable segmentation model called [SAM 2](https://ai.meta.com/sam2/) which
has capabilities of producing segmentation masks for instances of objects. 

**EXAMPLE REQUIRES DEDICATED DEPLOYMENT** and will not run in preview!

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibEUwWDhUMDBkQ25sSGFBWm5kdVoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzg3NTU1Nzd9.B8RtwIZrZwArof5yQKJK38qevGIy13NZ8wnsvXpfaCs?showGraph=true" loading="lazy" title="Roboflow Workflow for simple sam2" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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