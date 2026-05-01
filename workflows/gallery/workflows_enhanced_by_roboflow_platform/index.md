# Workflows enhanced by Roboflow Platform

Below you can find example workflows you can use as inspiration to build your apps.

## Data Collection for Active Learning

This example showcases how to stack models on top of each other - in this particular
case, we detect objects using object detection models, requesting only "dogs" bounding boxes
in the output of prediction. Additionally, we register cropped images in Roboflow dataset.

Thanks to this setup, we are able to collect production data and continuously train better models over time.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiY2k5bkswb1I2SDBtaGlObHpyQmYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjR9.ftvf2oCI5QzFZPPRKS-CnrqRgyC-A_nLAU8nX8JtKUw?showGraph=true" loading="lazy" title="Roboflow Workflow for data collection active learning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "data_percentage",
	            "default_value": 50.0
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "persist_predictions",
	            "default_value": true
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "tag",
	            "default_value": "my_tag"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "disable_sink",
	            "default_value": false
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "fire_and_forget",
	            "default_value": true
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "labeling_batch_prefix",
	            "default_value": "some"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "general_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "dog"
	            ]
	        },
	        {
	            "type": "roboflow_core/dynamic_crop@v1",
	            "name": "cropping",
	            "image": "$inputs.image",
	            "predictions": "$steps.general_detection.predictions"
	        },
	        {
	            "type": "roboflow_core/roboflow_classification_model@v3",
	            "name": "breds_classification",
	            "image": "$steps.cropping.crops",
	            "model_id": "dog-breed-xpaq6/1",
	            "confidence_mode": "custom",
	            "custom_confidence": 0.09
	        },
	        {
	            "type": "roboflow_core/roboflow_dataset_upload@v2",
	            "name": "data_collection",
	            "images": "$steps.cropping.crops",
	            "predictions": "$steps.breds_classification.predictions",
	            "target_project": "my_project",
	            "usage_quota_name": "my_quota",
	            "data_percentage": "$inputs.data_percentage",
	            "persist_predictions": "$inputs.persist_predictions",
	            "minutely_usage_limit": 10,
	            "hourly_usage_limit": 100,
	            "daily_usage_limit": 1000,
	            "max_image_size": [
	                100,
	                200
	            ],
	            "compression_level": 85,
	            "registration_tags": [
	                "a",
	                "b",
	                "$inputs.tag"
	            ],
	            "disable_sink": "$inputs.disable_sink",
	            "fire_and_forget": "$inputs.fire_and_forget",
	            "labeling_batch_prefix": "$inputs.labeling_batch_prefix",
	            "labeling_batches_recreation_frequency": "never"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.breds_classification.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "registration_message",
	            "selector": "$steps.data_collection.message"
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