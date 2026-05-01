# Integration with external apps

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow sending notification to Slack

This Workflow illustrates how to send notification to Slack.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZk9FOEkxUzM0V0Z4dDFmaTVHS28iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzd9.-YGKsGUFLkqY4GyCaOBcWJFW_47Og73350kEQOGv0dM?showGraph=true" loading="lazy" title="Roboflow Workflow for basic slack notification" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.4.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id",
	            "default_value": "yolov8n-640"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "channel_id"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "slack_token"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/slack_notification@v1",
	            "name": "notification",
	            "slack_token": "$inputs.slack_token",
	            "message": "Detected {{ '{{' }} $parameters.predictions {{ '}}' }} objects",
	            "channel": "$inputs.channel_id",
	            "message_parameters": {
	                "predictions": "$steps.detection.predictions"
	            },
	            "message_parameters_operations": {
	                "predictions": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            },
	            "fire_and_forget": false,
	            "cooldown_seconds": 0,
	            "cooldown_session_key": "some-unique-key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "status",
	            "selector": "$steps.notification.error_status"
	        }
	    ]
	}
    ```

## Workflow sending notification with attachments to Slack

This Workflow illustrates how to send notification with attachments to Slack.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSG16cHl4dnlpQVBjWmhwRU1EU0IiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzh9.m2ujd0Fu0kmHkA0G8M2NCvkOcJLGr5RHtvPWcZPGc6U?showGraph=true" loading="lazy" title="Roboflow Workflow for advanced slack notification" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.4.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id",
	            "default_value": "yolov8n-640"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "channel_id"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "slack_token"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "image_serialization",
	            "data": "$inputs.image",
	            "operations": [
	                {
	                    "type": "ConvertImageToJPEG"
	                }
	            ]
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "predictions_serialization",
	            "data": "$steps.detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsToDictionary"
	                },
	                {
	                    "type": "ConvertDictionaryToJSON"
	                }
	            ]
	        },
	        {
	            "type": "roboflow_core/slack_notification@v1",
	            "name": "notification",
	            "slack_token": "$inputs.slack_token",
	            "message": "Detected {{ '{{' }} $parameters.predictions {{ '}}' }} objects",
	            "channel": "$inputs.channel_id",
	            "message_parameters": {
	                "predictions": "$steps.detection.predictions"
	            },
	            "message_parameters_operations": {
	                "predictions": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            },
	            "attachments": {
	                "image.jpg": "$steps.image_serialization.output",
	                "prediction.json": "$steps.predictions_serialization.output"
	            },
	            "fire_and_forget": false,
	            "cooldown_seconds": 0,
	            "cooldown_session_key": "some-unique-key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "status",
	            "selector": "$steps.notification.error_status"
	        }
	    ]
	}
    ```

## Workflow sending SMS notification with Twilio

This Workflow illustrates how to send SMS notification with Twilio.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiWVEwOHFjazM4UnVkNHllVXhnZUYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzl9.Z3JV7tWeStjyFSUTDPdLeig5HMAFVVMLZ9jN5fW3qog?showGraph=true" loading="lazy" title="Roboflow Workflow for basic twilio sms notification" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.4.0",
	    "inputs": [
	        {
	            "type": "WorkflowImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id",
	            "default_value": "yolov8n-640"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "account_sid"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "auth_token"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "sender_number"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "receiver_number"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/twilio_sms_notification@v1",
	            "name": "notification",
	            "twilio_account_sid": "$inputs.account_sid",
	            "twilio_auth_token": "$inputs.auth_token",
	            "message": "Detected {{ '{{' }} $parameters.predictions {{ '}}' }} objects",
	            "sender_number": "$inputs.sender_number",
	            "receiver_number": "$inputs.receiver_number",
	            "message_parameters": {
	                "predictions": "$steps.detection.predictions"
	            },
	            "message_parameters_operations": {
	                "predictions": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            },
	            "fire_and_forget": false,
	            "cooldown_seconds": 0,
	            "cooldown_session_key": "some-unique-key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "status",
	            "selector": "$steps.notification.error_status"
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