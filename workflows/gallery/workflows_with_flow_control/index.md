# Workflows with flow control

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with if statement applied on nested batches

In this test scenario we verify if we can successfully apply conditional
branching when data dimensionality increases.
We first make detections on input images and perform crop increasing
dimensionality to 2. Then we make another detections on cropped images
and check if inside crop we only see one instance of class dog (very naive
way of making sure that bboxes contain only single objects).
Only if that condition is true, we run classification model - to
classify dog breed.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVVRtZDhBWEhtNXN1OUhDYzZhb1kiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjZ9.UY-_8fCKfTjVQQsv35Po2cExUgb2IgUAF-kyOa4URIQ?showGraph=true" loading="lazy" title="Roboflow Workflow for flow control nested batches" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "ObjectDetectionModel",
	            "name": "first_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "DetectionsTransformation",
	            "name": "enlarging_boxes",
	            "predictions": "$steps.first_detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsOffset",
	                    "offset_x": 50,
	                    "offset_y": 50
	                }
	            ]
	        },
	        {
	            "type": "Crop",
	            "name": "first_crop",
	            "image": "$inputs.image",
	            "predictions": "$steps.enlarging_boxes.predictions"
	        },
	        {
	            "type": "ObjectDetectionModel",
	            "name": "second_detection",
	            "image": "$steps.first_crop.crops",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "dog"
	            ]
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
	                            "operand_name": "prediction",
	                            "operations": [
	                                {
	                                    "type": "SequenceLength"
	                                }
	                            ]
	                        },
	                        "comparator": {
	                            "type": "(Number) =="
	                        },
	                        "right_operand": {
	                            "type": "StaticOperand",
	                            "value": 1
	                        }
	                    }
	                ]
	            },
	            "evaluation_parameters": {
	                "prediction": "$steps.second_detection.predictions"
	            },
	            "next_steps": [
	                "$steps.classification"
	            ]
	        },
	        {
	            "type": "ClassificationModel",
	            "name": "classification",
	            "image": "$steps.first_crop.crops",
	            "model_id": "dog-breed-xpaq6/1"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "dog_classification",
	            "selector": "$steps.classification.predictions"
	        }
	    ]
	}
    ```

## Workflow with if statement applied on non batch-oriented input

In this test scenario we show that we can use non-batch oriented conditioning (ContinueIf block).

If statement is effectively applied on input parameter that would determine path of execution for
all data passed in `image` input. When the value matches expectation - all dependent steps
will be executed, otherwise only the independent ones.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoieHh1dWtIWThaNnFXYWExQ0R2OEoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjd9.foZn68KMMeXF3MTIKXG3X0B1j-sheWc-6HmA923jsPY?showGraph=true" loading="lazy" title="Roboflow Workflow for flow control on parameter" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "ObjectDetectionModel",
	            "name": "first_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "DetectionsTransformation",
	            "name": "enlarging_boxes",
	            "predictions": "$steps.first_detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsOffset",
	                    "offset_x": 50,
	                    "offset_y": 50
	                }
	            ]
	        },
	        {
	            "type": "Crop",
	            "name": "first_crop",
	            "image": "$inputs.image",
	            "predictions": "$steps.enlarging_boxes.predictions"
	        },
	        {
	            "type": "ObjectDetectionModel",
	            "name": "second_detection",
	            "image": "$steps.first_crop.crops",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "dog"
	            ]
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
	                            "operand_name": "prediction",
	                            "operations": [
	                                {
	                                    "type": "SequenceLength"
	                                }
	                            ]
	                        },
	                        "comparator": {
	                            "type": "(Number) =="
	                        },
	                        "right_operand": {
	                            "type": "StaticOperand",
	                            "value": 1
	                        }
	                    }
	                ]
	            },
	            "evaluation_parameters": {
	                "prediction": "$steps.second_detection.predictions"
	            },
	            "next_steps": [
	                "$steps.classification"
	            ]
	        },
	        {
	            "type": "ClassificationModel",
	            "name": "classification",
	            "image": "$steps.first_crop.crops",
	            "model_id": "dog-breed-xpaq6/1"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "dog_classification",
	            "selector": "$steps.classification.predictions"
	        }
	    ]
	}
    ```

## Workflow with continue_if block using stop_delay

In this test scenario we verify the stop_delay functionality of the continue_if block.
The stop_delay parameter allows the conditional branch to continue executing for a
specified duration after the condition becomes false, enabling graceful degradation
and delayed termination scenarios.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidm11OHZxWWxQbW1JZXpxWW5WZU8iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMjl9.PjIG5Jb166y2W7azmo8gZa03zaeQhBf4r_Ze4rr9dFU?showGraph=true" loading="lazy" title="Roboflow Workflow for continue if stop delay" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "condition_value",
	            "default_value": 1
	        }
	    ],
	    "steps": [
	        {
	            "type": "ContinueIf",
	            "name": "continue_if_with_delay",
	            "condition_statement": {
	                "type": "StatementGroup",
	                "statements": [
	                    {
	                        "type": "BinaryStatement",
	                        "left_operand": {
	                            "type": "DynamicOperand",
	                            "operand_name": "condition"
	                        },
	                        "comparator": {
	                            "type": "(Number) =="
	                        },
	                        "right_operand": {
	                            "type": "StaticOperand",
	                            "value": 1
	                        }
	                    }
	                ]
	            },
	            "next_steps": [
	                "$steps.dependent_model"
	            ],
	            "evaluation_parameters": {
	                "condition": "$inputs.condition_value"
	            },
	            "stop_delay": 1.0
	        },
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "dependent_model",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.dependent_model.predictions"
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