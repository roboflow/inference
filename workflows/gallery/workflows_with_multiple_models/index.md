# Workflows with multiple models

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow detection model followed by classifier

This example showcases how to stack models on top of each other - in this particular
case, we detect objects using object detection models, requesting only "dogs" bounding boxes
in the output of prediction.

Based on the model predictions, we take each bounding box with dog and apply dynamic cropping
to be able to run classification model for each and every instance of dog separately.
Please note that for each inserted image we will have nested batch of crops (with size
dynamically determined in runtime, based on first model predictions) and for each crop
we apply secondary model.

Secondary model is supposed to make prediction from dogs breed classifier model
to assign detailed class for each dog instance.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibng0M2tDUmdUYkg0d0ZGaDZ2NUYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMTl9.-m5qHYgAVPCmbncBXaPfrm_vnvpXi_07VI1LbX08Low?showGraph=true" loading="lazy" title="Roboflow Workflow for detection plus classification" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.breds_classification.predictions"
	        }
	    ]
	}
    ```

## Workflow with classifier providing detailed labels for detected objects

This example illustrates how helpful Workflows could be when you have generic object detection model 
(capable of detecting common classes - like dogs) and specific classifier (capable of providing granular 
predictions for narrow high-level classes of objects - like dogs breed classifier). Having list
of classifier predictions for each detected dog is not handy way of dealing with output - 
as you kind of loose the information about location of specific dog. To avoid this problem, you
may want to replace class labels of original bounding boxes (from the first model localising dogs) with
classes predicted by classifier.

In this example, we use Detections Classes Replacement block which is also interesting from the 
perspective of difference of its inputs dimensionality levels. `object_detection_predictions` input
has level 1 (there is one prediction with bboxes for each input image) and `classification_predictions`
has level 2 (there are bunch of classification results for each input image). The block combines that
two inputs and produces result at dimensionality level 1 - exactly the same as predictions from 
object detection model.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiNGpkaHZvbUNNM1g0S3pkQzNrMVkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjF9.sVrm_gx_Wgms7ugyk8CSum7X-9TErjUTOPMQjIhD46M?showGraph=true" loading="lazy" title="Roboflow Workflow for detections classes replacement" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "general_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "class_filter": [
	                "dog"
	            ]
	        },
	        {
	            "type": "Crop",
	            "name": "cropping",
	            "image": "$inputs.image",
	            "predictions": "$steps.general_detection.predictions"
	        },
	        {
	            "type": "ClassificationModel",
	            "name": "breds_classification",
	            "image": "$steps.cropping.crops",
	            "model_id": "dog-breed-xpaq6/1",
	            "confidence": 0.09
	        },
	        {
	            "type": "DetectionsClassesReplacement",
	            "name": "classes_replacement",
	            "object_detection_predictions": "$steps.general_detection.predictions",
	            "classification_predictions": "$steps.breds_classification.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "original_predictions",
	            "selector": "$steps.general_detection.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "predictions_with_replaced_classes",
	            "selector": "$steps.classes_replacement.predictions"
	        }
	    ]
	}
    ```

## Workflow presenting models ensemble

This workflow presents how to combine predictions from multiple models running against the same 
input image with the block called Detections Consensus. 

First, we run two object detections models steps and we combine their predictions. Fusion may be 
performed in different scenarios based on Detections Consensus step configuration:

- you may combine predictions from models detecting different objects and then require only single 
model vote to add predicted bounding box to the output prediction

- you may combine predictions from models detecting the same objects and expect multiple positive 
votes to accept bounding box to the output prediction - this way you may improve the quality of 
predictions

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiR2o4aEpIYUtCQU9MOHp0Y29MQ0EiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjJ9.Kz_bB-CaTmMTfszuG20078UIrjDYGnC9dXSmSg83UyE?showGraph=true" loading="lazy" title="Roboflow Workflow for detections consensus" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "default_value": "yolov8n-640"
	        }
	    ],
	    "steps": [
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection_1",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence": 0.3
	        },
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection_2",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence": 0.83
	        },
	        {
	            "type": "DetectionsConsensus",
	            "name": "consensus",
	            "predictions_batches": [
	                "$steps.detection_1.predictions",
	                "$steps.detection_2.predictions"
	            ],
	            "required_votes": 2,
	            "required_objects": {
	                "person": 2
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.consensus.*"
	        }
	    ]
	}
    ```

## Comparison of detection models predictions

This example showcases how to compare predictions from two different models using Workflows and 
Model Comparison Visualization block.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZnUzUDBld1BCa1B2VDgwWE5TRksiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjN9.GCUFAzbhNJZwuOp8brjEzCEjXYpiLyACHcoawKjZ8jI?showGraph=true" loading="lazy" title="Roboflow Workflow for two detection models comparison" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "model_1",
	            "default_value": "yolov8n-640"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_2",
	            "default_value": "yolov8n-1280"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_1"
	        },
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_2"
	        },
	        {
	            "type": "roboflow_core/model_comparison_visualization@v1",
	            "name": "model_comparison_visualization",
	            "image": "$inputs.image",
	            "predictions_a": "$steps.model_1.predictions",
	            "predictions_b": "$steps.model.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_1_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "model_2_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model_1.predictions"
	        },
	        {
	            "type": "JsonField",
	            "name": "visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.model_comparison_visualization.image"
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