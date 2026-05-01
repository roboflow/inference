# Basic Workflows

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with bounding rect

This is the basic workflow that only contains a single object detection model and bounding rectangle extraction.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoid3FNeUNhUjlMT2c5RE9XNE11QloiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzl9.Egwi5eng1BXwXwP1KJMtH9Dq_b2VOrFkEj-fMzhgbJA?showGraph=true" loading="lazy" title="Roboflow Workflow for fit bounding rectangle" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "InstanceSegmentationModel",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-seg-640"
	        },
	        {
	            "type": "roboflow_core/bounding_rect@v1",
	            "name": "bounding_rect",
	            "predictions": "$steps.detection.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.bounding_rect.detections_with_rect"
	        }
	    ]
	}
    ```

## Workflow with Embeddings

This Workflow shows how to use an embedding model to compare the
similarity of two images with each other.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSXN3cXBRd2VVVGQybnJLVnZzbHYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDF9.B1lDcxu548JYX-it6SIa61qdBFqsKuphNC-bgOOVf3o?showGraph=true" loading="lazy" title="Roboflow Workflow for clip" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image_1"
	        },
	        {
	            "type": "InferenceImage",
	            "name": "image_2"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/clip@v1",
	            "name": "embedding_1",
	            "data": "$inputs.image_1",
	            "version": "RN50"
	        },
	        {
	            "type": "roboflow_core/clip@v1",
	            "name": "embedding_2",
	            "data": "$inputs.image_2",
	            "version": "RN50"
	        },
	        {
	            "type": "roboflow_core/cosine_similarity@v1",
	            "name": "cosine_similarity",
	            "embedding_1": "$steps.embedding_1.embedding",
	            "embedding_2": "$steps.embedding_2.embedding"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "similarity",
	            "coordinates_system": "own",
	            "selector": "$steps.cosine_similarity.similarity"
	        },
	        {
	            "type": "JsonField",
	            "name": "image_embeddings",
	            "coordinates_system": "own",
	            "selector": "$steps.embedding_1.embedding"
	        }
	    ]
	}
    ```

## Workflow with CLIP Comparison

This is the basic workflow that only contains a single CLIP Comparison block. 

Please take a look at how batch-oriented WorkflowImage data is plugged to 
detection step via input selector (`$inputs.image`) and how non-batch parameters 
(reference set of texts that the each image in batch will be compared to)
is dynamically specified - via `$inputs.reference` selector.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSXN3cXBRd2VVVGQybnJLVnZzbHYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDJ9.BjoGqksUGyGGZK_d07ZxIPkfINwTryCCFieS13gQTV0?showGraph=true" loading="lazy" title="Roboflow Workflow for clip" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "reference"
	        }
	    ],
	    "steps": [
	        {
	            "type": "ClipComparison",
	            "name": "comparison",
	            "images": "$inputs.image",
	            "texts": "$inputs.reference"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "similarity",
	            "selector": "$steps.comparison.similarity"
	        }
	    ]
	}
    ```

## Workflow with detections merge

This workflow demonstrates how to merge multiple object detections into a single bounding box.
This is useful when you want to:
- Combine overlapping detections of the same object
- Create a single region that contains multiple detected objects
- Simplify multiple detections into one larger detection

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVkMxOHRWN2h0dENXQlhhRnpzTkYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDN9.tIxmrvN6mMX6ahyRqkh4UAyv79s1cW7IGpKzNoRQY8E?showGraph=true" loading="lazy" title="Roboflow Workflow for merge detections" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/detections_merge@v1",
	            "name": "detections_merge",
	            "predictions": "$steps.detection.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.detections_merge.predictions"
	        }
	    ]
	}
    ```

## Workflow with static crop and object detection model

This is the basic workflow that contains single transformation (static crop)
followed by object detection model. This example may be inspiration for anyone
who would like to run specific model only on specific part of the image.
The Region of Interest does not necessarily have to be defined statically - 
please note that coordinates of static crops are referred via input selectors, 
which means that each time you run the workflow (for instance in each different
physical location, where RoI for static crop is location-dependent) you may 
provide different RoI coordinates.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidDJEZmJUUmZWQmpENFlxWlhaYTQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDR9.kj1xn_eL7HXDT6NuNUSwFIkHjSAZe-ZdV0AgBPaSg-I?showGraph=true" loading="lazy" title="Roboflow Workflow for static crop" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.7
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "x_center"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "y_center"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "width"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "height"
	        }
	    ],
	    "steps": [
	        {
	            "type": "AbsoluteStaticCrop",
	            "name": "crop",
	            "image": "$inputs.image",
	            "x_center": "$inputs.x_center",
	            "y_center": "$inputs.y_center",
	            "width": "$inputs.width",
	            "height": "$inputs.height"
	        },
	        {
	            "type": "RoboflowObjectDetectionModel",
	            "name": "detection",
	            "image": "$steps.crop.crops",
	            "model_id": "$inputs.model_id",
	            "confidence": "$inputs.confidence"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "crop",
	            "selector": "$steps.crop.crops"
	        },
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.detection.*"
	        },
	        {
	            "type": "JsonField",
	            "name": "result_in_own_coordinates",
	            "selector": "$steps.detection.*",
	            "coordinates_system": "own"
	        }
	    ]
	}
    ```

## Workflow writing data to OPC server

In this example data is written to OPC server.

In order to write to OPC this block is making use of [asyncua](https://github.com/FreeOpcUa/opcua-asyncio) package.

Writing to OPC enables workflows to expose insights extracted from camera to PLC controllers
allowing factory automation engineers to take advantage of machine vision when building PLC logic.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQnBpako0UjhqejVDbUJSTDlrUUIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDV9.wK4iPHzInhH3mqT3veoU7mpNuT92v6xcTa0yPNubDek?showGraph=true" loading="lazy" title="Roboflow Workflow for opc_writer" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceParameter",
	            "name": "opc_url"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_namespace"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_user_name"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_password"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_object_name"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_variable_name"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_value"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "opc_value_type"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_enterprise/opc_writer_sink@v1",
	            "name": "opc_writer",
	            "url": "$inputs.opc_url",
	            "namespace": "$inputs.opc_namespace",
	            "user_name": "$inputs.opc_user_name",
	            "password": "$inputs.opc_password",
	            "object_name": "$inputs.opc_object_name",
	            "variable_name": "$inputs.opc_variable_name",
	            "value": "$inputs.opc_value",
	            "value_type": "$inputs.opc_value_type",
	            "fire_and_forget": false
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "opc_writer_results",
	            "selector": "$steps.opc_writer.*"
	        }
	    ]
	}
    ```

## Workflow with single object detection model

This is the basic workflow that only contains a single object detection model.

Please take a look at how batch-oriented WorkflowImage data is plugged to
detection step via input selector (`$inputs.image`) and how non-batch parameters
are dynamically specified - via `$inputs.model_id` and `$inputs.confidence` selectors.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYUR2T0NscVN5QVVMYk1oQzJGNzYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwNDZ9.4bU1IqzIFttQhCkPfSD5fcwUzpnjQJQqgkRN9DcdsHg?showGraph=true" loading="lazy" title="Roboflow Workflow for basic object detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "confidence",
	            "default_value": 0.3
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "detection",
	            "image": "$inputs.image",
	            "model_id": "$inputs.model_id",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.detection.*"
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