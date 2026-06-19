# Workflows with Visual Language Models

Below you can find example workflows you can use as inspiration to build your apps.

## Prompting Anthropic Claude with arbitrary prompt

In this example, Anthropic Claude model is prompted with arbitrary text from user

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTzdkQ0k4V244ZW9uWHBzMHZUSDQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTB9.k7ZXnvHRt_tRpTUaK2SSKplC3rUTjr838AM-hSyOz5E?showGraph=true" loading="lazy" title="Roboflow Workflow for claude arbitrary prompt" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "unconstrained",
	            "prompt": "Give me dominant color of the image",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.claude.output"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as OCR model

In this example, Anthropic Claude model is used as OCR system. User just points task type and do not need to provide
any prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZ1IxMnpSWDlmQ1B2bEI5VXNHZ1UiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTF9.vtCPs96bNzFG3bGG2_CLYZMSTg2QxNKsypx5wV6IE5E?showGraph=true" loading="lazy" title="Roboflow Workflow for claude ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "ocr",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.claude.output"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as Visual Question Answering system

In this example, Anthropic Claude model is used as VQA system. User provides question via prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVW9wUGluWHB2ZGV3bzd2RE0xcUIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTJ9.UYt_-Kzn2L-yZgAqJUTbcc6nSaMMhC31eBap8kW7Fso?showGraph=true" loading="lazy" title="Roboflow Workflow for claude vqa" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "visual-question-answering",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.claude.output"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as Image Captioning system

In this example, Anthropic Claude model is used as Image Captioning system.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiOHVxT25jSFVEejdpS1FoZG5QcjkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTN9.ZNtm5AkaQXTMWYSBpL2ERPWanYmVABhLaPNUiHZKNqA?showGraph=true" loading="lazy" title="Roboflow Workflow for claude captioning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "caption",
	            "api_key": "$inputs.api_key",
	            "temperature": 1.0
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.claude.output"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as multi-class classifier

In this example, Anthropic Claude model is used as classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidUxyQWV2TllacG5VR0tZRXAzcFkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTR9.KbwpGx8lZCGQU3v5Y3APs5MRmr73TxTIdWA-gt2SeCg?showGraph=true" loading="lazy" title="Roboflow Workflow for claude multi class classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.claude.output",
	            "classes": "$steps.claude.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "claude_result",
	            "selector": "$steps.claude.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "top_class",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as multi-label classifier

In this example, Anthropic Claude model is used as multi-label classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiNGZvQWRhVE5qazgxUGVkNWhZUFIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTV9.6NO7TBVhEbhkz30EqqB_Cgqp5sMN3kBK1-aLlbOYgXM?showGraph=true" loading="lazy" title="Roboflow Workflow for claude multi label classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "multi-label-classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.claude.output",
	            "classes": "$steps.claude.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using Anthropic Claude to provide structured JSON

In this example, Anthropic Claude model is expected to provide structured output in JSON, which can later be
parsed by dedicated `roboflow_core/json_parser@v1` block which transforms string into dictionary 
and expose it's keys to other blocks for further processing. In this case, parsed output is
transformed using `roboflow_core/property_definition@v1` block.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMENIOU9PbXZMajRubjVCWGliTTkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTZ9.MR-_55mBKpg_iUTDMUQWLZIm6PZCJUNj-yzlxQ7ZPJ4?showGraph=true" loading="lazy" title="Roboflow Workflow for claude structured prompting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "structured-answering",
	            "output_structure": {
	                "dogs_count": "count of dogs instances in the image",
	                "cats_count": "count of cats instances in the image"
	            },
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/json_parser@v1",
	            "name": "parser",
	            "raw_json": "$steps.claude.output",
	            "expected_fields": [
	                "dogs_count",
	                "cats_count"
	            ]
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "property_definition",
	            "operations": [
	                {
	                    "type": "ToString"
	                }
	            ],
	            "data": "$steps.parser.dogs_count"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.property_definition.output"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as object-detection model

In this example, Anthropic Claude model is expected to provide output, which can later be
parsed by dedicated `roboflow_core/vlm_as_detector@v1` block which transforms string into `sv.Detections`, 
which can later be used by other blocks processing object-detection predictions.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiRlJ0a3BHWjlIQmppYm9HWWxsZHoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTd9.006ShHtyoE1CFcN4plhv7od4C6kWlT2EMDQurZHW764?showGraph=true" loading="lazy" title="Roboflow Workflow for claude object detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$inputs.image",
	            "task_type": "object-detection",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_detector@v2",
	            "name": "parser",
	            "vlm_output": "$steps.claude.output",
	            "image": "$inputs.image",
	            "classes": "$steps.claude.classes",
	            "model_type": "anthropic-claude",
	            "task_type": "object-detection"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "claude_result",
	            "selector": "$steps.claude.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.predictions"
	        }
	    ]
	}
    ```

## Using Anthropic Claude as secondary classifier

In this example, Anthropic Claude model is used as secondary classifier - first, YOLO model
detects dogs, then for each dog we run classification with VLM and at the end we replace 
detections classes to have bounding boxes with dogs breeds labels.

Breeds that we classify: `russell-terrier`, `wirehaired-pointing-griffon`, `beagle`

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQVFWYXlLRVY3b09FYkhVcDRvN1YiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTh9.h8HWBijJ_PFEzimEY5X_hp1rEy-4dg5nNdkhBDqdBVY?showGraph=true" loading="lazy" title="Roboflow Workflow for claude secondary classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes",
	            "default_value": [
	                "russell-terrier",
	                "wirehaired-pointing-griffon",
	                "beagle"
	            ]
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
	            "type": "roboflow_core/anthropic_claude@v1",
	            "name": "claude",
	            "images": "$steps.cropping.crops",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$steps.cropping.crops",
	            "vlm_output": "$steps.claude.output",
	            "classes": "$steps.claude.classes"
	        },
	        {
	            "type": "roboflow_core/detections_classes_replacement@v1",
	            "name": "classes_replacement",
	            "object_detection_predictions": "$steps.general_detection.predictions",
	            "classification_predictions": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.classes_replacement.predictions"
	        }
	    ]
	}
    ```

## Florence 2 - grounded classification

**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image, which are later classified by Florence 2 model. With Workflows it is possible 
to pass `grounding_detection` as an input for all of the tasks named `detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiM25NcGZzakZPQU9OcmM0Q2I2NWkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NTl9.WWCLiIxzBVJ-TQGQ2R3W8AaNGw4aum66td4IyVUPsaw?showGraph=true" loading="lazy" title="Roboflow Workflow for florence 2 detection grounded classification" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640",
	            "confidence_mode": "custom",
	            "custom_confidence": "$inputs.confidence"
	        },
	        {
	            "type": "roboflow_core/florence_2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "task_type": "detection-grounded-classification",
	            "grounding_detection": "$steps.model_1.predictions",
	            "grounding_selection_mode": "most-confident"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Florence 2 - grounded segmentation

**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image and run segmentation of selected region with Florence 2. With Workflows it is 
possible to pass `grounding_detection` as an input for all of the tasks named 
`detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidTVJMDNHN2VEclhGUnZCY0VQYzIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjB9.4V1PmlwygSu6th7D2KKblr8vnRc4_2yOR4LhwhNmUDk?showGraph=true" loading="lazy" title="Roboflow Workflow for florence 2 detection grounded segmentation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/florence_2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "task_type": "detection-grounded-instance-segmentation",
	            "grounding_detection": "$steps.model_1.predictions",
	            "grounding_selection_mode": "most-confident"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Florence 2 - grounded captioning

**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image and run captioning of selected region with Florence 2. With Workflows it is 
possible to pass `grounding_detection` as an input for all of the tasks named 
`detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiOGtINXAzUHg4VTdLS0tJZ1lDRVYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjF9.dwUaT4DYCiw-pGw_99PxUXrwXzYYbc1_7yX36sfXq6w?showGraph=true" loading="lazy" title="Roboflow Workflow for florence 2 detection grounded caption" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/florence_2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "task_type": "detection-grounded-instance-segmentation",
	            "grounding_detection": "$steps.model_1.predictions",
	            "grounding_selection_mode": "most-confident"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Florence 2 - object detection

**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use Florence 2 as zero-shot object detection model, specifically 
performing open-vocabulary detection. Input parameter `classes` can be used to
provide list of objects that model should find. Beware that Florence 2 is prone to 
seek for all of the classes provided in your list - so if you select class which is not
visible in the image, you can expect either big bounding box covering whole image, 
or multiple bounding boxes over one of detected instance, with auxiliary boxes
providing not meaningful labels for all of the objects you specified in class list.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiOGtINXAzUHg4VTdLS0tJZ1lDRVYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjJ9.k4ybZpt6lb8lnZmWu5S4mOhFLU1jBaKeE-wllv4NnSA?showGraph=true" loading="lazy" title="Roboflow Workflow for florence 2 detection grounded caption" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model_1",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/florence_2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "task_type": "detection-grounded-instance-segmentation",
	            "grounding_detection": "$steps.model_1.predictions",
	            "grounding_selection_mode": "most-confident"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Prompting Google's Gemini with arbitrary prompt

In this example, Google's Gemini model is prompted with arbitrary text from user

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSmtJU21GelJsdTFwZXJyQ2ZPd0IiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjN9.4AcLmQdkHYPwWWurKj-i8xxaB0m9WLYkX6zcOlmYISI?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini arbitrary prompt" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "unconstrained",
	            "prompt": "Give me dominant color of the image",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gemini.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as OCR model

In this example, Google's Gemini model is used as OCR system. User just points task type and do not need to provide
any prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZjVMTUIzVEFuMnFFSjZ6Zk9TemoiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjN9.5yddglFw8sWwAGyvnZw57XGOfxZWTVVkX0Au-4G7yOo?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "ocr",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gemini.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as Visual Question Answering system

In this example, Google's Gemini model is used as VQA system. User provides question via prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVURlNmhiNnM5b2xvVEhab3R0WGYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjV9.zWL_pGjR8szr2RqIEYkWl85kbcS-qeK9-dbgkY9cb4s?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini vqa" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "visual-question-answering",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gemini.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as Image Captioning system

In this example, Google's Gemini model is used as Image Captioning system.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQVB5YUtnTmo2amxPeUM3U3Y5dFgiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NjZ9.DHR3a2GP9vLgI_CjYzGTPgzM8j0JMwuRY6bJc9DzJ3I?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini captioning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "caption",
	            "api_key": "$inputs.api_key",
	            "temperature": 1.0
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gemini.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as multi-class classifier

In this example, Google's Gemini model is used as classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQkZ2Q3VPN2hwRHpLNENmejJ2eUciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Njd9.3IiR2g4quyBhNPMlSaUSxVN50ZdfAOU_xFPnUobgxTA?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini multi class classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.gemini.output",
	            "classes": "$steps.gemini.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "gemini_result",
	            "selector": "$steps.gemini.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "top_class",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using Google's Gemini as multi-label classifier

In this example, Google's Gemini model is used as multi-label classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVlB0dm9meHRZMlhkenRjQXlObDUiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Njh9.k0hH3D8h5VyYivJQIOG868lIyhBqVxYZ5mbEnTbCibI?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini multi label classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "multi-label-classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.gemini.output",
	            "classes": "$steps.gemini.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using Google's Gemini to provide structured JSON

In this example, Google's Gemini model is expected to provide structured output in JSON, which can later be
parsed by dedicated `roboflow_core/json_parser@v1` block which transforms string into dictionary 
and expose it's keys to other blocks for further processing. In this case, parsed output is
transformed using `roboflow_core/property_definition@v1` block.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoicElGSDR4ZFFJSjZHRFhUZURpQ28iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Njl9.s6-vlwzq_iZh7Cr93NVOrE2dg2olPrqslUkyFd8bbCo?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini structured prompting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "structured-answering",
	            "output_structure": {
	                "dogs_count": "count of dogs instances in the image",
	                "cats_count": "count of cats instances in the image"
	            },
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/json_parser@v1",
	            "name": "parser",
	            "raw_json": "$steps.gemini.output",
	            "expected_fields": [
	                "dogs_count",
	                "cats_count"
	            ]
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "property_definition",
	            "operations": [
	                {
	                    "type": "ToString"
	                }
	            ],
	            "data": "$steps.parser.dogs_count"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.property_definition.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as object-detection model

In this example, Google's Gemini model is expected to provide output, which can later be
parsed by dedicated `roboflow_core/vlm_as_detector@v1` block which transforms string into `sv.Detections`, 
which can later be used by other blocks processing object-detection predictions.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibEN2ejR3Z2EyOUozOHdBUmFoTGsiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzB9.mR3G-tn0RdduvTZqgBkicF5z-03PrsidPlYEIZypl88?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini object detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "object-detection",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_detector@v2",
	            "name": "parser",
	            "vlm_output": "$steps.gemini.output",
	            "image": "$inputs.image",
	            "classes": "$steps.gemini.classes",
	            "model_type": "google-gemini",
	            "task_type": "object-detection"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "gemini_result",
	            "selector": "$steps.gemini.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.predictions"
	        }
	    ]
	}
    ```

## Using different versions of Google's Gemini for Image Captioning

In this example, we test different Gemini model versions for image captioning.
    This workflow allows specifying any supported Gemini model version as input parameter.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiejJIU0hCYXN6ME91aE1FbUZjMjIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzF9.-qdQr8JJmqbxSLA2t1oCN55IUgz5vClRpHFRn464CU4?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini version captioning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_version"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$inputs.image",
	            "task_type": "caption",
	            "api_key": "$inputs.api_key",
	            "model_version": "$inputs.model_version"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gemini.output"
	        }
	    ]
	}
    ```

## Using Google's Gemini as secondary classifier

In this example, Google's Gemini model is used as secondary classifier - first, YOLO model
detects dogs, then for each dog we run classification with VLM and at the end we replace 
detections classes to have bounding boxes with dogs breeds labels.

Breeds that we classify: `russell-terrier`, `wirehaired-pointing-griffon`, `beagle`

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibkl5YkNpNmZiSVUwVmtqQXBwMEsiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzJ9.a8POfq5JZin2YC_iN8j7C9IPKkIn1Q9k42eV_morqYI?showGraph=true" loading="lazy" title="Roboflow Workflow for gemini secondary classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes",
	            "default_value": [
	                "russell-terrier",
	                "wirehaired-pointing-griffon",
	                "beagle"
	            ]
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
	            "type": "roboflow_core/google_gemini@v1",
	            "name": "gemini",
	            "images": "$steps.cropping.crops",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$steps.cropping.crops",
	            "vlm_output": "$steps.gemini.output",
	            "classes": "$steps.gemini.classes"
	        },
	        {
	            "type": "roboflow_core/detections_classes_replacement@v1",
	            "name": "classes_replacement",
	            "object_detection_predictions": "$steps.general_detection.predictions",
	            "classification_predictions": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.classes_replacement.predictions"
	        }
	    ]
	}
    ```

## Prompting LLama Vision 3.2 with arbitrary prompt

In this example, LLama Vision 3.2 model is prompted with arbitrary text from user

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoia0xZNzJoNGtheU9QMk5XNkl1R3giLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzN9.YaeGojmpw5dJK5p-D_Oz3Li5BauAw1z9lVlv90XVRWk?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 arbitrary prompt" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "unconstrained",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.llama.output"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as OCR model

In this example, LLama Vision 3.2 model is used as OCR system. User just points task type and do not need to provide
any prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiU0lGVGE5b2pvZ3RsYUU2ZjVMY04iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzR9.j-5KjK7qZWciyJIGXKu0LSDwl2FLsr5EYQkI1W1UYSA?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "ocr",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.llama.output"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as Visual Question Answering system

In this example, LLama Vision 3.2 model is used as VQA system. User provides question via prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoicTRTbEN0aFJ4blJqUTk2ZEFneGYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzV9.UzsTxfbdzNunddNolPrAy0jZXIgcqsPRJim9aWSThHw?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 vqa" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "visual-question-answering",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.llama.output"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as Image Captioning system

In this example, LLama Vision 3.2 model is used as Image Captioning system.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZkhaMTN0Y1I1MmZ4TDRUMG1rNFMiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1NzZ9.pVVpqhgdoufqO8rMU1Y5r5sd8Jq89e-VS2fCSpDNDao?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 captioning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "caption",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.llama.output"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as multi-class classifier

In this example, LLama Vision 3.2 model is used as classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns LLama Vision 3.2 output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoib0ZQYzhiaEZKd3B0d2JNdVlMWmkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Nzd9.8vo_X9rL73q0Zbc2iJoWLyA3BzWXv78rLjvJyZWKkoc?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 multi class classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.llama.output",
	            "classes": "$steps.llama.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "llama_result",
	            "selector": "$steps.llama.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "top_class",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as multi-label classifier

In this example, LLama Vision 3.2 model is used as multi-label classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v1` block which turns LLama Vision 3.2 output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiRmNmMmZMQWl0aTRnb0FvSnZoUUEiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Nzh9.J53vZnYoPd9V0SvZZz7l-7h2-Y1mPUDi9u9-7dYc4HY?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 multi label classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "multi-label-classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.llama.output",
	            "classes": "$steps.llama.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 to provide structured JSON

In this example, LLama Vision 3.2 model is expected to provide structured output in JSON, which can later be
parsed by dedicated `roboflow_core/json_parser@v1` block which transforms string into dictionary 
and expose it's keys to other blocks for further processing. In this case, parsed output is
transformed using `roboflow_core/property_definition@v1` block.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiUlM4ZU5qbER2R21QZFl5TlRWbW0iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1Nzl9.HSVQuV9ALxbcL4m7ilVsHZ3jWNQghLUBrtx9pd2km24?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 structured prompting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$inputs.image",
	            "task_type": "structured-answering",
	            "output_structure": {
	                "dogs_count": "count of dogs instances in the image",
	                "cats_count": "count of cats instances in the image"
	            },
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        },
	        {
	            "type": "roboflow_core/json_parser@v1",
	            "name": "parser",
	            "raw_json": "$steps.llama.output",
	            "expected_fields": [
	                "dogs_count",
	                "cats_count"
	            ]
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "property_definition",
	            "operations": [
	                {
	                    "type": "ToString"
	                }
	            ],
	            "data": "$steps.parser.dogs_count"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "llama_output",
	            "selector": "$steps.llama.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.property_definition.output"
	        }
	    ]
	}
    ```

## Using LLama Vision 3.2 as secondary classifier

In this example, LLama Vision 3.2 model is used as secondary classifier - first, YOLO model
detects dogs, then for each dog we run classification with VLM and at the end we replace 
detections classes to have bounding boxes with dogs breeds labels.

Breeds that we classify: `russell-terrier`, `wirehaired-pointing-griffon`, `beagle`

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZ1ZIaERiVFlsQ2tMSWFZVEpBYlQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODB9.MmYQYwJeL_4WA525nUY7o8G2IYYRN9I3nrS61jRnAVo?showGraph=true" loading="lazy" title="Roboflow Workflow for llama vision 3 2 secondary classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes",
	            "default_value": [
	                "russell-terrier",
	                "wirehaired-pointing-griffon",
	                "beagle"
	            ]
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
	            "type": "roboflow_core/llama_3_2_vision@v1",
	            "name": "llama",
	            "images": "$steps.cropping.crops",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key",
	            "model_version": "11B (Regular) - OpenRouter"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$steps.cropping.crops",
	            "vlm_output": "$steps.llama.output",
	            "classes": "$steps.llama.classes"
	        },
	        {
	            "type": "roboflow_core/detections_classes_replacement@v1",
	            "name": "classes_replacement",
	            "object_detection_predictions": "$steps.general_detection.predictions",
	            "classification_predictions": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.classes_replacement.predictions"
	        }
	    ]
	}
    ```

## Moondream 2 - object detection

Use Moondream2 to detect objects in an image.

    You can pass in a prompt to the model to specify what you want to detect. The model will return a list of detection coordinates corresponding to the prompt.

    This block only works with one class at a time. This is because Moondream2 does not allow zero shot detection on more than one class at once.



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
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/moondream2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "prompt": "dog"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Prompting GPT with arbitrary prompt

In this example, GPT model is prompted with arbitrary text from user

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiUDUyeXdMSG9WRkxtUk5oR0hINzQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODF9.hlnEiLNfKFZv9h5ZmnBWzGzOSc30tjjxGCwAhm9znOU?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt arbitrary prompt" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "unconstrained",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gpt.output"
	        }
	    ]
	}
    ```

## Using GPT as OCR model

In this example, GPT model is used as OCR system. User just points task type and do not need to provide
any prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYWo3dU1vVmY3NEVVQUJtbXJZMUIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODJ9.vzncm_pjQIeGcXoLuLwE-U8E_bP1jFB3nikHPEVV_9o?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt ocr" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "ocr",
	            "api_key": "$inputs.api_key",
	            "model_version": "gpt-4o-mini"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gpt.output"
	        }
	    ]
	}
    ```

## Using GPT as Visual Question Answering system

In this example, GPT model is used as VQA system. User provides question via prompt.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZThRTDZlTDAwWmN3UWZoblBvWksiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODN9.Yy3AQoOrmo1SZGSUvo66t1421L200xn0rCXSyQ-xsk8?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt vqa" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "visual-question-answering",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gpt.output"
	        }
	    ]
	}
    ```

## Using GPT as Image Captioning system

In this example, GPT model is used as Image Captioning system.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSldhOHk0dEk4QjR2ZWVwT2E0bE4iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODR9.MFUEtXkx_hzEup-WiamAmuPxoNkGGnlgm_fWj5Q5o9s?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt captioning" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "caption",
	            "api_key": "$inputs.api_key"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.gpt.output"
	        }
	    ]
	}
    ```

## Using GPT as multi-class classifier

In this example, GPT model is used as classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns GPT output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVE5EV0YwOTZjUVdjUWVFbU9xMm0iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODR9.bmSDTKppXhk-IW8vpaSXgFK7qKI-MNDmTiVt2RqMklI?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt multi class classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.gpt.output",
	            "classes": "$steps.gpt.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "gpt_result",
	            "selector": "$steps.gpt.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "top_class",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using GPT as multi-label classifier

In this example, GPT model is used as multi-label classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v1` block which turns GPT output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVk5aTFBZbDRxUkV3VnJ2ZWxhYTIiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODV9.ZoGEVrRjmU2eR9W6TwjyTx_9ypY-FSUH8fZAO0FqStg?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt multi label classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "multi-label-classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$inputs.image",
	            "vlm_output": "$steps.gpt.output",
	            "classes": "$steps.gpt.classes"
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "top_class",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ],
	            "data": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.top_class.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "parsed_prediction",
	            "selector": "$steps.parser.*"
	        }
	    ]
	}
    ```

## Using GPT to provide structured JSON

In this example, GPT model is expected to provide structured output in JSON, which can later be
parsed by dedicated `roboflow_core/json_parser@v1` block which transforms string into dictionary 
and expose it's keys to other blocks for further processing. In this case, parsed output is
transformed using `roboflow_core/property_definition@v1` block.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoic1hKWEt6M21BVFYwVmh4bGdYN2IiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODd9.zDM_5tW74xs2Za5rsXTq-qHnQGsgCWQkuzZdtjv8WvE?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt structured prompting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$inputs.image",
	            "task_type": "structured-answering",
	            "output_structure": {
	                "dogs_count": "count of dogs instances in the image",
	                "cats_count": "count of cats instances in the image"
	            },
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/json_parser@v1",
	            "name": "parser",
	            "raw_json": "$steps.gpt.output",
	            "expected_fields": [
	                "dogs_count",
	                "cats_count"
	            ]
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "property_definition",
	            "operations": [
	                {
	                    "type": "ToString"
	                }
	            ],
	            "data": "$steps.parser.dogs_count"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "selector": "$steps.property_definition.output"
	        }
	    ]
	}
    ```

## Using GPT as secondary classifier

In this example, GPT model is used as secondary classifier - first, YOLO model
detects dogs, then for each dog we run classification with VLM and at the end we replace 
detections classes to have bounding boxes with dogs breeds labels.

Breeds that we classify: `russell-terrier`, `wirehaired-pointing-griffon`, `beagle`

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiY3p3V3ZLeGlzT1M0V3plYllMRUEiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODd9.9DeIMnZDhxJsXq_Uz9ajGYo0IMtMlWSJY9376eoGIts?showGraph=true" loading="lazy" title="Roboflow Workflow for gpt secondary classifier" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "classes",
	            "default_value": [
	                "russell-terrier",
	                "wirehaired-pointing-griffon",
	                "beagle"
	            ]
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
	            "type": "roboflow_core/open_ai@v2",
	            "name": "gpt",
	            "images": "$steps.cropping.crops",
	            "task_type": "classification",
	            "classes": "$inputs.classes",
	            "api_key": "$inputs.api_key"
	        },
	        {
	            "type": "roboflow_core/vlm_as_classifier@v2",
	            "name": "parser",
	            "image": "$steps.cropping.crops",
	            "vlm_output": "$steps.gpt.output",
	            "classes": "$steps.gpt.classes"
	        },
	        {
	            "type": "roboflow_core/detections_classes_replacement@v1",
	            "name": "classes_replacement",
	            "object_detection_predictions": "$steps.general_detection.predictions",
	            "classification_predictions": "$steps.parser.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.classes_replacement.predictions"
	        }
	    ]
	}
    ```

## SmolVLM2

**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

Use SmolVLM2 to ask questions about images, including documents and photos, and get answers in natural language.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYVBRUUxaYUp1dTQ3ckpmRXlhZW8iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODh9.y_nL4w5dwrLEnSxl6cB3-YIdgrYsJADtd4MtUbOCF7g?showGraph=true" loading="lazy" title="Roboflow Workflow for smolvlm2" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "confidence",
	            "default_value": 0.4
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/smolvlm2@v1",
	            "name": "model",
	            "images": "$inputs.image",
	            "task_type": "lmm",
	            "prompt": "What is in this image?"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "model_predictions",
	            "coordinates_system": "own",
	            "selector": "$steps.model.*"
	        }
	    ]
	}
    ```

## Prompting Stability-AI with arbitrary prompt

In this example, Stability-AI image generation model is prompted with arbitrary text from user

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQmY0UXlDODh5RzRkSko2RVNPOXMiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3ODE5MDQ1ODl9.4mHcrUOSGMze6bKtDFXDHANMWvn56LJDdFXaI1vKz1o?showGraph=true" loading="lazy" title="Roboflow Workflow for stability ai arbitrary prompt" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "WorkflowParameter",
	            "name": "api_key"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "prompt",
	            "default_value": "Raccoon in space suit"
	        },
	        {
	            "type": "InferenceImage",
	            "name": "image"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/stability_ai_image_gen@v1",
	            "name": "stability_ai_image_generation",
	            "prompt": "$inputs.prompt",
	            "api_key": "$inputs.api_key",
	            "image": "$inputs.image",
	            "strength": 0.3
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "stability_ai_image_generation",
	            "coordinates_system": "own",
	            "selector": "$steps.stability_ai_image_generation.image"
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