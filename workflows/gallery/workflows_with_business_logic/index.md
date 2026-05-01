# Workflows with business logic

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow with extraction of classes for detections (1)

In practical use-cases you may find the need to inject pieces of business logic inside 
your Workflow, such that it is easier to integrate with app created in Workflows ecosystem.

Translation of model predictions into domain-specific language of your business is possible 
with specialised blocks that let you parametrise such programming constructs 
as switch-case statements.

In this example, our goal is to:

- tell how many objects are detected

- verify that the picture presents exactly two dogs

To achieve that goal, we run generic object detection model as first step, then we use special
block called Property Definition that is capable of executing various operations to
transform input data into desired output. We have two such blocks:

- `instances_counter` which takes object detection predictions and apply operation to extract sequence length - 
effectively calculating number of instances of objects that were predicted

- `property_extraction` which extracts class names from all detected bounding boxes

`instances_counter` basically completes first goal of the workflow, but to satisfy the second one we need to 
build evaluation logic that will tell "PASS" / "FAIL" based on comparison of extracted class names with 
reference parameter (provided via Workflow input `$inputs.reference`). We can use Expression block to achieve 
that goal - building custom case statements (checking if class names being list of classes 
extracted from object detection prediction matches reference passed in the input).

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMWZRTDhXQ1ZTdzRCRXo5dFk0QnciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzJ9.xGsos79AkYDwEp_j0KV-xwrwSnX1e4_x4JTk0i7JJAo?showGraph=true" loading="lazy" title="Roboflow Workflow for business logic 1" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "ObjectDetectionModel",
	            "name": "general_detection",
	            "image": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "PropertyDefinition",
	            "name": "property_extraction",
	            "data": "$steps.general_detection.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsPropertyExtract",
	                    "property_name": "class_name"
	                }
	            ]
	        },
	        {
	            "type": "PropertyDefinition",
	            "name": "instances_counter",
	            "data": "$steps.general_detection.predictions",
	            "operations": [
	                {
	                    "type": "SequenceLength"
	                }
	            ]
	        },
	        {
	            "type": "Expression",
	            "name": "expression",
	            "data": {
	                "class_names": "$steps.property_extraction.output",
	                "reference": "$inputs.reference"
	            },
	            "switch": {
	                "type": "CasesDefinition",
	                "cases": [
	                    {
	                        "type": "CaseDefinition",
	                        "condition": {
	                            "type": "StatementGroup",
	                            "statements": [
	                                {
	                                    "type": "BinaryStatement",
	                                    "left_operand": {
	                                        "type": "DynamicOperand",
	                                        "operand_name": "class_names"
	                                    },
	                                    "comparator": {
	                                        "type": "=="
	                                    },
	                                    "right_operand": {
	                                        "type": "DynamicOperand",
	                                        "operand_name": "reference"
	                                    }
	                                }
	                            ]
	                        },
	                        "result": {
	                            "type": "StaticCaseResult",
	                            "value": "PASS"
	                        }
	                    }
	                ],
	                "default": {
	                    "type": "StaticCaseResult",
	                    "value": "FAIL"
	                }
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "detected_classes",
	            "selector": "$steps.property_extraction.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "number_of_detected_boxes",
	            "selector": "$steps.instances_counter.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "verdict",
	            "selector": "$steps.expression.output"
	        }
	    ]
	}
    ```

## Workflow with extraction of classes for detections (2)

In practical use-cases you may find the need to inject pieces of business logic inside 
your Workflow, such that it is easier to integrate with app created in Workflows ecosystem.

Translation of model predictions into domain-specific language of your business is possible 
with specialised blocks that let you parametrise such programming constructs 
as switch-case statements.

In this example, our goal is to:

- run generic object detection model to find instances of dogs

- crop dogs detection

- run specialised dogs breed classifier to assign granular label for each dog

- compare predicted dogs breeds to verify if detected labels matches exactly reverence value passed in input.

This example is quite complex as it requires quite deep understanding of Workflows ecosystem. Let's start from
the beginning - we run object detection model, crop its detections according to dogs class to perform 
classification. This is quite typical for workflows (you may find such pattern in remaining examples). 

The complexity increases when we try to handle classification output. We need to have a list of classes
for each input image, but for now we have complex objects with all classification predictions details
provided by `breds_classification` step - what is more - we have batch of such predictions for
each input image (as we created dogs crops based on object detection model predictions). To solve the 
problem, at first we apply Property Definition step taking classifier predictions and turning them into
strings representing predicted classes. We still have batch of class names at dimensionality level 2, 
which needs to be brought into dimensionality level 1 to make a single comparison against reference 
value for each input image. To achieve that effect we use Dimension Collapse block which does nothing
else but grabs the batch of classes and turns it into list of classes at dimensionality level 1 - one 
list for each input image.

That would solve our problems, apart from one nuance that must be taken into account. First-stage model
is not guaranteed to detect any dogs - and if that happens we do not execute cropping and further 
processing for that image, leaving all outputs derived from downstream computations `None` which is
suboptimal. To compensate for that, we may use First Non Empty Or Default block which will take 
`outputs_concatenation` step output and replace missing values with empty list (as effectively this is 
equivalent of not detecting any dog).

Such prepared output of `empty_values_replacement` step may be now plugged into Expression block, 
performing switch-case like logic to deduce if breeds of detected dogs match with reference value 
passed to workflow execution.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiemVhaFROSWNHRGNKbHFtbDliWWYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMzR9.2vElVE-_iz_Iyt1ch7F5IG8ZOjo5TdfO--aGUrj2pks?showGraph=true" loading="lazy" title="Roboflow Workflow for business logic 2" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "PropertyDefinition",
	            "name": "property_extraction",
	            "data": "$steps.breds_classification.predictions",
	            "operations": [
	                {
	                    "type": "ClassificationPropertyExtract",
	                    "property_name": "top_class"
	                }
	            ]
	        },
	        {
	            "type": "DimensionCollapse",
	            "name": "outputs_concatenation",
	            "data": "$steps.property_extraction.output"
	        },
	        {
	            "type": "FirstNonEmptyOrDefault",
	            "name": "empty_values_replacement",
	            "data": [
	                "$steps.outputs_concatenation.output"
	            ],
	            "default": []
	        },
	        {
	            "type": "Expression",
	            "name": "expression",
	            "data": {
	                "detected_classes": "$steps.empty_values_replacement.output",
	                "reference": "$inputs.reference"
	            },
	            "switch": {
	                "type": "CasesDefinition",
	                "cases": [
	                    {
	                        "type": "CaseDefinition",
	                        "condition": {
	                            "type": "StatementGroup",
	                            "statements": [
	                                {
	                                    "type": "BinaryStatement",
	                                    "left_operand": {
	                                        "type": "DynamicOperand",
	                                        "operand_name": "detected_classes"
	                                    },
	                                    "comparator": {
	                                        "type": "=="
	                                    },
	                                    "right_operand": {
	                                        "type": "DynamicOperand",
	                                        "operand_name": "reference"
	                                    }
	                                }
	                            ]
	                        },
	                        "result": {
	                            "type": "StaticCaseResult",
	                            "value": "PASS"
	                        }
	                    }
	                ],
	                "default": {
	                    "type": "StaticCaseResult",
	                    "value": "FAIL"
	                }
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "detected_classes",
	            "selector": "$steps.property_extraction.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "wrapped_classes",
	            "selector": "$steps.empty_values_replacement.output"
	        },
	        {
	            "type": "JsonField",
	            "name": "verdict",
	            "selector": "$steps.expression.output"
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