# Data analytics in Workflows

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow producing CSV

This example showcases how to export CSV file out of Workflow. Object detection results are 
processed with **CSV Formatter** block to produce aggregated results.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiOHNoWDljcE84S1IxWUlub0M5b0siLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMDN9.6a27nj6_c4iRWZEZVOCKqthU5r6_2oCTdjjfVjSemt0?showGraph=true" loading="lazy" title="Roboflow Workflow for csv formatter" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "additional_column_value"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-640"
	        },
	        {
	            "type": "roboflow_core/csv_formatter@v1",
	            "name": "csv_formatter",
	            "columns_data": {
	                "predicted_classes": "$steps.model.predictions",
	                "number_of_bounding_boxes": "$steps.model.predictions",
	                "additional_column": "$inputs.additional_column_value"
	            },
	            "columns_operations": {
	                "predicted_classes": [
	                    {
	                        "type": "DetectionsPropertyExtract",
	                        "property_name": "class_name"
	                    }
	                ],
	                "number_of_bounding_boxes": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "csv",
	            "coordinates_system": "own",
	            "selector": "$steps.csv_formatter.csv_content"
	        }
	    ]
	}
    ```

## Aggregation of results over time

This example shows how to aggregate and analyse predictions using Workflows.

The key for data analytics in this example is **Data Aggregator** block which is fed with model 
predictions and perform the following aggregations **on each 6 consecutive predictions:**

* taking **classes names** from  bounding boxes, it outputs **unique classes names, number of unique classes and
number of bounding boxes for each class**

* taking the **number of detected bounding boxes** in each prediction, it outputs **minimum, maximum and total number** 
of bounding boxes per prediction in aggregated time window 

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoianltSWlPTURKenJUdG1JbFplTzciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMDR9.6s690PwsrkAJti6d6bMTFlhozaMi47qIvWDmu_Cr14M?showGraph=true" loading="lazy" title="Roboflow Workflow for data aggregation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/data_aggregator@v1",
	            "name": "data_aggregation",
	            "data": {
	                "predicted_classes": "$steps.model.predictions",
	                "number_of_predictions": "$steps.model.predictions"
	            },
	            "data_operations": {
	                "predicted_classes": [
	                    {
	                        "type": "DetectionsPropertyExtract",
	                        "property_name": "class_name"
	                    }
	                ],
	                "number_of_predictions": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            },
	            "aggregation_mode": {
	                "predicted_classes": [
	                    "distinct",
	                    "count_distinct",
	                    "values_counts"
	                ],
	                "number_of_predictions": [
	                    "min",
	                    "max",
	                    "sum"
	                ]
	            },
	            "interval": 6,
	            "interval_unit": "runs"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "aggregation_results",
	            "selector": "$steps.data_aggregation.*"
	        }
	    ]
	}
    ```

## Saving Workflow results into file

This Workflow was created to achieve few ends:

* getting predictions from object detection model and returning them to the caller

* persisting the predictions - each one in separate JSON file

* aggregating the predictions data - producing report on each 6th input image

* saving the results in CSV file, appending rows until file size is exceeded 

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZWhZSUQzdTNkVEZZcFpFdmlHVjUiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMDV9.IUmbx5EDY9ZvWDNwZk4AFok0N2IogPuspbSNf7f1kx8?showGraph=true" loading="lazy" title="Roboflow Workflow for file sink for data aggregation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "name": "target_directory"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "model_id",
	            "default_value": "yolov8n-640"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/roboflow_object_detection_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "$inputs.model_id"
	        },
	        {
	            "type": "roboflow_core/expression@v1",
	            "name": "json_formatter",
	            "data": {
	                "predictions": "$steps.model.predictions"
	            },
	            "switch": {
	                "type": "CasesDefinition",
	                "cases": [],
	                "default": {
	                    "type": "DynamicCaseResult",
	                    "parameter_name": "predictions",
	                    "operations": [
	                        {
	                            "type": "DetectionsToDictionary"
	                        },
	                        {
	                            "type": "ConvertDictionaryToJSON"
	                        }
	                    ]
	                }
	            }
	        },
	        {
	            "type": "roboflow_core/local_file_sink@v1",
	            "name": "predictions_sink",
	            "content": "$steps.json_formatter.output",
	            "file_type": "json",
	            "output_mode": "separate_files",
	            "target_directory": "$inputs.target_directory",
	            "file_name_prefix": "prediction"
	        },
	        {
	            "type": "roboflow_core/data_aggregator@v1",
	            "name": "data_aggregation",
	            "data": {
	                "predicted_classes": "$steps.model.predictions",
	                "number_of_predictions": "$steps.model.predictions"
	            },
	            "data_operations": {
	                "predicted_classes": [
	                    {
	                        "type": "DetectionsPropertyExtract",
	                        "property_name": "class_name"
	                    }
	                ],
	                "number_of_predictions": [
	                    {
	                        "type": "SequenceLength"
	                    }
	                ]
	            },
	            "aggregation_mode": {
	                "predicted_classes": [
	                    "count_distinct"
	                ],
	                "number_of_predictions": [
	                    "min",
	                    "max",
	                    "sum"
	                ]
	            },
	            "interval": 6,
	            "interval_unit": "runs"
	        },
	        {
	            "type": "roboflow_core/csv_formatter@v1",
	            "name": "csv_formatter",
	            "columns_data": {
	                "number_of_distinct_classes": "$steps.data_aggregation.predicted_classes_count_distinct",
	                "min_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_min",
	                "max_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_max",
	                "total_number_of_bounding_boxes": "$steps.data_aggregation.number_of_predictions_sum"
	            }
	        },
	        {
	            "type": "roboflow_core/local_file_sink@v1",
	            "name": "reports_sink",
	            "content": "$steps.csv_formatter.csv_content",
	            "file_type": "csv",
	            "output_mode": "append_log",
	            "target_directory": "$inputs.target_directory",
	            "file_name_prefix": "aggregation_report"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "predictions",
	            "selector": "$steps.model.predictions"
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