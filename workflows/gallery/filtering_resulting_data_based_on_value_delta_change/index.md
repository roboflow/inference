# Filtering resulting data based on value delta change

Below you can find example workflows you can use as inspiration to build your apps.

## Saving Workflow results into file, but only if value changes between frames

This Workflow was created to achieve few ends:

* getting predictions from object detection model

* filtering out predictions found outside of zone

* counting detections in zone

* if count of detection in zone changes save results to csv file

!!! warning "Run on video to produce *meaningful* results"

    This workflow will not work using the docs preview. You must run it on video file.
    Copy the template into your Roboflow app, start `inference` server and use video preview 
    to get the results.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiZWhZSUQzdTNkVEZZcFpFdmlHVjUiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgxMDd9.GO76nWa05r622hi1R7Iw6NSdpCj331QcVSxd4w6HPhI?showGraph=true" loading="lazy" title="Roboflow Workflow for file sink for data aggregation" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/detections_filter@v1",
	            "name": "detections_filter",
	            "predictions": "$steps.model.predictions",
	            "operations": [
	                {
	                    "type": "DetectionsFilter",
	                    "filter_operation": {
	                        "type": "StatementGroup",
	                        "operator": "and",
	                        "statements": [
	                            {
	                                "type": "BinaryStatement",
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "_",
	                                    "operations": [
	                                        {
	                                            "type": "ExtractDetectionProperty",
	                                            "property_name": "center"
	                                        }
	                                    ]
	                                },
	                                "comparator": {
	                                    "type": "(Detection) in zone"
	                                },
	                                "right_operand": {
	                                    "type": "StaticOperand",
	                                    "value": [
	                                        [
	                                            0,
	                                            0
	                                        ],
	                                        [
	                                            0,
	                                            1000
	                                        ],
	                                        [
	                                            1000,
	                                            1000
	                                        ],
	                                        [
	                                            1000,
	                                            0
	                                        ]
	                                    ]
	                                },
	                                "negate": false
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {}
	        },
	        {
	            "type": "roboflow_core/property_definition@v1",
	            "name": "property_definition",
	            "data": "$steps.detections_filter.predictions",
	            "operations": [
	                {
	                    "type": "SequenceLength"
	                }
	            ]
	        },
	        {
	            "type": "roboflow_core/delta_filter@v1",
	            "name": "delta_filter",
	            "value": "$steps.property_definition.output",
	            "image": "$inputs.image",
	            "next_steps": [
	                "$steps.csv_formatter"
	            ]
	        },
	        {
	            "type": "roboflow_core/csv_formatter@v1",
	            "name": "csv_formatter",
	            "columns_data": {
	                "Class Name": "$steps.detections_filter.predictions"
	            },
	            "columns_operations": {
	                "Class Name": [
	                    {
	                        "type": "DetectionsPropertyExtract",
	                        "property_name": "class_name"
	                    }
	                ]
	            }
	        },
	        {
	            "type": "roboflow_core/local_file_sink@v1",
	            "name": "reports_sink",
	            "content": "$steps.csv_formatter.csv_content",
	            "file_type": "csv",
	            "output_mode": "append_log",
	            "target_directory": "$inputs.target_directory",
	            "file_name_prefix": "csv_containing_changes"
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

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>