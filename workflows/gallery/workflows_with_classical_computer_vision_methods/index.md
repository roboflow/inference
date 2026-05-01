# Workflows with classical Computer Vision methods

Below you can find example workflows you can use as inspiration to build your apps.

## Workflow for background subtraction in video stream

This example shows how Background Subtraction block can be used to extract motion masks from a video stream.

The background subtraction block uses MOG2 (Mixture of Gaussians) algorithm to identify pixels that
differ significantly from the background model. This is useful for motion detection, object tracking,
and video analysis tasks.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiRVpISzZtejU1VDBudW1sNXBYeDkiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjV9.tPZEdo8_Re3oBPMiku4VMOO9jIbR4QvSOIEjASATAfg?showGraph=true" loading="lazy" title="Roboflow Workflow for background subtraction" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/background_subtraction@v1",
	            "name": "bg_subtractor",
	            "image": "$inputs.image",
	            "threshold": 16,
	            "history": 30
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "output_image",
	            "coordinates_system": "own",
	            "selector": "$steps.bg_subtractor.image"
	        }
	    ]
	}
    ```

## Workflow removing camera distortions

In this example, we demonstrate how to remove distortions from the camera based on coefficients provided.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoidXNwbmdDRlVnNlU5Q0tkdmpxUFUiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjd9.zZ_v3CoSp0WAjT0_vYzn8pIekYAxg4zm88AjQo3qxc8?showGraph=true" loading="lazy" title="Roboflow Workflow for camera calibration" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "images"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "fx"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "fy"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "cx"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "cy"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "k1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "k2"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "k3"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "p1"
	        },
	        {
	            "type": "WorkflowParameter",
	            "name": "p2"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/camera-calibration@v1",
	            "name": "camera_calibration",
	            "images": "$inputs.images",
	            "fx": "$inputs.fx",
	            "fy": "$inputs.fy",
	            "cx": "$inputs.cx",
	            "cy": "$inputs.cy",
	            "k1": "$inputs.k1",
	            "k2": "$inputs.k2",
	            "k3": "$inputs.k3",
	            "p1": "$inputs.p1",
	            "p2": "$inputs.p2"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "camera_calibration_image",
	            "coordinates_system": "own",
	            "selector": "$steps.camera_calibration.calibrated_image"
	        }
	    ]
	}
    ```

## Workflow generating camera focus measure

In this example, we demonstrate how to evaluate camera focus using the Tenengrad focus measure
with visualization overlays including zebra warnings, focus peaking, HUD, and composition grid.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiQVIzRkJubjZFUzZzRHdXTVRONXYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjh9.HRw9qHpC-2bCmT_0qvYmWezwkscCtedi0XumuBZXpOg?showGraph=true" loading="lazy" title="Roboflow Workflow for camera focus" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/camera_focus@v2",
	            "name": "camera_focus",
	            "image": "$inputs.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "camera_focus_image",
	            "coordinates_system": "own",
	            "selector": "$steps.camera_focus.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "camera_focus_measure",
	            "selector": "$steps.camera_focus.focus_measure"
	        },
	        {
	            "type": "JsonField",
	            "name": "bbox_focus_measures",
	            "selector": "$steps.camera_focus.bbox_focus_measures"
	        }
	    ]
	}
    ```

## Workflow detecting contours

In this example we show how classical contour detection works in cooperation
with blocks performing its pre-processing (conversion to gray and blur).

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiaElHRDVGMkEwa29KN3Jhd1l2b2wiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMjl9.ijwddhxl_-Di8r6eEpYapTZCEyuONlqDsGzeVgIJZQ4?showGraph=true" loading="lazy" title="Roboflow Workflow for contours detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/convert_grayscale@v1",
	            "name": "image_convert_grayscale",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/image_blur@v1",
	            "name": "image_blur",
	            "image": "$steps.image_convert_grayscale.image"
	        },
	        {
	            "type": "roboflow_core/threshold@v1",
	            "name": "image_threshold",
	            "image": "$steps.image_blur.image",
	            "thresh_value": 200,
	            "threshold_type": "binary_inv"
	        },
	        {
	            "type": "roboflow_core/contours_detection@v1",
	            "name": "image_contours",
	            "image": "$steps.image_threshold.image",
	            "raw_image": "$inputs.image",
	            "line_thickness": 5
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "number_contours",
	            "coordinates_system": "own",
	            "selector": "$steps.image_contours.number_contours"
	        },
	        {
	            "type": "JsonField",
	            "name": "contour_image",
	            "coordinates_system": "own",
	            "selector": "$steps.image_contours.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "contours",
	            "coordinates_system": "own",
	            "selector": "$steps.image_contours.contours"
	        },
	        {
	            "type": "JsonField",
	            "name": "grayscale_image",
	            "coordinates_system": "own",
	            "selector": "$steps.image_convert_grayscale.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "blurred_image",
	            "coordinates_system": "own",
	            "selector": "$steps.image_blur.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "thresholded_image",
	            "coordinates_system": "own",
	            "selector": "$steps.image_threshold.image"
	        }
	    ]
	}
    ```

## Workflow calculating pixels with dominant color

This example shows how Dominant Color block and Pixel Color Count block can be used together.

First, dominant color gets detected and then number of pixels with that color is calculated.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoibU55eDdhMDd0VzZPWjFxRzFrb0EiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzB9.NcWtGngqoPGVcBMstdQqEuzZulwZbWeeJGrzftoBlic?showGraph=true" loading="lazy" title="Roboflow Workflow for pixels counting" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/dominant_color@v1",
	            "name": "dominant_color",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/pixel_color_count@v1",
	            "name": "pixelation",
	            "image": "$inputs.image",
	            "target_color": "$steps.dominant_color.rgb_color"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "matching_pixels_count",
	            "coordinates_system": "own",
	            "selector": "$steps.pixelation.matching_pixels_count"
	        }
	    ]
	}
    ```

## Workflow calculating dominant color

This example shows how Dominant Color block can be used against input image.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiVlpaS1dGYnprWDc5TWw0bkdzaDYiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzF9.zQmRIkgrMFJAzuJ-0MjdITafCZPEb4_3gRrgGoN-f7s?showGraph=true" loading="lazy" title="Roboflow Workflow for dominant color" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/dominant_color@v1",
	            "name": "dominant_color",
	            "image": "$inputs.image"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "color",
	            "coordinates_system": "own",
	            "selector": "$steps.dominant_color.rgb_color"
	        }
	    ]
	}
    ```

## Workflow with dynamic zone and perspective converter

In this example dynamic zone with 4 vertices is calculated from detected segmentations.
Perspective correction is applied to the input image as well as to detected segmentations based on this zone.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiTFQ5Q05SZTdJeHJhMXJEM1ZvckQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzN9.Qt9JgizcTmcgjA9WXfVboyi-JvOhzVgOVyMePyFjKsQ?showGraph=true" loading="lazy" title="Roboflow Workflow for dynamic_zone_and_perspective_converter" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	            "name": "model",
	            "images": "$inputs.image",
	            "model_id": "yolov8n-seg-640"
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
	                                "negate": false,
	                                "left_operand": {
	                                    "type": "DynamicOperand",
	                                    "operand_name": "_",
	                                    "operations": [
	                                        {
	                                            "type": "ExtractDetectionProperty",
	                                            "property_name": "class_name"
	                                        }
	                                    ]
	                                },
	                                "comparator": {
	                                    "type": "in (Sequence)"
	                                },
	                                "right_operand": {
	                                    "type": "StaticOperand",
	                                    "value": [
	                                        "banana"
	                                    ]
	                                }
	                            }
	                        ]
	                    }
	                }
	            ],
	            "operations_parameters": {}
	        },
	        {
	            "type": "roboflow_core/dynamic_zone@v1",
	            "name": "dynamic_zone",
	            "predictions": "$steps.detections_filter.predictions",
	            "required_number_of_vertices": 4
	        },
	        {
	            "type": "roboflow_core/perspective_correction@v1",
	            "name": "perspective_correction",
	            "images": "$inputs.image",
	            "perspective_polygons": "$steps.dynamic_zone.zones",
	            "predictions": "$steps.model.predictions",
	            "warp_image": true,
	            "extend_perspective_polygon_by_detections_anchor": "BOTTOM_CENTER"
	        },
	        {
	            "type": "roboflow_core/polygon_visualization@v1",
	            "name": "perspective_visualization",
	            "image": "$steps.perspective_correction.warped_image",
	            "predictions": "$steps.perspective_correction.corrected_coordinates"
	        },
	        {
	            "type": "roboflow_core/polygon_visualization@v1",
	            "name": "polygon_visualization",
	            "image": "$inputs.image",
	            "predictions": "$steps.model.predictions"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "polygons_visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.polygon_visualization.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "perspective_visualization",
	            "coordinates_system": "own",
	            "selector": "$steps.perspective_visualization.image"
	        },
	        {
	            "type": "JsonField",
	            "name": "perspective_correction_outputs",
	            "coordinates_system": "own",
	            "selector": "$steps.perspective_correction.*"
	        },
	        {
	            "type": "JsonField",
	            "name": "dynamic_zones",
	            "coordinates_system": "own",
	            "selector": "$steps.dynamic_zone.zones"
	        }
	    ]
	}
    ```

## Workflow resizing the input image

This example shows how the Image Preprocessing block can be used to resize an input image.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYWN3cksxTmZ5YXZ2WnpXMld1SHQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzR9.3Osg9NvJJB1yPWXK0pH-ZSLjANw7RiRy4iRafRLyV4o?showGraph=true" loading="lazy" title="Roboflow Workflow for resize image" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/image_preprocessing@v1",
	            "name": "resize_image",
	            "image": "$inputs.image",
	            "task_type": "resize",
	            "width": 1000,
	            "height": 800
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "resized_image",
	            "coordinates_system": "own",
	            "selector": "$steps.resize_image.image"
	        }
	    ]
	}
    ```

## Workflow detecting motion in video stream

This example shows how Motion Detection block can be used to detect motion in a video stream.

The motion detector uses background subtraction to identify moving elements in the video.
It outputs a motion flag, an alarm flag (which triggers when motion starts), detected objects,
and a visualization of the detected motion.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiMFBKWmU0ZE12bExJck1qNFlWTEciLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzZ9.bNr3h4SuczvcI77ZbpMiSHfJyxJoWsoHYS8kuoUjUts?showGraph=true" loading="lazy" title="Roboflow Workflow for motion detection" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "roboflow_core/motion_detection@v1",
	            "name": "motion_detector",
	            "image": "$inputs.image",
	            "threshold": 16,
	            "history": 30,
	            "minimum_contour_area": 200,
	            "morphological_kernel_size": 3
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "motion_detected",
	            "coordinates_system": "own",
	            "selector": "$steps.motion_detector.motion"
	        },
	        {
	            "type": "JsonField",
	            "name": "motion_alarm",
	            "coordinates_system": "own",
	            "selector": "$steps.motion_detector.alarm"
	        },
	        {
	            "type": "JsonField",
	            "name": "detections",
	            "coordinates_system": "own",
	            "selector": "$steps.motion_detector.detections"
	        },
	        {
	            "type": "JsonField",
	            "name": "motion_zones",
	            "coordinates_system": "own",
	            "selector": "$steps.motion_detector.motion_zones"
	        }
	    ]
	}
    ```

## SIFT in Workflows

In this example we check how SIFT-based pattern matching works in cooperation
with expression block.

The Workflow first calculates SIFT features for input image and reference template, 
then image features are compared to template features. At the end - switch-case 
statement is built with Expression block to produce output. 

Important detail: If there is empty output from SIFT descriptors calculation
for (which is a valid output if no feature gets recognised) the sift comparison won't 
execute - hence First Non Empty Or Default block is used to provide default outcome 
for `images_match` output of SIFT comparison block.

Please note that a single image can be passed as template, and batch of images
are passed as images to look for template. This workflow does also validate
Execution Engine capabilities to broadcast batch-oriented inputs properly.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiV1hSSWNVWVk3bEpzMHZiZ0g2a24iLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzd9.YaeUr2gn6RCf0ikpT8sF4Ut9iniLHSQ8ADHhz27qWXI?showGraph=true" loading="lazy" title="Roboflow Workflow for sift" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

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
	            "type": "InferenceImage",
	            "name": "template"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/sift@v1",
	            "name": "image_sift",
	            "image": "$inputs.image"
	        },
	        {
	            "type": "roboflow_core/sift@v1",
	            "name": "template_sift",
	            "image": "$inputs.template"
	        },
	        {
	            "type": "roboflow_core/sift_comparison@v1",
	            "name": "sift_comparison",
	            "descriptor_1": "$steps.image_sift.descriptors",
	            "descriptor_2": "$steps.template_sift.descriptors",
	            "good_matches_threshold": 50
	        },
	        {
	            "type": "roboflow_core/first_non_empty_or_default@v1",
	            "name": "empty_values_replacement",
	            "data": [
	                "$steps.sift_comparison.images_match"
	            ],
	            "default": false
	        },
	        {
	            "type": "roboflow_core/expression@v1",
	            "name": "is_match_expression",
	            "data": {
	                "images_match": "$steps.empty_values_replacement.output"
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
	                                    "type": "UnaryStatement",
	                                    "operand": {
	                                        "type": "DynamicOperand",
	                                        "operand_name": "images_match"
	                                    },
	                                    "operator": {
	                                        "type": "(Boolean) is True"
	                                    }
	                                }
	                            ]
	                        },
	                        "result": {
	                            "type": "StaticCaseResult",
	                            "value": "MATCH"
	                        }
	                    }
	                ],
	                "default": {
	                    "type": "StaticCaseResult",
	                    "value": "NO MATCH"
	                }
	            }
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "result",
	            "coordinates_system": "own",
	            "selector": "$steps.is_match_expression.output"
	        }
	    ]
	}
    ```

## Workflow stitching images

In this example two images of the same scene are stitched together.
Given enough shared details order of the images does not influence final result.

Please note that images need to have enough common details for the algorithm to stitch them properly.

<div style="height: 768px; min-height: 400px; min-width: 768px; overflow: hidden;"><iframe src="https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiYzlpQ0tGVThkRnh2eU1FcWlXZTQiLCJ3b3Jrc3BhY2VJZCI6IkppUUdZcmR1WXFMOGM5alRNZ29RIiwidXNlcklkIjoiZG9jcy1nZW5lcmF0ZWQiLCJpYXQiOjE3Nzc2NDgwMzh9.GWkuPvWMnO0lmK7j1wdZvgRpqudWl7ntpKfUAh7GnPg?showGraph=true" loading="lazy" title="Roboflow Workflow for stitch_images" style="width: 100%; height: 100%; min-height: 400px; border: none;"></iframe></div>

??? tip "Workflow definition"

    ```json
    {
	    "version": "1.0",
	    "inputs": [
	        {
	            "type": "InferenceImage",
	            "name": "image1"
	        },
	        {
	            "type": "InferenceImage",
	            "name": "image2"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "count_of_best_matches_per_query_descriptor"
	        },
	        {
	            "type": "InferenceParameter",
	            "name": "max_allowed_reprojection_error"
	        }
	    ],
	    "steps": [
	        {
	            "type": "roboflow_core/stitch_images@v1",
	            "name": "stitch_images",
	            "image1": "$inputs.image1",
	            "image2": "$inputs.image2",
	            "count_of_best_matches_per_query_descriptor": "$inputs.count_of_best_matches_per_query_descriptor",
	            "max_allowed_reprojection_error": "$inputs.max_allowed_reprojection_error"
	        }
	    ],
	    "outputs": [
	        {
	            "type": "JsonField",
	            "name": "stitched_image",
	            "selector": "$steps.stitch_images.stitched_image"
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