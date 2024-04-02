# `DetectionOffset`

Apply fixed offset on width and height of detections.

## Step parameters
* `type`: must be `DetectionOffset` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: reference to `predictions` output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`, `DetectionOffset`, `YoloWorld`] (required)
* `offset_x`: reference to input parameter of integer value for detection width offset (required)
* `offset_y`: reference to input parameter of integer value for detection height offset (required)

## Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - denoting parent model type


## Format of `predictions`
`predictions` is batch-major list of size `[batch_size, #detections_for_input_image]`.
Each detection is in format:
```json
{
    "parent_id": "uuid_of_parent_element",
    "class": "class_determined_by_model",
    "class_id": 0,
    "confidence": 1.0,
    "x": 128.5,
    "y": 327.8,
    "width": 200.0,
    "height": 150.0,
    "detection_id": "uuid_of_detection", 
    "keypoints": [
      {"x":  30.5, "y":  128.3, "confidence":  0.3, "class_id":  0, "class_name": "ankle"}
    ],
    "points": [
      {"x":  30.5, "y":  128.3}
    ]      
}
```

!!! note
    
    `keypoints` field will only be present when `KeypointsDetectionModel` output given
    `points` field will only be present when `InstanceSegmentationModel` output given
    