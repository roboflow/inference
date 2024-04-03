# `DetectionFilter`

Filter predictions from detection models based on conditions defined.

## Step parameters
* `type`: must be `DetectionFilter` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: reference to `predictions` output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`, `DetectionOffset`, `YoloWorld`] (required)
* `filter_definition`: definition of the filter (required)

Filter definition can be either `DetectionFilterDefinition`
```json
{
  "type": "DetectionFilterDefinition",
  "field_name": "confidence",
  "operator": "greater_or_equal_than",
  "reference_value": 0.2
}
```
or `CompoundDetectionFilterDefinition`
```json
{
    "type": "CompoundDetectionFilterDefinition",
    "left": {
        "type": "DetectionFilterDefinition",
        "field_name": "class_name",
        "operator": "equal",
        "reference_value": "car"
    },
    "operator": "and",
    "right": {
        "type": "DetectionFilterDefinition",
        "field_name": "confidence",
        "operator": "greater_or_equal_than",
        "reference_value": 0.2
    }
}
```

where `DetectionFilterDefinition` uses binary operator and the left operand is detection field pointed by `field_name`
and right operand is `reference_value`. `"operaror"` can be filled with values:
* `equal` (field value equal to `reference_value`)
* `not_equal`
* `lower_than`
* `greater_than`
* `lower_or_equal_than`
* `greater_or_equal_than`
* `in` (field value in range of `reference_value`)
* `str_starts_with` (field value - string - starts from `reference_value`)
* `str_ends_with` (field value - string - ends with `reference_value`)
* `str_contains` (field value - string - contains substring pointed in `reference_value`)

In case if `CompoundDetectionFilterDefinition`, logical operators `or`, `and` can be used to combine simple filters.
This let user define recursive structure of filters.

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
    
    While creating `filter_definition` you may use as `field_name` that is present within input 
    detection dict. Formats of predictions may be checked in input steps' docs.
