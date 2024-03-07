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