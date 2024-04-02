# `Crop`

Create dynamic crops of all regions returned as bounding boxes from an object detection or segmentation model.

## Step parameters
* `type`: must be `Crop` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `predictions`: must be a reference to `predictions` property of steps: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`, `YoloWorld`] (required)

## Step outputs:
* `crops` - `image` cropped based on `predictions`
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
