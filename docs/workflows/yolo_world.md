# `YoloWorld`
This `workflows` block is supposed to bring [Yolo World model](https://blog.roboflow.com/what-is-yolo-world/) to the
`workflows` world! You can use it in a very similar way as other object detection models within `workflows`.

!!! important

    This step for now only works in Python package and `inference` HTTP container hosted locally. Hosted Roboflow platform does not expose this model - hence you cannot use workflow with this step against `https://detect.roboflow.com` API and you cannot use it in combination with `remote` execution when remote target is set to `hosted` (applies for Python package and `inference` HTTP container).

## Step parameters
* `type`: must be `YoloWorld` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `class_names` - must be reference to parameter or list of strings with names of classes to be detected - Yolo World 
model makes it possible to predict across classes that you pass in the runtime - so in each request to `workflows` you 
may detect different objects without model retraining. (required)
* `version` - allows to specify model version. It is optional parameter, but when value is given it must be one of 
[`s`, `m`, `l`]
* `confidence` - optional parameter to specify confidence threshold. If given - must be number in range `[0.0, 1.0]`

## Step outputs
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
* `prediction_type` - denoting `keypoint-detection` model