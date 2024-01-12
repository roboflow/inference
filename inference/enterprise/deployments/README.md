# Deployments compiler

> [!IMPORTANT] 
> We require a Roboflow Enterprise License to use this in production. See inference/enterpise/LICENSE.txt for details.


## Overview
We are under development of new feature that would allow clients to define the ML workflow in a declarative form
(JSON configuration or WYSIWYG UI) and let the `inference` care about all required computations. That goal can be
achieved thanks to the compilation and runtime engine that is created here.

The `deopoyment` module contains components capable to:
* parse the deployment specification (see: [schemas of configuration entities](./entities))
* validate the correctness of deployment specification (see: [validator module](./complier/validator.py))
* construct computational graph and validate its consistency prior to any computations (see: [graph parser](./complier/graph_parser.py))
* analyse runtime input parameter and link them with graph placeholders (see: [input validator](./complier/runtime_input_validator.py))
* execute the computation workflow (see: [execution engine](./complier/execution_engine.py))

![overview diagram](./assets/named_deployments_overview.jpg)

## How to create deployment specification?

### Deployment specification basics

Deployment specification is defined via a JSON document in the following format:
```json
{
  "specification": {
    "version": "1.0",
    "inputs": [],
    "steps": [],
    "outputs": []
  }
}
```

In general, we have three main elements of specification:
* `inputs` - the section where we define all parameters that can be passed in the execution time by `inference` user
* `steps` - the section where we define computation steps, their interconnections, connections to `inputs` and `outputs`
* `outputs` - the section where we define all fields that needs to be rendered in the final result

### How can we refer between elements of specification?
To create a graph of computations, we need to define links between steps - in order to do it - we need to have a 
way to refer to specific elements. By convention, the following references are allowed: 
`${type_of_element}.{name_of_element}` and `${type_of_element}.{name_of_element}.{property}`.
Examples:
* `$inputs.image` - reference to an input called `image`
* `$steps.my_step.predictions` - reference to a step called `my_step` and its property `predictions`

In the code, we usually call references **selectors**.

### How can we define `inputs`?
At the moment, the compiler supports two types of inputs `InferenceParameter` and `InferenceImage`.

#### `InferenceImage`
This input is reserved to represent image or list of images. Definition format:
```json
{"type": "InferenceImage", "name": "my_image"}
```
When creating `InferenceImage` you do not point a specific image - you just create a placeholder that will be linked
with other element of the graph. This placeholder will be substituted with actual image when you run the deployment 
graph and provide input parameter called `my_image` that can be `np.ndarray` or other formats that `inference` support,
like:
```json
{
  "type": "url",
  "value": "https://here.is/my/image.jpg"
}
```

### `InferenceParameter`
Similar to `InferenceImage` - `InferenceParameter` creates a placeholder for a parameters that can be used in runtime 
to alter execution of deployment graph.
```json
{"type": "InferenceParameter", "name": "confidence_threshold", "default_value": 0.5}
```
`InferenceParameters` may be optionally defined with default values that will be used, if no actual parameter 
of given name is present in user-defined input while executing the deployment graph. Type of parameter is not 
explicitly defined, but will be checked in runtime, prior to execution based on types of parameters that 
steps using this parameters can accept.

### How can we define `steps`?
Compiler supports multiple type of steps (that will be described later), but let's see how to define a simple one,
that would be responsible for making prediction from object-detection model:
```json
{
    "type": "ObjectDetectionModel",
    "name": "my_object_detection",
    "image": "$inputs.image",
    "model_id": "yolov8n-640"
}
```
You can see that the step must have its type associated (that's how we link JSON document elements into code definitions)
and name (unique within all steps). Another required parameters are `image` and `model_id`.
In case of `image` - we use reference to the input - that's how we create a link between parameter that will be provided
in runtime and computational step. Steps parameters can be also provided as predefined values (like `model_id` in this 
case). Majority of parameters can be defined both as references to inputs (or outputs of other steps) and predefined
values.

### How can we define `outputs`?
Definition of single output looks like that:
```json
{"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.predictions"}
```
it defines a single output dictionary key (of name `predictions`) that will be created. `selector` field creates a
link between step output and result. In this case, selector points `step_1` and its property - `predictions`.

Additionally, optional parameter `coordinates_system` can be defined with one of two values (`"own", "parent"`).
This parameter defaults to `parent` and describe the coordinate system of detections that should be used.
This setting is only important in case of more complicated graphs (where we crop based on predicted detections and
later on make another detections on each and every crop).

### Example
In the following example, we create a pipeline that at first makes classification first. Based on results
(the top class), `step_2` decides which object detection model to use (if model predicts car, `step_3` will be executed,
`step_4` will be used otherwise).
Result is build from the outputs of all models. Always one of field `step_3_predictions` and `step_4_predictions` will
be empty due to conditional execution.
```json
{
    "specification": {
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"}
        ],
        "steps": [
            {
                "type": "ClassificationModel",
                "name": "step_1",
                "image": "$inputs.image",
                "model_id": "vehicle-classification-eapcd/2",
                "confidence": 0.4
            },
            {
                "type": "Condition",
                "name": "step_2",
                "left": "$steps.step_1.top",
                "operator": "equal",
                "right": "Car",
                "step_if_true": "$steps.step_3",
                "step_if_false": "$steps.step_4"
            },
            {
                "type": "ObjectDetectionModel",
                "name": "step_3",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": 0.5,
                "iou_threshold": 0.4
            },
            {
                "type": "ObjectDetectionModel",
                "name": "step_4",
                "image": "$inputs.image",
                "model_id": "yolov8n-1280",
                "confidence": 0.5,
                "iou_threshold": 0.4
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "top_class", "selector": "$steps.step_1.top"},
            {"type": "JsonField", "name": "step_3_predictions", "selector": "$steps.step_3.predictions"},
            {"type": "JsonField", "name": "step_4_predictions", "selector": "$steps.step_4.predictions"}
        ]  
    }
}
```

### What kind of steps are available?

#### `ClassificationModel`
This step represents inference from multi-class classification model.

##### Step parameters
* `type`: must be `ClassificationModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 

##### Step outputs:
* `predictions` - details of predictions
* `top` - top class
* `confidence` - confidence of prediction
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `MultiLabelClassificationModel`
This step represents inference from multi-label classification model.

##### Step parameters
* `type`: must be `MultiLabelClassificationModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 

##### Step outputs:
* `predictions` - details of predictions
* `predicted_classes` - top classes
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `ObjectDetectionModel`
This step represents inference from object detection model.

##### Step parameters
* `type`: must be `ObjectDetectionModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 
* `class_agnostic_nms`: optional boolean flag to control NMS - can be selector to 
input parameter 
* `class_filter`: optional list of classes using as filter - can be selector to 
input parameter 
* `iou_threshold`: optional float value in range [0, 1] with NMS IoU threshold - can be selector to 
input parameter 
* `max_detections`: optional integer parameter of NMS - can be selector to input parameter 
* `max_candidates`: optional integer parameter of NMS - can be selector to input parameter 

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `KeypointsDetectionModel`
This step represents inference from keypoints detection model.

##### Step parameters
* `type`: must be `KeypointsDetectionModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 
* `class_agnostic_nms`: optional boolean flag to control NMS - can be selector to 
input parameter 
* `class_filter`: optional list of classes using as filter - can be selector to 
input parameter 
* `iou_threshold`: optional float value in range [0, 1] with NMS IoU threshold - can be selector to 
input parameter 
* `max_detections`: optional integer parameter of NMS - can be selector to input parameter 
* `max_candidates`: optional integer parameter of NMS - can be selector to input parameter 
* `keypoint_confidence`: optional float value in range [0, 1] with keypoints confidence threshold - can be selector to 
input parameter 

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `InstanceSegmentationModel`
This step represents inference from instance segmentation model.

##### Step parameters
* `type`: must be `InstanceSegmentationModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `model_id`: must be either valid Roboflow model ID or selector to  input parameter (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `disable_active_learning`: optional boolean flag to control Active Learning at request level - can be selector to 
input parameter 
* `confidence`: optional float value in range [0, 1] with threshold - can be selector to 
input parameter 
* `class_agnostic_nms`: optional boolean flag to control NMS - can be selector to 
input parameter 
* `class_filter`: optional list of classes using as filter - can be selector to 
input parameter 
* `iou_threshold`: optional float value in range [0, 1] with NMS IoU threshold - can be selector to 
input parameter 
* `max_detections`: optional integer parameter of NMS - can be selector to input parameter 
* `max_candidates`: optional integer parameter of NMS - can be selector to input parameter 
* `mask_decode_mode`: optional parameter of post-processing - can be selector to input parameter 
* `tradeoff_factor`: optional parameter of post-processing - can be selector to 
input parameter 

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `OCRModel`
This step represents inference from OCR model.

##### Step parameters
* `type`: must be `OCRModel` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)

##### Step outputs:
* `result` - details of predictions
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines
