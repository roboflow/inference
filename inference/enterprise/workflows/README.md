# Workflows compiler

> [!IMPORTANT] 
> We require a Roboflow Enterprise License to use this in production. See inference/enterpise/LICENSE.txt for details.


## Overview
We are under development of new feature that would allow clients to define the ML workflow in a declarative form
(JSON configuration or WYSIWYG UI) and let the `inference` care about all required computations. That goal can be
achieved thanks to the compilation and runtime engine that is created here.

The `workflows` module contains components capable to:
* parse the workflow specification (see: [schemas of configuration entities](./entities))
* validate the correctness of workflows specification (see: [validator module](./complier/validator.py))
* construct computational graph and validate its consistency prior to any computations (see: [graph parser](./complier/graph_parser.py))
* analyse runtime input parameter and link them with graph placeholders (see: [input validator](./complier/runtime_input_validator.py))
* execute the computation workflow (see: [execution engine](./complier/execution_engine.py))

![overview diagram](./assets/workflows_overview.jpg)

## How `workflows` can be used?

### Behind Roboflow hosted API
```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

client.infer_from_workflow(
    specification={},  # workflow specification goes here
    images={},  # input images goes here
    parameters={},  # input parameters other than image goes here
)
```

### Behind `inference` HTTP API

Use `inference_cli` to start server
```bash
inference server start
```

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key="YOUR_API_KEY"
)

client.infer_from_workflow(
    specification={},  # workflow specification goes here
    images={},  # input images goes here
    parameters={},  # input parameters other than image goes here
)
```

### Integration with Python code

```python
from inference.enterprise.workflows.complier.core import compile_and_execute

IMAGE = ...
result = compile_and_execute(
    workflow_specification={},
    runtime_parameters={
        "image": IMAGE,
    },
    api_key="YOUR_API_KEY",
)
```

## How to create workflow specification?

### Workflow specification basics

Workflow specification is defined via a JSON document in the following format:
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
with other element of the graph. This placeholder will be substituted with actual image when you run the workflow 
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
to alter execution of workflow graph.
```json
{"type": "InferenceParameter", "name": "confidence_threshold", "default_value": 0.5}
```
`InferenceParameters` may be optionally defined with default values that will be used, if no actual parameter 
of given name is present in user-defined input while executing the workflow graph. Type of parameter is not 
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
![example pipeline](./assets/example_pipeline.jpg)
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

### The notion of parents in `workflows`
Let's imagine a scenario when we have a graph definition that requires inference from object detection model on input 
image. For each image that we have as an input - there will be most likely several detections. There is nothing that
prevents us to do something with those detections. For instance, we can crop original image to extract RoIs with objects
that the model detected. For each crop, we may then apply yet another, specialised object detection model to 
detect lower resolution details. As you probably know, when `inference` makes prediction, it outputs the coordinates
of detections scaled to the size of input image. But in this example, the input image is unknown when we start the 
process - those will be inferred by first model. To make it possible to combine predictions, we introduced `parent_id`
identifier of prediction. It will be randomly generated string or name of input element that is responsible for 
certain prediction. 

In our example, each detection from first model will be assigned unique identifier (`detection_id`). This identifier 
will be a `parent_id` for each prediction that is made based on the crop originated in detection. What is more,
each output can be `coordinates_system` parameter deciding how to present the result. If `parent` coordinates 
mode is selected - detections made against crop will be translated to the coordinates of original image that was
submitted. Thanks to that, results can be easily overlay on the input image (for instance using `supervision` library). 

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
input parameter. Default: `0.3`.
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
input parameter. Default: `0.3`.
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
input parameter. Default: `0.3`.
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

#### `Crop`
This step produces **dynamic** crops based on detections from detections-based model.

##### Step parameters
* `type`: must be `Crop` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `detections`: must be a reference to `predictions` property of steps: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`] (required)

##### Step outputs:
* `crops` - `image` cropped based on `detections`
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines


#### `Condition`
This step is responsible for flow-control in execution graph based on the condition defined in its body.
As for now, only capable to make conditions based on output of binary operators that takes two operands.

##### Step parameters
* `type`: must be `Condition` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `left`: left operand of `operator`, can be actual value, reference to input or step output (required)
* `right`: left operand of `operator`, can be actual value, reference to input or step output (required)
* `operator`: one of `equal`, `not_equal`, `lower_than`, `greater_than`, `lower_or_equal_than`, `greater_or_equal_than`
or `in` (required)
* `step_if_true`: reference to the step that will be executed if condition is true (required)
* `step_if_false`: reference to the step that will be executed if condition is false (required)


#### `DetectionFilter`
This step is responsible for filtering detections based predictions based on conditions defined.

##### Step parameters
* `type`: must be `DetectionFilter` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: reference to `predictions` output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`] (required)
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
and right operand is `reference_value`.
In case if `CompoundDetectionFilterDefinition`, logical operators `or`, `and` can be used to combine simple filters.
This let user define recursive structure of filters.

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `DetectionOffset`
This step is responsible for applying fixed offset on width and height of detections.


##### Step parameters
* `type`: must be `DetectionOffset` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: reference to `predictions` output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`] (required)
* `offset_x`: reference to input parameter of integer value for detection width offset (required)
* `offset_y`: reference to input parameter of integer value for detection height offset (required)

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines


#### `AbsoluteStaticCrop` and `RelativeStaticCrop`
Responsible for cropping RoIs from images - using absolute coordinates (integer pixel values) or relative coordinates
(fraction of width and height in range [0.0, 1.0]) respectively.

##### Step parameters
* `type`: must be `AbsoluteStaticCrop` / `RelativeStaticCrop` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `x_center`: OX center coordinate of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `y_center`: OY center coordinate of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `width`: width of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`
* `height`: height of crop or reference to `InputParameter` - must be integer for `AbsoluteStaticCrop`
or float in range [0.0, 1.0] in case of `RelativeStaticCrop`

##### Step outputs:
* `crops` - `image` cropped based on step parameters
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines

#### `ClipComparison`
Step to execute comparison of Clip embeddings between image and text.

##### Step parameters
* `type`: must be `ClipComparison` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `image`: must be a reference to input of type `InferenceImage` or `crops` output from steps executing cropping (
`Crop`, `AbsoluteStaticCrop`, `RelativeStaticCrop`) (required)
* `text`: reference to `InputParameter` of list of texts to compare against `image` using Clip model

##### Step outputs:
* `similarity` - for each element of `image` - list of float values representing similarity to each element of `text`
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines


#### `DetectionsConsensus`
Step that is meant to combine predictions from potentially multiple detections models. 
Steps checks for object presence (according to configurable criteria), combines detections and
decides on requested objects presence (based on overlap of predictions from different models). It works for 
`object-detection`, `instance-segmentation` and `keypoint-detection` models, but consensus output is only
applied at detections level. 

Step executes following operations (in order):
* get only the predictions from `classes_to_consider` (if specified)
* for every prediction finds predictions with max overlap from all other sources (at most one per source) that reaches 
`iou_threshold` 
* for each group of overlapping predictions from different sources - if the size of group is at least
`required_votes` and merged boxe meet `confidence` threshold - those are discarded from the pool of detections to be 
picked up and are merged into element of `predictions` output that can be called consensus output. 
`class_aware` parameter decides if class names matter while merging - should be `False` when different class names are 
produced by different models  but the visual concept that models predict is the same.
* merge is done based on `detections_merge_confidence_aggregation` and `detections_merge_coordinates_aggregation` 
parameters that control how to pick the merged box class, confidence and box coordinates
* once all elements of consensus outputs are ready, the step prepares `object_present` status
and `presence_confidence` that form a summary of consensus output. One may state `required_objects`
as integer or dict mapping class name to required instance of objects. In the final state, the step
logic will check if required number of objects (possibly from different classes) are detected in consensus
output. If that's the case - `object_present` field will be `True` and `presence_confidence` will be calculated
(using `presence_confidence_aggregation` method). Otherwise - `presence_confidence` will be an empty dict.
In the case of `class_aware=False`:
  * when `required_objects` is dict with class to count mapping - effective `required_objects` will be sum of dictionary 
  values
  * the `presence_confidence` will hold `any_object` key with confidence aggregated among all merged detections. 

![How consensus block works?](./assets/detection_consensus_step_diagram.jpg)

##### Step parameters
* `type`: must be `DetectionsConsensus` (required)
* `name`: must be unique within all steps - used as identifier (required)
* `predictions`: list of selectors pointing to outputs of detections models output of the detections model: [`ObjectDetectionModel`, 
`KeypointsDetectionModel`, `InstanceSegmentationModel`, `DetectionFilter`, `DetectionsConsensus`] (required, must contain at least 2 elements)
* `required_votes`: number of models that must agree on the detection - integer or selector pointing at
`InferenceParameter` (required)
* `class_aware`: flag deciding if class names are taken into account when finding overlapping bounding boxes
from multiple models and object presence check. Can be `bool` or selector to `InferenceParameter`. Default: `True`
* `iou_threshold`: optional float value in range [0, 1] with IoU threshold that must be meet to consider
two bounding boxes overlapping. Can be float or selector to `InferenceParameter`. Default: `0.3`.
* `confidence`: optional float value in range [0, 1] minimal confidence of **aggregated** detection that must be met to 
be taken into account in presence assessment and consensus procedure. For prior-consensus filtering - use
confidence threshold at model level or `DetectionsFilter`. Default: `0.0`.
* `classes_to_consider`: Optional list of classes to consider in consensus procedure.
Can be list of `str` or selector to `InferenceParameter`. Default: `None` - in this case 
classes filtering of predictions will not be enabled.
* `required_objects` - If given, it holds the number of objects that must be present in merged results, to assume that 
object presence is reached. Can be selector to `InferenceParameter`, integer value or dictionary with mapping of class name into
minimal number of merged detections of given class to assume consensus.
* `presence_confidence_aggregation` - mode dictating aggregation of confidence scores
and classes both in case of object presence deduction procedure. One of `average`, `max`, `min`. Default: `max`.
* `detections_merge_confidence_aggregation` - mode dictating aggregation of confidence scores
and classes both in case of boxes consensus procedure. 
One of `average`, `max`, `min`. Default: `average`. While using for merging overlapping boxes, 
against classes - `average` equals to majority vote, `max` - for the class of detection with max confidence,
`min` - for the class of detection with min confidence.
* `detections_merge_coordinates_aggregation` - mode dictating aggregation of bounding boxes. One of `average`, `max`, `min`. 
Default: `average`. `average` means taking mean from all boxes coordinates, `min` - taking smallest box, `max` - taking 
largest box.

##### Step outputs:
* `predictions` - details of predictions
* `image` - size of input image, that `predictions` coordinates refers to 
* `parent_id` - identifier of parent image / associated detection that helps to identify predictions with RoI in case
of multi-step pipelines (can be `undefined` if all sources of predictions give no prediction)
* `object_present` - for each input image, boolean flag with information whether or not
objects specified in config are present
* `presence_confidence` - for each input image, for each present class - aggregated confidence indicating presence
of objects

## Different modes of execution
Workflows can be executed in `local` environment, or `remote` environment can be used. `local` means that model steps
will be executed within the context of process running the code. `remote` will re-direct model steps into remote API
using HTTP requests to send images and get predictions back. 

When `workflows` are used directly, in Python code - `compile_and_execute(...)` and `compile_and_execute_async(...)`
functions accept `step_execution_mode` parameter that controls the execution mode.

Additionally, `max_concurrent_steps` parameter dictates how many steps in parallel can be executed. This will
improve efficiency of `remote` execution (up to the limits of remote API capacity) and can improve `local` execution
if `model_manager` instance is capable of running parallel requests (only using extensions from 
`inference.enterprise.parallel`).

There are environmental variables that controls `workflows` behaviour:
* `DISABLE_WORKFLOW_ENDPOINTS` - disabling workflows endpoints from HTTP API
* `WORKFLOWS_STEP_EXECUTION_MODE` - with values `local` and `remote` allowed to control how `workflows` are executed
in `inference` HTTP container
* `WORKFLOWS_REMOTE_API_TARGET` - with values `hosted` and `self-hosted` allowed - to point API to be used in `remote`
execution mode
* `LOCAL_INFERENCE_API_URL` will be used if `WORKFLOWS_REMOTE_API_TARGET=self-hosted` and 
`WORKFLOWS_STEP_EXECUTION_MODE=remote`
* `WORKFLOWS_MAX_CONCURRENT_STEPS` - max concurrent steps to be allowed by `workflows` executor
* `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE` - max batch size for requests into remote API made when `remote`
execution mode is chosen
* `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS` - max concurrent requests to be possible in scope of
single step execution when `remote` execution mode is chosen
