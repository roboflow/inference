# Understanding Workflows

!!! important

    This document talks about how Workflows work in depth. This guide is recommended for advanced users.

## How to create workflow specification?

### Workflow specification deep dive

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

* `inputs` - the section where we define all parameters that can be passed in the execution time by `inference` user.
* `steps` - the section where we define computation steps, their interconnections, connections to `inputs` and `outputs`.
* `outputs` - the section where we define all fields that needs to be rendered in the final result

### How can we refer between elements of specification?
To create a graph of computations, we need to define links between steps - in order to do it - we need to have a  way to refer to specific elements. By convention, the following references are allowed:  `${type_of_element}.{name_of_element}` and `${type_of_element}.{name_of_element}.{property}`.

Examples:

* `$inputs.image` - reference to an input called `image`
* `$steps.my_step.predictions` - reference to a step called `my_step` and its property `predictions`
Additionally, defining **outputs**, it is allowed (since `v0.9.14`) to use wildcard selector
(`${type_of_element}.{name_of_element}.*`) with intention to extract all properties of given step.

In the code, we usually call references **selectors**.

### How can we define `inputs`?
At the moment, the compiler supports two types of inputs `InferenceParameter` and `InferenceImage`.

#### `InferenceImage`

This input is reserved to represent image or list of images. Definition format:
```json
{"type": "InferenceImage", "name": "my_image"}
```
When creating `InferenceImage` you do not point a specific image - you just create a placeholder that will be linked with other element of the graph. This placeholder will be substituted with actual image when you run the workflow  graph and provide input parameter called `my_image` that can be `np.ndarray` or other formats that `inference` support, like:

```json
{
  "type": "url",
  "value": "https://here.is/my/image.jpg"
}
```

### `InferenceParameter`
Similar to `InferenceImage` - `InferenceParameter` creates a placeholder for a parameters that can be used in runtime to alter execution of workflow graph.
```json
{"type": "InferenceParameter", "name": "confidence_threshold", "default_value": 0.5}
```
`InferenceParameters` may be optionally defined with default values that will be used, if no actual parameter  of given name is present in user-defined input while executing the workflow graph. Type of parameter is not explicitly defined, but will be checked in runtime, prior to execution based on types of parameters that steps using this parameters can accept.

### How can we define `steps`?
Compiler supports multiple type of steps (that will be described later), but let's see how to define a simple one, that would be responsible for making prediction from object-detection model:
```json
{
    "type": "ObjectDetectionModel",
    "name": "my_object_detection",
    "image": "$inputs.image",
    "model_id": "yolov8n-640"
}
```
You can see that the step must have its type associated (that's how we link JSON document elements into code definitions) and name (unique within all steps). Another required parameters are `image` and `model_id`.

In case of `image` - we use reference to the input - that's how we create a link between parameter that will be provided in runtime and computational step. Steps parameters can be also provided as predefined values (like `model_id` in thiscase). Majority of parameters can be defined both as references to inputs (or outputs of other steps) and predefined values.

### How can we define `outputs`?

Definition of single output looks like that:
```json
{"type": "JsonField", "name": "predictions", "selector": "$steps.step_1.predictions"}
```
it defines a single output dictionary key (of name `predictions`) that will be created. `selector` field creates a link between step output and result. In this case, selector points `step_1` and its property - `predictions`.

Additionally, optional parameter `coordinates_system` can be defined with one of two values (`"own", "parent"`). This parameter defaults to `parent` and describe the coordinate system of detections that should be used. This setting is only important in case of more complicated graphs (where we crop based on predicted detections and later on make another detections on each and every crop).

### Example
In the following example, we create a pipeline that at first makes classification first. Based on results (the top class), `step_2` decides which object detection model to use (if model predicts car, `step_3` will be executed, `step_4` will be used otherwise).

Result is build from the outputs of all models. Always one of field `step_3_predictions` and `step_4_predictions` will be empty due to conditional execution.

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

Let's imagine a scenario when we have a graph definition that requires inference from object detection model on input  image. For each image that we have as an input - there will be most likely several detections. There is nothing that prevents us to do something with those detections. For instance, we can crop original image to extract RoIs with objects that the model detected. For each crop, we may then apply yet another, specialised object detection model to  detect lower resolution details. As you probably know, when `inference` makes prediction, it outputs the coordinates of detections scaled to the size of input image. 

But in this example, the input image is unknown when we start the process - those will be inferred by first model. To make it possible to combine predictions, we introduced `parent_id` identifier of prediction. It will be randomly generated string or name of input element that is responsible for certain prediction. 

In our example, each detection from first model will be assigned unique identifier (`detection_id`). This identifier will be a `parent_id` for each prediction that is made based on the crop originated in detection. What is more, each output can be `coordinates_system` parameter deciding how to present the result. If `parent` coordinates mode is selected - detections made against crop will be translated to the coordinates of original image that was submitted. Thanks to that, results can be easily overlay on the input image (for instance using `supervision` library). 
