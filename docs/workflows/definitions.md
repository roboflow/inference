# Understanding Workflows Definitions syntax

In Roboflow Workflows, the Workflow Definition is the internal "programming language". It provides a structured 
way to define how different blocks interact, specifying the necessary inputs, outputs, and configurations. 
By using this syntax, users can create workflows without UI.

Let's start from examining the Workflow Definition created in [this tutorial](./create_and_run.md) and
analyse it step by step.

??? Tip "Workflow definition"
    
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
          "name": "model",
          "default_value": "yolov8n-640"
        }
      ],
      "steps": [
        {
          "type": "roboflow_core/roboflow_object_detection_model@v1",
          "name": "model",
          "images": "$inputs.image",
          "model_id": "$inputs.model"
        },
        {
          "type": "roboflow_core/dynamic_crop@v1",
          "name": "dynamic_crop",
          "images": "$inputs.image",
          "predictions": "$steps.model.predictions"
        },
        {
          "type": "roboflow_core/roboflow_classification_model@v1",
          "name": "model_1",
          "images": "$steps.dynamic_crop.crops",
          "model_id": "dog-breed-xpaq6/1"
        },
        {
          "type": "roboflow_core/detections_classes_replacement@v1",
          "name": "detections_classes_replacement",
          "object_detection_predictions": "$steps.model.predictions",
          "classification_predictions": "$steps.model_1.predictions"
        },
        {
          "type": "roboflow_core/bounding_box_visualization@v1",
          "name": "bounding_box_visualization",
          "predictions": "$steps.detections_classes_replacement.predictions",
          "image": "$inputs.image"
        },
        {
          "type": "roboflow_core/label_visualization@v1",
          "name": "label_visualization",
          "predictions": "$steps.detections_classes_replacement.predictions",
          "image": "$steps.bounding_box_visualization.image"
        }
      ],
      "outputs": [
        {
          "type": "JsonField",
          "name": "detections",
          "coordinates_system": "own",
          "selector": "$steps.detections_classes_replacement.predictions"
        },
        {
          "type": "JsonField",
          "name": "visualisation",
          "coordinates_system": "own",
          "selector": "$steps.label_visualization.image"
        }
      ]
    }
    ```

## Version marker

Every Workflow Definition begins with the version parameter, which specifies the compatible version of the 
Workflows Execution Engine. Roboflow utilizes [Semantic Versioning](https://semver.org/) to manage these 
versions and maintains one version from each major release to ensure backward compatibility. 
This means that a workflow defined for Execution Engine version `1.0.0` will function with version `1.3.4` and other 
newer versions, but workflows created for more recent versions may not be compatible with earlier ones.

List of Execution Engine versions loaded on the Roboflow Hosted platform is available 
[here](https://detect.roboflow.com/workflows/execution_engine/versions).


## Inputs

Our example workflow specifies two inputs:
```json
[
    {
      "type": "WorkflowImage", "name": "image"
    },
    {
      "type": "WorkflowParameter", "name": "model", "default_value": "yolov8n-640"
    }
]
```
This entry in definition creates two placeholders that can be filled with data while running workflow. 

The first placeholder is named `image` and is of type `WorkflowImage`. This special input type is batch-oriented, 
meaning it can accept one or more images at runtime to be processed as a single batch. You can add multiple inputs 
of the type `WorkflowImage`, and it is expected that the data provided to these placeholders will contain 
the same number of elements. Alternatively, you can mix inputs of sizes `N` and 1, where `N` represents the number 
of elements in the batch.

The second placeholder is a straightforward `WorkflowParameter` called model. This type of input allows users to 
inject hyperparameters — such as model variants, confidence thresholds, and reference values — at runtime. The
value is not expected to be a batch of elements, so when you provide a list, it will be interpreted as list of 
elements, rather than batch of elements, each to be processed individually.

More details about the nature of batch-oriented data processing in workflows can be found 
[here](./workflow_execution.md).

### Generic batch-oriented inputs

Since Execution Engine `v1.3.0` (inference release `v0.27.0`), Workflows support
batch oriented inputs of any *[kind](/workflows/kinds/index.md)* and 
*[dimensionality](./workflow_execution.md#steps-interactions-with-data)*. 
This inputs are **not enforced for now**, but we expect that as the ecosystem grows, they will 
be more and more useful.

??? Tip "Defining generic batch-oriented inputs"

    If you wanted to replace the `WorkflowImage` input with generic batch-oriented input,
    use the following construction:
    
    ```json
    {
      "inputs": [
        {
          "type": "WorkflowBatchInput",
          "name": "image",
          "kind": ["image"]
        }
      ]
    }
    ```
    
    Additionally, if your image is supposed to sit at higher *dimensionality level*, 
    add `dimensionality` property:

    ```{ .json linenums="1" hl_lines="7" }
    {
      "inputs": [
        {
          "type": "WorkflowBatchInput",
          "name": "image",
          "kind": ["image"],
          "dimensionality": 2
        }
      ]
    }
    ```
    
    This will alter the expected format of `image` data in Workflow run -
    `dimensionality=2` enforces `image` to be nested batch of images - namely list 
    of list of images.


## Steps

As mentioned [here](./understanding.md), steps are instances of Workflow blocks connected with inputs and outputs 
of other steps to dictate how data flows through the workflow. Let's see example step definition:

```json
{
  "type": "roboflow_core/roboflow_object_detection_model@v1",
  "name": "model",
  "images": "$inputs.image",
  "model_id": "$inputs.model"
}
```

Two common properties for each step are `type` and `name`. Type tells which block to load and name gives the step 
unique identifier, based on which other steps may refer to output of given step.

Two remaining properties declare `selectors` (this is how we call references in Workflows) to inputs - `image` and
`model`. While running the workflow, data passed into those placeholders will be provided for block to process.

Our documentation showcases what is the structure of each block and provides examples of how each block can be 
used as workflow step. Explore our blocks collection [here](/workflows/blocks/index.md) where you can find what are 
block data inputs, outputs and configuration properties.

Input data bindings of blocks (like `images` property) can be filled with selectors to batch-oriented inputs and 
step outputs. Configuration properties of blocks (like `model_id`) usually can be filled with either values
hardcoded in workflow definition (they cannot be altered in runtime) or selectors to inputs of type `WorkflowParameter`.
For instance, valid definition can be obtained when `model_id` is either `"$inputs.image"` or `yolov8n-640`.

Let's see now how step outputs are referred as inputs of another step:
```json
{
  "type": "roboflow_core/dynamic_crop@v1",
  "name": "dynamic_crop",
  "images": "$inputs.image",
  "predictions": "$steps.model.predictions"
}
```
In this particular case, `predictions` property defines output of step named `model`. Construction of selector is
the following: `$steps.{step_name}.{step_output_name}`. Thanks to this reference, `model` step is connected with 
`dynamic_crop` and in runtime model predictions will be passed into dynamic crop and will be reference for image 
cropping procedure.

## Outputs

This section of Workflow Definition specifies how response from workflow execution looks like. Definitions of 
each response field looks like that:

```json
{
  "type": "JsonField",
  "name": "detections",
  "selector": "$steps.detections_classes_replacement.predictions"
}
```

The `selector` can reference either an input or a step output. Additionally, you can specify the `"coordinates_system"` 
property, which accepts two values: `"own"` or `"parent"`. This property is relevant for outputs that provide model 
detections and determines the coordinate system used for the detections. This becomes crucial when applying a 
secondary object detection model on image crops derived from predictions of a primary model. In such cases, 
the secondary model’s predictions are based on the coordinates of the crops, not the original input image. 
To ensure these coordinates are not translated back to the parent coordinate system, set 
`"coordinates_system": "own"` (`parent` is default option).

Additionally, outputs selectors support wildcards (`$steps.step_nane.*"`) to grab all outputs of specific step.

To fully understand how output structure is created - read about 
[data processing in Workflows](./workflow_execution.md).
