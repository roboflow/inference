Workflows allow you to define multi-step processes that run one or more models and return a result based on the output of the models.

You can create and deploy workflows in the cloud or in Inference.

To create an advanced workflow for use with Inference, you need to define a specification. A specification is a JSON document that states:

1. The version of workflows you are using.
2. The expected inputs.
3. The steps the workflow should run (i.e. models to run, filters to apply, etc.).
4. The expected output format.

In this guide, we walk through how to create a basic workflow that completes three steps:

1. Run a model to detect objects in an image.
2. Crops each region.
3. Runs a classification model to return a specific label for each cropped region.

This is a common use case for workflows, where you have a model that can detect abstract objects (i.e. car parts) and a classification model that can identify specific features (i.e. car parts).

You can use the guidance below as a template to learn the structure of workflows, or verbatim to create your own detect-then-classify workflows.

## Understanding Workflow Specifications

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
Additionally, defining **outputs**, it is allowed (since `v0.9.14`) to use wildcard selector
(`${type_of_element}.{name_of_element}.*`) with intention to extract all properties of given step.

### How can we refer between elements of specification?
To create a graph of computations, we need to define links between steps - in order to do it - we need to have a 
way to refer to specific elements. By convention, the following references are allowed: 
`${type_of_element}.{name_of_element}` and `${type_of_element}.{name_of_element}.{property}`.
Examples:
* `$inputs.image` - reference to an input called `image`
* `$steps.my_step.predictions` - reference to a step called `my_step` and its property `predictions`
Additionally, defining **outputs**, it is allowed (since `v0.9.14`) to use wildcard selector
(`${type_of_element}.{name_of_element}.*`) with intention to extract all properties of given step.

## Step #1: Define an Input

## Step #2: Define Processing Steps

## Step #3: Define an Output

## Step #4: Run Your Workflow

=== "Run Locally with Inference"

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

=== "Run in the Roboflow Cloud"

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

=== "Integrate with SDK (Advanced)"

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

## Next Steps

Now that you have created and run your first workflow, you can explore our other supported blocks and create a more complex workflow.

Refer to our [Supported Blocks](/workflows/supported_blocks/) documentation to learn more about what blocks are supported.