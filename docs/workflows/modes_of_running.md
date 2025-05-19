# What are the options for running workflows?

There are few ways on how to run Workflow, including:

- Request to HTTP API (Roboflow Hosted API or self-hosted `inference` server) running Workflows Execution Engine

- Video processing using [InferencePipeline](../using_inference/inference_pipeline.md)

- `inference` Python package, where you can use Workflows Execution Engine directly in your Python app

## HTTP API request

This way of running Workflows is ideal for clients who:

- Want to use Workflows as a stand-alone, independent part of their systems.

- Maintain their main applications in languages other than Python.

- Prefer to offload compute-heavy tasks to dedicated servers.


Roboflow offers a hosted HTTP API that clients can use without needing their own infrastructure. 
Alternatively, the `inference` server (which can run Workflows) can be set up on-site if needed.

Running Workflows with Roboflow Hosted API has several limitations:

- Workflow runtime is limited to 20s

- Response payload is limited to 6MB, which means that some blocks (especially visualization ones) if used
in too large numbers, or with input images that are too large may result in failed request



Integrating via HTTP is simple: just send a [request](https://detect.roboflow.com/docs#/default/infer_from_predefined_workflow__workspace_name__workflows__workflow_id__post)
to the server. You can do this using a HTTP client library in your preferred programming language, 
leverage our Inference SDK in Python, or even use cURL. Explore the examples below to see how it’s done.

!!! example "HTTP integration"

    === "cURL"
        
        To run your workflow created in Roboflow APP with `cURL`, use the following command:

        ```bash
        curl --location 'https://detect.roboflow.com/infer/workflows/<your-workspace-name>/<your-workflow-id>' \
            --header 'Content-Type: application/json' \
            --data '{
            "api_key": "<YOUR-API-KEY>",
            "inputs": {
                "image": {"type": "url", "value": "https://your-image-url"},
                "parameter": "some-value"
            }
        }'
        ```
        
        Please note that:
        
        - `<your-workspace-name>`, `<your-workflow-id>`, `<YOUR-API-KEY>` must be replaced with actual values - 
        valid for your Roboflow account
        
        - keys of `inputs` dictionary are dictated by your Workflow, names may differ **dependent on 
        parameters you define**
    
        - values of `inputs` dictionary are also dependent on your Workflow definition - inputs declared as
        `WorkflowImage` have special structure - dictionary with `type` and `value` keys - using cURL your 
        options are `url` and `base64` as `type` - and value adjusted accordingly

    === "Inference SDK in Python (Roboflow Hosted API)"
        
        To run your workflow created in Roboflow APP with `InferenceClient`:
        
        ```python
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="<YOUR-API-KEY>",
        )
        
        result = client.run_workflow(
            workspace_name="<your-workspace-name>",
            workflow_id="<your-workflow-id>",
            images={
                "image": ["https://your-image-url", "https://your-other-image-url"]
            },
            parameters={
                "parameter": "some-value"
            },
        )
        ```

        Please note that:
        
        - `<your-workspace-name>`, `<your-workflow-id>`, `<YOUR-API-KEY>` must be replaced with actual values - 
        valid for your Roboflow account
        
        - method parameter named `images` is supposed to be filled with dictionary that contains names and values
        for all Workflow inputs declared as `WorkflowImage`. Names must match your Workflow definition,
        as value you can pass either `np.array`, `PIL.Image`, URL to your image, local path to your image
        or image in `base64` string. It is optional if Workflow does not define images as inputs. 
        
        - **Batch input for images is supported - simply pass list of images under given input name.**
    
        - method parameter named `parameters` is supposed to be filled with dictionary that contains names and values
        for all Workflow inputs of type `WorkflowParameter`. It's optional and must be filled according to Workflow
        definition.

        !!! note
            
            Please make sure you have `inference-sdk` package installed in your environment

    === "Inference SDK in Python (on-prem)"
        
        To run your workflow created in Roboflow APP with `InferenceClient`:
        
        ```python
        from inference_sdk import InferenceHTTPClient

        client = InferenceHTTPClient(
            api_url="http://127.0.0.1:9001",  # please modify that value according to URL of your server
            api_key="<YOUR-API-KEY>",
        )
        
        result = client.run_workflow(
            workspace_name="<your-workspace-name>",
            workflow_id="<your-workflow-id>",
            images={
                "image": ["https://your-image-url", "https://your-other-image-url"]
            },
            parameters={
                "parameter": "some-value"
            }    
        )
        ```

        Please note that:
        
        - `<your-workspace-name>`, `<your-workflow-id>`, `<YOUR-API-KEY>` must be replaced with actual values - 
        valid for your Roboflow account
        
        - method parameter named `images` is supposed to be filled with dictionary that contains names and values
        for all Workflow inputs declared as `WorkflowImage`. Names must match your Workflow definition,
        as value you can pass either `np.array`, `PIL.Image`, URL to your image, local path to your image
        or image in `base64` string. It is optional if Workflow does not define images as inputs.

        - **Batch input for images is supported - simply pass list of images under given input name.**
    
        - method parameter named `parameters` is supposed to be filled with dictionary that contains names and values
        for all Workflow inputs of type `WorkflowParameter`. It's optional and must be filled according to Workflow
        definition.

        !!! note
            
            - Please make sure you have `inference-sdk` package installed in your environment.
            
            - Easiest way to run `inference` server on-prem is to use `inference-cli` package command:
            ```bash
            inference server start
            ```


The above examples present how to run Workflow created and saved in Roboflow APP. It is also possible to
create and run workflow that is created from scratch and may not contain API-KEY gated blocks (for instance 
your own blocks). Then you should use the 
[following endpoint](https://detect.roboflow.com/docs#/default/infer_from_workflow_workflows_run_post) or
[Inference SDK](../inference_helpers/inference_sdk.md#inference-workflows) as showcased in docs.


## Video processing using `InferencePipeline`

For use cases involving video files or streams, we recommend using [InferencePipeline](../using_inference/inference_pipeline.md), which can run 
Workflows on each video frame.

This option is ideal for clients who:

- Need low-latency, high-throughput video processing.

- Design workflows with single-frame processing times that meet real-time requirements (though complex workflows 
might not be suitable for real-time processing)


Explore the example below to see how to combine `InferencePipeline` with Workflows.

!!! example "Integration with InferencePipeline"

    ```python
    from inference import InferencePipeline
    from inference.core.interfaces.camera.entities import VideoFrame

    def my_sink(result: dict, video_frame: VideoFrame):
        print(result) # here you can find dictionary with outputs from your workflow
        
    
    # initialize a pipeline object
    pipeline = InferencePipeline.init_with_workflow(
        api_key="<YOUR-API-KEY>",
        workspace_name="<your-workspace-name>",
        workflow_id="<your-workflow-id>",
        video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
        on_prediction=my_sink,
        image_input_name="image",  # this parameter holds the name of Workflow input that represents 
        # image to be processed - please ADJUST it to your Workflow Definition 
    )
    pipeline.start() #start the pipeline
    pipeline.join() #wait for the pipeline thread to finish
    ```

     Please note that:
        
    - `<your-workspace-name>`, `<your-workflow-id>`, `<YOUR-API-KEY>` must be replaced with actual values - 
    valid for your Roboflow account
    
    - your Workflow must accept video frames under `image` parameter - when multiple video streams are 
    given for processing, all collected video frames will be submitted in batch under `image` parameter  
    for workflow run. `image` parameter must be single batch oriented input of your workflow

    - additional (non-batch oriented) inputs for your workflow can be passed as parameter to `init_with_workflow(...)` 
    method see [docs](/reference/inference/core/interfaces/stream/inference_pipeline#inference.core.interfaces.stream.inference_pipeline.InferencePipeline.init_with_workflow)

    !!! note
        
        Make sure you have `inference` or `inference-gpu` package installed in your Python environment

## Batch processing using `inference-cli`

[`inference-cli`](../inference_helpers/inference_cli.md) is command-line wrapper library around `inference`. You can use it
to process your data using Workflows without writing a single line of code. You simply point the data to be processed,
select your Workflow and specify where results should be saved. Thanks to `inference-cli` you can process:

* individual images

* directories of images

* video files

!!! example "Processing directory of images"

    You can start the processing using the following command:

    ```bash
    inference workflows process-images-directory \
        -i {your_input_directory} \
        -o {your_output_directory} \[workflows.py](..%2F..%2Finference_cli%2Fworkflows.py)
        --workspace_name {your-roboflow-workspace-url} \
        --workflow_id {your-workflow-id} \
        --api-key {your_roboflow_api_key}
    ```

    As a result, in the directory specified in `-o` option you should be able to find:

    * sub-directories named after files in your original directory with `results.json` file that contain Worklfow 
    results and optionally additional `*.jpg` files with images created during Workflow execution

    * `aggregated_results.csv` file that contain concatenated results of Workflow execution for all input image file

    !!! note
        
        Make sure you have `inference` or `inference-cli` package installed in your Python environment


## Workflows in Python package

Workflows Compiler and Execution Engine are bundled with [`inference`](https://pypi.org/project/inference/) package.
Running Workflow directly may be ideal for clients who:

- maintain their applications in Python

- agree for resource-heavy computations directly in their app

- want to avoid additional latency and errors related to sending HTTP requests

- expect full control over Workflow execution

In this scenario, you are supposed to provide all required initialisation values for blocks used in your Workflow, what
makes this mode most technologically challenging, requiring you to understand handful of topics that we cover in 
developer guide.

Here you can find example on how to run simple workflow in Python code.

!!! example "Integration in Python"

    ```python
    from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
    from inference.core.managers.base import ModelManager
    from inference.core.workflows.core_steps.common.entities import StepExecutionMode
    from inference.core.env import MAX_ACTIVE_MODELS
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES
    
    # initialisation of Model registry to manage models load into memory 
    # (required by core blocks exposing Roboflow models)
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
    
    # workflow definition
    OBJECT_DETECTION_WORKFLOW = {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "model_id"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3},
        ],
        "steps": [
            {
                "type": "RoboflowObjectDetectionModel",
                "name": "detection",
                "image": "$inputs.image",
                "model_id": "$inputs.model_id",
                "confidence": "$inputs.confidence",
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "result", "selector": "$steps.detection.*"}
        ],
    }
    
    # example init parameters for blocks - dependent on set of blocks
    # used in your workflow
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "<YOUR-API-KEY>,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }

    # instance of Execution Engine - init(...) method invocation triggers
    # the compilation process
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # runing the workflow
    result = execution_engine.run(
        runtime_parameters={
            "image": [<your-image>],
            "model_id": "yolov8n-640",
        }
    )
    ```
