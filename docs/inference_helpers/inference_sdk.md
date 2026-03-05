# Inference Client

The `InferenceHTTPClient` enables you to interact with an [Inference Server]() over HTTP, both on the Roboflow platform and on device. It accepts images (local path, URL, `np.ndarray`, `PIL.Image`, or base64), directories of images, and video files. Async equivalents and batch/parallel support are available for most methods.

!!! tip

    Read our [Run Model on an Image](../quickstart/run_model_on_image.md) guide to learn how to run a model with the Inference Client.

## Quickstart

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"
result = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")
```

### AsyncIO client
```python
import asyncio
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"
loop = asyncio.get_event_loop()
result = loop.run_until_complete(
  CLIENT.infer_async(image_url, model_id="soccer-players-5fuqs/1")
)
```

## Configuration options

### Configuring with context managers

Methods `use_configuration(...)` and `use_model(...)` are designed to
work in context managers. **Once context manager is left - old config values are restored.**

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

custom_configuration = InferenceConfiguration(confidence_threshold=0.8)
# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

with CLIENT.use_configuration(custom_configuration):
    _ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

with CLIENT.use_model("soccer-players-5fuqs/1"):
    _ = CLIENT.infer(image_url)

# after leaving context manager - changes are reverted and `model_id` is still required
_ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")
```

As you can see - `model_id` is required to be given for prediction method only when default model is not configured.
--8<-- "include/model_id.md"

### Setting the configuration once and using till next change

Methods `configure(...)` and `select_model(...)` are designed alter the client
state and will be preserved until next change.

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

custom_configuration = InferenceConfiguration(confidence_threshold=0.8)
# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.configure(custom_configuration)
CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

# custom configuration still holds
CLIENT.select_model(model_id="soccer-players-5fuqs/1")
_ = CLIENT.infer(image_url)

# custom configuration and selected model - still holds
_ = CLIENT.infer(image_url)
```

One may also initialise in `chain` mode:

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(api_url="http://localhost:9001", api_key="ROBOFLOW_API_KEY") \
    .select_model("soccer-players-5fuqs/1")
```

### Overriding `model_id` for specific call

`model_id` can be overriden for specific call

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(api_url="http://localhost:9001", api_key="ROBOFLOW_API_KEY") \
    .select_model("soccer-players-5fuqs/1")

_ = CLIENT.infer(image_url, model_id="another-model/1")
```

## Parallel / Batch inference

You may want to predict against multiple images at single call. There are two parameters of `InferenceConfiguration`
that specifies batching and parallelism options:
- `max_concurrent_requests` - max number of concurrent requests that can be started
- `max_batch_size` - max number of elements that can be injected into single request

Thanks to that the following improvements can be achieved:
- if you run inference container with API on prem on powerful GPU machine - setting `max_batch_size` properly
may bring performance / throughput benefits
- if you run inference against hosted Roboflow API - setting `max_concurrent_requests` will cause multiple images
being served at once bringing performance / throughput benefits
- combination of both options can be beneficial for clients running inference container with API on cluster of machines,
then the load of single node can be optimised and parallel requests to different nodes can be made at a time

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
predictions = CLIENT.infer([image_url] * 5, model_id="soccer-players-5fuqs/1")

print(predictions)
```

Methods that support batching / parallelism:
- `infer(...)` and `infer_async(...)`
- `ocr_image(...)` and `ocr_image_async(...)` (enforcing `max_batch_size=1`)
- `detect_gazes(...)` and `detect_gazes_async(...)`
- `get_clip_image_embeddings(...)` and `get_clip_image_embeddings_async(...)`


## Client for core models

`InferenceHTTPClient` supports core models hosted via `inference`. Part of the models can be used on the Roboflow
hosted inference platform (use `https://infer.roboflow.com` as url), other are possible to be deployed locally (usually
local server will be available under `http://localhost:9001`).

!!! tip

    Install `inference-cli` package to easily run `inference` API locally
    ```bash
    pip install inference-cli
    inference server start
    ```

### Clip

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.get_clip_image_embeddings(inference_input="./my_image.jpg")  # single image request
CLIENT.get_clip_image_embeddings(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
CLIENT.get_clip_text_embeddings(text="some")  # single text request
CLIENT.get_clip_text_embeddings(text=["some", "other"])  # other text request
CLIENT.clip_compare(
    subject="./my_image.jpg",
    prompt=["fox", "dog"],
)
```

`CLIENT.clip_compare(...)` method allows to compare different combination of `subject_type` and `prompt_type`:

- `(image, image)`
- `(image, text)`
- `(text, image)`
- `(text, text)`
  Default mode is `(image, text)`.

!!! tip

    Check out async methods for Clip model:
    ```python
    from inference_sdk import InferenceHTTPClient

    CLIENT = InferenceHTTPClient(
        api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
        api_key="ROBOFLOW_API_KEY"
    )

    async def see_async_method():
      await CLIENT.get_clip_image_embeddings_async(inference_input="./my_image.jpg")  # single image request
      await CLIENT.get_clip_image_embeddings_async(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
      await CLIENT.get_clip_text_embeddings_async(text="some")  # single text request
      await CLIENT.get_clip_text_embeddings_async(text=["some", "other"])  # other text request
      await CLIENT.clip_compare_async(
          subject="./my_image.jpg",
          prompt=["fox", "dog"],
      )
    ```

### DocTR

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.ocr_image(inference_input="./my_image.jpg")  # single image request
CLIENT.ocr_image(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
```

!!! tip

    Check out async methods for DocTR model:
    ```python
    from inference_sdk import InferenceHTTPClient

    CLIENT = InferenceHTTPClient(
        api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
        api_key="ROBOFLOW_API_KEY"
    )

    async def see_async_method():
      await CLIENT.ocr_image(inference_input="./my_image.jpg")  # single image request
    ```

### Gaze

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # only local hosting supported
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.detect_gazes(inference_input="./my_image.jpg")  # single image request
CLIENT.detect_gazes(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
```

!!! tip

    Check out async methods for Gaze model:
    ```python
    from inference_sdk import InferenceHTTPClient

    CLIENT = InferenceHTTPClient(
        api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
        api_key="ROBOFLOW_API_KEY"
    )

    async def see_async_method():
      await CLIENT.detect_gazes(inference_input="./my_image.jpg")  # single image request
    ```


## Model Weights Download

When using a self-hosted Inference server, you can pre-load models to download and cache weights before running inference:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY"
)

# Pre-load the model (downloads weights to server cache)
client.load_model(model_id="rfdetr-base")
```

Alternatively, running a first inference will trigger the download automatically.

For workflows, you should also pre-load all models used in the workflow and run the workflow once to cache its definition.

You can verify which models are loaded on the server:

```python
loaded_models = client.list_loaded_models()
print(f"Loaded models: {loaded_models}")
```

!!! tip

    Read more about [weights caching, persistent storage, and Docker configuration](../using_inference/offline_weights_download.md).

## Inference against stream

One may want to infer against video or directory of images - and that modes are supported in `inference-client`

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
for frame_id, frame, prediction in CLIENT.infer_on_stream("video.mp4", model_id="soccer-players-5fuqs/1"):
    # frame_id is the number of frame
    # frame - np.ndarray with video frame
    # prediction - prediction from the model
    pass

for file_path, image, prediction in CLIENT.infer_on_stream("local/dir/", model_id="soccer-players-5fuqs/1"):
    # file_path - path to the image
    # frame - np.ndarray with video frame
    # prediction - prediction from the model
    pass
```

## What is actually returned as prediction?

`inference_client` returns plain Python dictionaries that are responses from model serving API. Modification
is done only in context of `visualization` key that keep server-generated prediction visualisation (it
can be transcoded to the format of choice) and in terms of client-side re-scaling.

## Methods to control `inference` server

### Getting server info

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.get_server_info()
```

### Listing loaded models

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.list_loaded_models()
```

!!! tip

    This method has async equivaluent: `list_loaded_models_async()`


### Getting specific model description

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.get_model_description(model_id="some/1", allow_loading=True)
```

If `allow_loading` is set to `True`: model will be loaded as side-effect if it is not already loaded.
Default: `True`.

!!! tip

    This method has async equivaluent: `get_model_description_async()`

### Loading model

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.load_model(model_id="some/1", set_as_default=True)
```

The pointed model will be loaded. If `set_as_default` is set to `True`: after successful load, model
will be used as default model for the client. Default value: `False`.

!!! tip

    This method has async equivaluent: `load_model_async()`

### Unloading model

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.unload_model(model_id="some/1")
```

Sometimes (to avoid OOM at server side) - unloading model will be required.


!!! tip

    This method has async equivaluent: `unload_model_async()`

### Unloading all models

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
CLIENT.unload_all_models()
```

!!! tip

    This method has async equivaluent: `unload_all_models_async()`


## Inference `workflows`

!!! tip

    Use `inference-cli` to run a local container with HTTP API:
    ```
    inference server start
    ```

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.run_workflow(
    specification={
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "InferenceParameter", "name": "my_param"},
        ],
        # ...
    },
    # OR
    # workspace_name="my_workspace_name",
    # workflow_id="my_workflow_id",

    images={
        "image": "url or your np.array",
    },
    parameters={
        "my_param": 37,
    },
)
```

Please note that either `specification` is provided with specification of workflow as described
[here](../workflows/definitions.md) or
both `workspace_name` and `workflow_id` are given to use workflow predefined in Roboflow app. `workspace_name`
can be found in Roboflow APP URL once browser shows the main panel of workspace.

!!! warning "Server-side caching of Workflow definitions"

    When you use `run_workflow(...)` method with `workspace_name` and `workflow_id`,
    the server will cache the definition for 15 minutes. If you change the definition in Workflows UI and re-run the
    method, you may not see the change. To force processing without cache, pass `use_cache=False` as a parameter of
    `run_workflow(...)` method.


!!! tip "Workflows profiling"

    You may request a profiler trace of your Workflow execution from the server by passing
    `enable_profiling=True` parameter to `run_workflow(...)` method. If server configuration enables traces exposure,
    you will be able to find a JSON file with trace in a directory specified by `profiling_directory` parameter of
    `InferenceConfiguration` - by default it is `inference_profiling` directory in your current working directory.
    The traces can be directly loaded and rendered in Google Chrome - navigate into `chrome://tracing` in your
    browser and hit "load" button.


## Details about client configuration

`inference-client` provides `InferenceConfiguration` dataclass to hold whole configuration.

```python
from inference_sdk import InferenceConfiguration
```

Overriding fields in this config changes the behaviour of client (and API serving model). Specific fields are
used in specific contexts. In particular:

### Classification model:

- `visualize_predictions`: flag to enable / disable visualisation
- `confidence_threshold` as `confidence`
- `stroke_width`: width of stroke in visualisation
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)
- `active_learning_target_dataset` - making inference from specific model (let's say `project_a/1`), when we want
to save data in another project `project_b` - the latter should be pointed to by this parameter. **Please remember that
you cannot use different type of models in `project_a` and `project_b` - if that is the case - data will not be
registered)**
- `source` Optional string to set a "source" attribute on the inference call; if using model monitoring, this will get logged with the inference request so you can filter/query inference requests coming from a particular source. e.g. to identify which application, system, or deployment is making the request.
- `source_info` Optional string to set additional "source_info" attribute on the inference call; e.g. to identify a sub component in an app.

### Object detection model:

- `visualize_predictions`: flag to enable / disable visualisation
- `visualize_labels`: flag to enable / disable labels visualisation if visualisation is enabled
- `confidence_threshold` as `confidence`
- `class_filter` to filter out list of classes
- `class_agnostic_nms`: flag to control whether NMS is class-agnostic
- `fix_batch_size`
- `iou_threshold`: to dictate NMS IoU threshold
- `stroke_width`: width of stroke in visualisation
- `max_detections`: max detections to return from model
- `max_candidates`: max candidates to post-processing from model
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)
- `source` Optional string to set a "source" attribute on the inference call; if using model monitoring, this will get logged with the inference request so you can filter/query inference requests coming from a particular source. e.g. to identify which application, system, or deployment is making the request.
- `source_info` Optional string to set additional "source_info" attribute on the inference call; e.g. to identify a sub component in an app.
- `active_learning_target_dataset` - making inference from specific model (let's say `project_a/1`), when we want
to save data in another project `project_b` - the latter should be pointed to by this parameter. **Please remember that
you cannot use different type of models in `project_a` and `project_b` - if that is the case - data will not be
registered)**

### Keypoints detection model:

- `visualize_predictions`: flag to enable / disable visualisation
- `visualize_labels`: flag to enable / disable labels visualisation if visualisation is enabled
- `confidence_threshold` as `confidence`
- `keypoint_confidence_threshold` as (`keypoint_confidence`) - to filter out detected keypoints
  based on model confidence
- `class_filter` to filter out list of object classes
- `class_agnostic_nms`: flag to control whether NMS is class-agnostic
- `fix_batch_size`
- `iou_threshold`: to dictate NMS IoU threshold
- `stroke_width`: width of stroke in visualisation
- `max_detections`: max detections to return from model
- `max_candidates`: max candidates to post-processing from model
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)
- `source` Optional string to set a "source" attribute on the inference call; if using model monitoring, this will get logged with the inference request so you can filter/query inference requests coming from a particular source. e.g. to identify which application, system, or deployment is making the request.
- `source_info` Optional string to set additional "source_info" attribute on the inference call; e.g. to identify a sub component in an app.
- `active_learning_target_dataset` - making inference from specific model (let's say `project_a/1`), when we want
to save data in another project `project_b` - the latter should be pointed to by this parameter. **Please remember that
you cannot use different type of models in `project_a` and `project_b` - if that is the case - data will not be
registered)**

### Instance segmentation model:

- `visualize_predictions`: flag to enable / disable visualisation
- `visualize_labels`: flag to enable / disable labels visualisation if visualisation is enabled
- `confidence_threshold` as `confidence`
- `class_filter` to filter out list of classes
- `class_agnostic_nms`: flag to control whether NMS is class-agnostic
- `fix_batch_size`
- `iou_threshold`: to dictate NMS IoU threshold
- `stroke_width`: width of stroke in visualisation
- `max_detections`: max detections to return from model
- `max_candidates`: max candidates to post-processing from model
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `mask_decode_mode`
- `tradeoff_factor`
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)
- `source` Optional string to set a "source" attribute on the inference call; if using model monitoring, this will get logged with the inference request so you can filter/query inference requests coming from a particular source. e.g. to identify which application, system, or deployment is making the request.
- `source_info` Optional string to set additional "source_info" attribute on the inference call; e.g. to identify a sub component in an app.
- `active_learning_target_dataset` - making inference from specific model (let's say `project_a/1`), when we want
to save data in another project `project_b` - the latter should be pointed to by this parameter. **Please remember that
you cannot use different type of models in `project_a` and `project_b` - if that is the case - data will not be
registered)**

### Configuration of client
- `output_visualisation_format`: one of (`VisualisationResponseFormat.BASE64`, `VisualisationResponseFormat.NUMPY`,
  `VisualisationResponseFormat.PILLOW`) - given that server-side visualisation is enabled - one may choose what
  format should be used in output
- `image_extensions_for_directory_scan`: while using `CLIENT.infer_on_stream(...)` with local directory
  this parameter controls type of files (extensions) allowed to be processed -
  default: `["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]`
- `client_downsizing_disabled`: set to `False` if you want to perform client-side downsizing - default `True`.
  Client-side scaling is only supposed to down-scale (keeping aspect-ratio) the input for inference -
  to utilise internet connection more efficiently (but for the price of images manipulation / transcoding).
  Model input size information will be used to determine the target size; if not available,
  `default_max_input_size` will be in use.
- `max_concurrent_requests` - max number of concurrent requests that can be started
- `max_batch_size` - max number of elements that can be injected into single request
- `workflow_run_retries_enabled` - flag that decides if transient errors in Workflows executions should be retried.
Defaults to `true` and the default can be altered with environment variable called `WORKFLOW_RUN_RETRIES_ENABLED`

### Configuration of Workflows execution

- `profiling_directory`: parameter specify the location where Workflows profiler traces are saved. By default, it is
`./inference_profiling` directory.
