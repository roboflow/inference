# `InferenceHTTPClient`

`InferenceHTTPClient` was created to make it easy for users to consume HTTP API exposed by `inference` server. You
can think of it, as a friendly wrapper over `requests` that you can use, instead of creating calling logic on
your own.

## 🔥 quickstart

```python
from inference_sdk import InferenceHTTPClient

image_url = "https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg"

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)
predictions = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

print(predictions)
```

## What are the client capabilities?
* Executing inference for models hosted at Roboflow platform (use client version `v0`)
* Executing inference for models hosted in local (or on-prem) docker images with `inference` HTTP API
* Works against single image (given as a local path, URL, `np.ndarray` or `PIL.Image`)
* Minimalistic batch inference implemented (you can pass multiple images)
* Implemented inference from video file and directory with images

## Why client has two modes - `v0` and `v1`?
We are constantly improving our `infrence` package - initial version (`v0`) is compatible with
models deployed at Roboflow platform (task types: `classification`, `object-detection`, `instance-segmentation` and
`keypoints-detection`)
are supported. Version `v1` is available in locally hosted Docker images with HTTP API. 

Locally hosted `inference` server exposes endpoints for model manipulations, but those endpoints are not available
at the moment for models deployed at Roboflow platform.

`api_url` parameter passed to `InferenceHTTPClient` will decide on default client mode - URLs with `*.roboflow.com`
will be defaulted to version `v0`.

Usage of model registry control methods with `v0` clients will raise `WrongClientModeError`.

## How I can adjust `InferenceHTTPClient` to work in my use-case?
There are few ways on how configuration can be altered:

### configuring with context managers
Methods `use_configuration(...)`, `use_api_v0(...)`, `use_api_v1(...)`, `use_model(...)` are designed to
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
with CLIENT.use_api_v0():
    _ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

with CLIENT.use_configuration(custom_configuration):
    _ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

with CLIENT.use_model("soccer-players-5fuqs/1"):
    _ = CLIENT.infer(image_url)

# after leaving context manager - changes are reverted and `model_id` is still required
_ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")
```

As you can see - `model_id` is required to be given for prediction method only when default model is not configured.

### Setting the configuration once and using till next change
Methods `configure(...)`, `select_api_v0(...)`, `select_api_v1(...)`, `select_model(...)` are designed alter the client
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
CLIENT.select_api_v0()
_ = CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

# API v0 still holds
CLIENT.configure(custom_configuration)
CLIENT.infer(image_url, model_id="soccer-players-5fuqs/1")

# API v0 and custom configuration still holds
CLIENT.select_model(model_id="soccer-players-5fuqs/1")
_ = CLIENT.infer(image_url)

# API v0, custom configuration and selected model - still holds
_ = CLIENT.infer(image_url)
```

One may also initialise in `chain` mode:

```python
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(api_url="http://localhost:9001", api_key="ROBOFLOW_API_KEY") \
    .select_api_v0() \
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

## Batch inference
You may want to predict against multiple images at single call. It is possible, but so far - client-side
batching is implemented in naive way (sequential requests to API) - stay tuned for future improvements.

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

## Methods to control `inference` server (in `v1` mode only)

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

If `allow_loading` is set to `True` - model will be loaded as side-effect if it is not already loaded.
Default: `True`.


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

The pointed model will be loaded. If `set_as_default` is set to `True` - after successful load, model
will be used as default model for the client. Default value: `False`.


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
[test_postprocessing.py](..%2F..%2Ftests%2Finference_client%2Funit_tests%2Fhttp%2Futils%2Ftest_postprocessing.py)

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


## Details about client configuration

`inference-client` provides `InferenceConfiguration` dataclass to hold whole configuration.

```python
from inference_sdk import InferenceConfiguration
```

Overriding fields in this config changes the behaviour of client (and API serving model). Specific fields are
used in specific contexts. In particular:

### Inference in `v0` mode
The following fields are passed to API
* `confidence_threshold` (as `confidence`) - to alter model thresholding
* `keypoint_confidence_threshold` as (`keypoint_confidence`) - to filter out detected keypoints
based on model confidence
* `format` - to visualise on server side - use `image` (but then you loose prediction details from response)
* `visualize_labels` (as `labels`) - used in visualisation to show / hide labels for classes
* `mask_decode_mode`
* `tradeoff_factor`
* `max_detections` - max detections to return from model
* `iou_threshold` (as `overlap`) - to dictate NMS IoU threshold
* `stroke_width` - width of stroke in visualisation
* `count_inference` as `countinference`
* `service_secret`
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`, 
`disable_preproc_static_crop` to alter server-side pre-processing


### Classification model in `v1` mode:
* `visualize_predictions` - flag to enable / disable visualisation
* `confidence_threshold` as `confidence`
* `stroke_width` - width of stroke in visualisation
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`, 
`disable_preproc_static_crop` to alter server-side pre-processing


### Object detection model in `v1` mode:
* `visualize_predictions` - flag to enable / disable visualisation
* `visualize_labels` - flag to enable / disable labels visualisation if visualisation is enabled
* `confidence_threshold` as `confidence`
* `class_filter` to filter out list of classes
* `class_agnostic_nms` - flag to control whether NMS is class-agnostic
* `fix_batch_size`
* `iou_threshold` - to dictate NMS IoU threshold
* `stroke_width` - width of stroke in visualisation
* `max_detections` - max detections to return from model
* `max_candidates` - max candidates to post-processing from model
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`, 
`disable_preproc_static_crop` to alter server-side pre-processing


### Keypoints detection model in `v1` mode:
* `visualize_predictions` - flag to enable / disable visualisation
* `visualize_labels` - flag to enable / disable labels visualisation if visualisation is enabled
* `confidence_threshold` as `confidence`
* `keypoint_confidence_threshold` as (`keypoint_confidence`) - to filter out detected keypoints
based on model confidence
* `class_filter` to filter out list of object classes
* `class_agnostic_nms` - flag to control whether NMS is class-agnostic
* `fix_batch_size`
* `iou_threshold` - to dictate NMS IoU threshold
* `stroke_width` - width of stroke in visualisation
* `max_detections` - max detections to return from model
* `max_candidates` - max candidates to post-processing from model
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`, 
`disable_preproc_static_crop` to alter server-side pre-processing


### Instance segmentation model in `v1` mode:
* `visualize_predictions` - flag to enable / disable visualisation
* `visualize_labels` - flag to enable / disable labels visualisation if visualisation is enabled
* `confidence_threshold` as `confidence`
* `class_filter` to filter out list of classes
* `class_agnostic_nms` - flag to control whether NMS is class-agnostic
* `fix_batch_size`
* `iou_threshold` - to dictate NMS IoU threshold
* `stroke_width` - width of stroke in visualisation
* `max_detections` - max detections to return from model
* `max_candidates` - max candidates to post-processing from model
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`, 
`disable_preproc_static_crop` to alter server-side pre-processing
* `mask_decode_mode`
* `tradeoff_factor`

### Configuration of client
* `output_visualisation_format` - one of (`VisualisationResponseFormat.BASE64`, `VisualisationResponseFormat.NUMPY`, 
`VisualisationResponseFormat.PILLOW`) - given that server-side visualisation is enabled - one may choose what
format should be used in output
* `image_extensions_for_directory_scan` - while using `CLIENT.infer_on_stream(...)` with local directory
this parameter controls type of files (extensions) allowed to be processed - 
default: `["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]`
* `client_downsizing_disabled` - set to `True` if you want to avoid client-side downsizing - default `False`.
Client-side scaling is only supposed to down-scale (keeping aspect-ratio) the input for inference -
to utilise internet connection more efficiently (but for the price of images manipulation / transcoding).
If model registry endpoint is available (mode `v1`) - model input size information will be used, if not:
`default_max_input_size` will be in use.
