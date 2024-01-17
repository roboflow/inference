# Inference Client

The `InferenceHTTPClient` enables you to interact with Inference over HTTP.

You can use this client to run models hosted:

1. On the Roboflow platform (use client version `v0`), and;
2. On device with Inference.

For models trained at Roboflow platform, client accepts the following inputs:

- A single image (Given as a local path, URL, `np.ndarray` or `PIL.Image`);
- Multiple images;
- A directory of images, or;
- A video file.

For core model - client exposes dedicated methods to be used, but standard image loader used accepts
file paths, URLs, `np.ndarray` and `PIL.Image` formats. Apart from client version (`v0` or `v1`) - options
provided via configuration are used against models trained at the platform, not the core models.

The client returns a dictionary of predictions for each image or frame.

!!! tip

    Read our [Run Model on an Image](/quickstart/run_model_on_image) guide to learn how to run a model with the Inference Client.

## Client for core models

`InferenceHTTPClient` now supports core models hosted via `inference`. Part of the models can be used at Roboflow hosted
inference platform (use `https://infer.roboflow.com` as url), other are possible to be deployed locally (usually
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

### CogVLM

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # only local hosting supported
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.prompt_cogvlm(
    visual_prompt="./my_image.jpg",
    text_prompt="So - what is your final judgement about the content of the picture?",
    chat_history=[("I think the image shows XXX", "You are wrong - the image shows YYY")], # optional parameter
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

## Configuration options (used for models trained at Roboflow platform)

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
{% include 'model_id.md' %}

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

If `allow_loading` is set to `True`: model will be loaded as side-effect if it is not already loaded.
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

The pointed model will be loaded. If `set_as_default` is set to `True`: after successful load, model
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

# <<<<<<< HEAD:docs/inference_helpers/inference_sdk.md

The following fields are passed to API

- `confidence_threshold` (as `confidence`) - to alter model thresholding
- `keypoint_confidence_threshold` as (`keypoint_confidence`) - to filter out detected keypoints
  based on model confidence
- `format`: to visualise on server side - use `image` (but then you loose prediction details from response)
- `visualize_labels` (as `labels`) - used in visualisation to show / hide labels for classes
- `mask_decode_mode`
- `tradeoff_factor`
- `max_detections`: max detections to return from model
- `iou_threshold` (as `overlap`) - to dictate NMS IoU threshold
- `stroke_width`: width of stroke in visualisation
- `count_inference` as `countinference`
- `service_secret`
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)
  > > > > > > > dfec32274e82c99ca74fce696538ad1522c1f187:docs/inference_sdk/http_client.md

The following fields are passed to API

- `confidence_threshold` (as `confidence`) - to alter model thresholding
- `keypoint_confidence_threshold` as (`keypoint_confidence`) - to filter out detected keypoints
  based on model confidence
- `format`: to visualise on server side - use `image` (but then you loose prediction details from response)
- `visualize_labels` (as `labels`) - used in visualisation to show / hide labels for classes
- `mask_decode_mode`
- `tradeoff_factor`
- `max_detections`: max detections to return from model
- `iou_threshold` (as `overlap`) - to dictate NMS IoU threshold
- `stroke_width`: width of stroke in visualisation
- `count_inference` as `countinference`
- `service_secret`
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for instance while testing model)

### Classification model in `v1` mode:

- `visualize_predictions`: flag to enable / disable visualisation
- `confidence_threshold` as `confidence`
- `stroke_width`: width of stroke in visualisation
- `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
- `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for
  instance while testing model)

* `visualize_predictions`: flag to enable / disable visualisation
* `confidence_threshold` as `confidence`
* `stroke_width`: width of stroke in visualisation
* `disable_preproc_auto_orientation`, `disable_preproc_contrast`, `disable_preproc_grayscale`,
  `disable_preproc_static_crop` to alter server-side pre-processing
* `disable_active_learning` to prevent Active Learning feature from registering the datapoint (can be useful for instance while testing model)

### Object detection model in `v1` mode:

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

### Keypoints detection model in `v1` mode:

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

### Instance segmentation model in `v1` mode:

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

### Configuration of client

- `output_visualisation_format`: one of (`VisualisationResponseFormat.BASE64`, `VisualisationResponseFormat.NUMPY`,
  `VisualisationResponseFormat.PILLOW`) - given that server-side visualisation is enabled - one may choose what
  format should be used in output
- `image_extensions_for_directory_scan`: while using `CLIENT.infer_on_stream(...)` with local directory
  this parameter controls type of files (extensions) allowed to be processed -
  default: `["jpg", "jpeg", "JPG", "JPEG", "png", "PNG"]`
- `client_downsizing_disabled`: set to `True` if you want to avoid client-side downsizing - default `False`.
  Client-side scaling is only supposed to down-scale (keeping aspect-ratio) the input for inference -
  to utilise internet connection more efficiently (but for the price of images manipulation / transcoding).
  If model registry endpoint is available (mode `v1`) - model input size information will be used, if not:
  `default_max_input_size` will be in use.

## FAQs

## Why does the Inference client have two modes (`v0` and `v1`)?

We are constantly improving our `infrence` package - initial version (`v0`) is compatible with
models deployed at Roboflow platform (task types: `classification`, `object-detection`, `instance-segmentation` and
`keypoints-detection`)
are supported. Version `v1` is available in locally hosted Docker images with HTTP API.

Locally hosted `inference` server exposes endpoints for model manipulations, but those endpoints are not available
at the moment for models deployed at Roboflow platform.

`api_url` parameter passed to `InferenceHTTPClient` will decide on default client mode - URLs with `*.roboflow.com`
will be defaulted to version `v0`.

Usage of model registry control methods with `v0` clients will raise `WrongClientModeError`.
