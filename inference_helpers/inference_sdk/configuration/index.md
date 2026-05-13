# Configuration

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
