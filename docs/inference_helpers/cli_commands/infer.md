# Making predictions from your models

`inference infer` command offers an easy way to make predictions from your model based on your input images or video
files sending requests to `inference` server, depending on command configuration.

!!! Tip "Discovering command capabilities"

    To check detail of the command, run:
    
    ```bash
    inference infer --help
    ```

## Command details

`inference infer` takes input path / url and model version to produce predictions (and optionally make visualisation 
using `supervision`). You can also specify a host to run inference on our hosted inference server.

!!! note
    
    If you decided to use hosted inference server - make sure command `inference server start` was used first

!!! tip
    
    Roboflow API key can be provided via `ROBOFLOW_API_KEY` environment variable

## Examples

Below, you have usage examples illustrated.

### Predict On Local Image

This command is going to make a prediction from local image using selected model and print the prediction on 
the console.

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY}
```

To display visualised prediction use `-D` option. To save prediction and visualisation in a local directory,
use `-o {path_to_your_directory}` option. Those options work also in other modes.

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY} -D -o {path_to_your_output_directory}
```

### Predict On Image URL

```bash
inference infer -i https://[YOUR_HOSTED_IMAGE_URL] -m {your_project}/{version} --api-key {YOUR_API_KEY}
```

### Using Hosted API

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY} -h https://detect.roboflow.com
```

### Predict From Local Directory

```bash
inference infer -i {your_directory_with_images} -m {your_project}/{version} -o {path_to_your_output_directory} --api-key {YOUR_API_KEY}
```

### Predict On Video File

```bash
inference infer -i {path_to_your_video_file} -m {your_project}/{version} -o {path_to_your_output_directory} --api-key {YOUR_API_KEY}
```

### Configure The Visualization

Option `-c` can be provided with a path to `*.yml` file configuring `supervision` visualisation.
There are few pre-defined configs:
- `bounding_boxes` - with `BoxAnnotator` and `LabelAnnotator` annotators
- `bounding_boxes_tracing` - with `ByteTracker` and annotators (`BoxAnnotator`, `LabelAnnotator`)
- `masks` - with `MaskAnnotator` and `LabelAnnotator` annotators
- `polygons` - with `PolygonAnnotator` and `LabelAnnotator` annotators

Custom configuration can be created following the schema:
```yaml
annotators:
  - type: "bounding_box"
    params:
      thickness: 2
  - type: "label"
    params:
      text_scale: 0.5
      text_thickness: 2
      text_padding: 5
  - type: "trace"
    params:
      trace_length: 60
      thickness: 2
tracking:
  track_activation_threshold: 0.25
  lost_track_buffer: 30
  minimum_matching_threshold: 0.8
  frame_rate: 30
```
`annotators` field is a list of dictionaries with two keys: `type` and `param`. `type` points to 
name of annotator class:
```python
from supervision import *
ANNOTATOR_TYPE2CLASS = {
    "bounding_box": BoxAnnotator,
    "box": BoxAnnotator,
    "mask": MaskAnnotator,
    "polygon": PolygonAnnotator,
    "color": ColorAnnotator,
    "halo": HaloAnnotator,
    "ellipse": EllipseAnnotator,
    "box_corner": BoxCornerAnnotator,
    "circle": CircleAnnotator,
    "dot": DotAnnotator,
    "label": LabelAnnotator,
    "blur": BlurAnnotator,
    "trace": TraceAnnotator,
    "heat_map": HeatMapAnnotator,
    "pixelate": PixelateAnnotator,
    "triangle": TriangleAnnotator,
}
```
`param` is a dictionary of annotator constructor parameters (check them in 
[`supervision`](https://github.com/roboflow/supervision) docs - you would only be able
to use primitive values, classes and enums that are defined in constructors may not be possible
to resolve from yaml config).

`tracking` is an optional key that holds a dictionary with constructor parameters for
`ByteTrack`.

### Provide Inference Hyperparameters 

`-mc` parameter can be provided with path to `*.yml` file that specifies 
model configuration (like confidence threshold or IoU threshold). If given,
configuration will be used to initialise `InferenceConfiguration` object
from `inference_sdk` library. See [sdk docs](../inference_sdk.md) to discover
which options can be configured via `*.yml` file - configuration keys must match
with names of fields in `InferenceConfiguration` object.
