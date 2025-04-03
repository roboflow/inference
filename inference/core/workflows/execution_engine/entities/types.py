from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, Field, StringConstraints
from typing_extensions import Annotated


class Kind(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    docs: Optional[str] = Field(default=None)
    serialised_data_type: Optional[str] = Field(
        default=None,
        description="Provides Python type hint for data format that should guide "
        "external clients on how to produce / consume serialised data of specific kind.",
    )
    internal_data_type: Optional[str] = Field(
        default=None,
        description="Provides type hint regarding internal data representation that specific "
        "kind translates into when Workflow is run by Execution Engine. Relevant for "
        "blocks developers.",
    )

    def __hash__(self) -> int:
        return (
            self.name.__hash__()
            + self.description.__hash__()
            + self.docs.__hash__()
            + self.serialised_data_type.__hash__()
            + self.internal_data_type.__hash__()
        )


REFERENCE_KEY = "reference"
SELECTED_ELEMENT_KEY = "selected_element"
KIND_KEY = "kind"
DIMENSIONALITY_OFFSET_KEY = "dimensionality_offset"
DIMENSIONALITY_REFERENCE_PROPERTY_KEY = "dimensionality_reference_property"
SELECTOR_POINTS_TO_BATCH_KEY = "selector_points_to_batch"

WILDCARD_KIND_DOCS = """
This is a special kind that represents Any value - which is to be used by default if 
kind annotation is not specified. It will not tell the compiler what is to be expected
as a specific value in runtime, but at the same time, it makes it possible to run the 
workflow when we do not know or do not care about types. 

**Important note:** Usage of this kind reduces execution engine capabilities to predict 
problems with workflow and make those problems to be visible while running the workflow.
"""
WILDCARD_KIND = Kind(
    name="*",
    description="Equivalent of any element",
    docs=WILDCARD_KIND_DOCS,
    serialised_data_type="Any",
    internal_data_type="Any",
)
IMAGE_KIND_DOCS = """
This is the representation of image in `workflows`. Underlying data type has different internal and
external representation. As an input we support:

!!! note "Update added in Execution Engine `v1.2.0`"

    `video_metadata` added as optional property - should be injected in context of video processing to 
    provide necessary context for blocks dedicated to video processing.
    

* `np.ndarray` image when Workflows Execution Engine is used directly in `inference` python package (array can be
provided in a form of dictionary presented below, if `video_metadata` is intended to be injected)

* dictionary compatible with [inference image utils](https://inference.roboflow.com/reference/inference/core/utils/image_utils/):

```python
{
    "type": "url",   # there are different types supported, including np arrays and PIL images
    "value": "..."   # value depends on `type`,
    "video_metadata": {  
        # optional - can be added in context of video processing - introduced in 
        # Execution Engine `v1.2.0` - released in inference `v0.23.0`
        "video_identifier": "rtsp://some.com/stream1",
        "comes_from_video_file": False,
        "fps": 23.99,
        "measured_fps": 20.05,
        "frame_number": 24,
        "frame_timestamp": "2024-08-21T11:13:44.313999", 
    }  
}
```

Whe using Workflows Execution Engine exposed behind `inference` server, two most common `type` values are `base64` and 
`url`.

Internally, [`WorkflowImageData`](/workflows/internal_data_types/#workflowimagedata) is used. If you are a
Workflow block developer, we advise checking out [usage guide](/workflows/internal_data_types/#workflowimagedata).
"""
IMAGE_KIND = Kind(
    name="image",
    description="Image in workflows",
    docs=IMAGE_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="WorkflowImageData",
)


VIDEO_METADATA_KIND_DOCS = """

!!! warning "Deprecated since Execution Engine `v1.2.0`"

    `inference` maintainers decided to sunset `video_metadata` kind in favour of
    auxiliary metadata added to `image` kind. 

This is representation of metadata that describe images that come from videos.  
It is helpful in cases of stateful video processing, as the metadata may bring 
pieces of information that are required by specific blocks.

The kind has different internal end external representation. As input we support:
```
{
    "video_identifier": "rtsp://some.com/stream1",
    "comes_from_video_file": False,
    "fps": 23.99,
    "measured_fps": 20.05,
    "frame_number": 24,
    "frame_timestamp": "2024-08-21T11:13:44.313999", 
}   
```
Internally, [`VideoMetadata`](/workflows/internal_data_types/#videometadata) is used. If you are a
Workflow block developer, we advise checking out [usage guide](/workflows/internal_data_types/#videometadata).
"""

VIDEO_METADATA_KIND = Kind(
    name="video_metadata",
    description="Video image metadata",
    docs=VIDEO_METADATA_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="VideoMetadata",
)

ROBOFLOW_MODEL_ID_KIND_DOCS = """
This kind represents value specific for Roboflow platform. At the platform, models are
identified with special strings in the format: `<project_name>/<version>`. You should
expect this value to be provided once this kind is used. In some special cases, Roboflow 
platform accepts model alias as model id which will not conform provided schema. List
of aliases can be found [here](https://inference.roboflow.com/quickstart/aliases/).  
"""
ROBOFLOW_MODEL_ID_KIND = Kind(
    name="roboflow_model_id",
    description="Roboflow model id",
    docs=ROBOFLOW_MODEL_ID_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

ROBOFLOW_PROJECT_KIND_DOCS = """
This kind represents value specific for Roboflow platform. At the platform, each project has 
unique name and the value behind this kind represent this name. To learn more 
on how to kick-off with Roboflow project - visit [this](https://blog.roboflow.com/getting-started-with-roboflow/) page. 
"""
ROBOFLOW_PROJECT_KIND = Kind(
    name="roboflow_project",
    description="Roboflow project name",
    docs=ROBOFLOW_PROJECT_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

ROBOFLOW_API_KEY_KIND_DOCS = """
This kind represents API key that grants access to Roboflow platform.
To learn more about Roboflow API keys visit [this](https://docs.roboflow.com/api-reference/authentication) 
page.
"""
ROBOFLOW_API_KEY_KIND = Kind(
    name="roboflow_api_key",
    description="Roboflow API key",
    docs=ROBOFLOW_API_KEY_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

FLOAT_ZERO_TO_ONE_KIND_DOCS = """
This kind represents float value from 0.0 to 1.0. 

Examples:
```
0.1
0.4
0.999
```
"""
FLOAT_ZERO_TO_ONE_KIND = Kind(
    name="float_zero_to_one",
    description="`float` value in range `[0.0, 1.0]`",
    docs=FLOAT_ZERO_TO_ONE_KIND_DOCS,
    serialised_data_type="float",
    internal_data_type="float",
)

LIST_OF_VALUES_KIND_DOCS = """
This kind represents Python list of Any values.

Examples:
```
["a", 1, 1.0]
["a", "b", "c"]
```
"""
LIST_OF_VALUES_KIND = Kind(
    name="list_of_values",
    description="List of values of any type",
    docs=LIST_OF_VALUES_KIND_DOCS,
    serialised_data_type="List[Any]",
    internal_data_type="List[Any]",
)

EMBEDDING_KIND_DOCS = """
This kind represents a vector embedding. It is a list of floating point numbers.

Embeddings are used in various machine learning tasks like clustering, classification,
and similarity search. They are used to represent data in a continuous, low-dimensional space.

Typically, vectors that are close to each other in the embedding space are considered similar.
"""
EMBEDDING_KIND = Kind(
    name="embedding",
    description="A list of floating point numbers representing a vector embedding.",
    docs=EMBEDDING_KIND_DOCS,
    serialised_data_type="List[float]",
    internal_data_type="List[float]",
)

RGB_COLOR_KIND_DOCS = """
This kind represents RGB color as a tuple (R, G, B).

Examples:
```
(128, 32, 64)
(255, 255, 255)
```
"""

RGB_COLOR_KIND = Kind(
    name="rgb_color",
    description="RGB color",
    docs=RGB_COLOR_KIND_DOCS,
    serialised_data_type="Tuple[int, int, int]",
    internal_data_type="Tuple[int, int, int]",
)

IMAGE_KEYPOINTS_KIND_DOCS = """
The kind represents image keypoints that are detected by classical Computer Vision methods.
Underlying representation is serialised OpenCV KeyPoint object.

Examples:
```
{
    "pt": (2.429290294647217, 1197.7939453125),
    "size": 1.9633429050445557,
    "angle": 183.4322509765625,
    "response": 0.03325376659631729,
    "octave": 6423039,
    "class_id": -1
}
``` 
"""

IMAGE_KEYPOINTS_KIND = Kind(
    name="image_keypoints",
    description="Image keypoints detected by classical Computer Vision method",
    docs=IMAGE_KEYPOINTS_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="dict",
)

SERIALISED_PAYLOADS_KIND_DOCS = f"""
This value represents list of serialised values. Each serialised value is either
string or bytes - if something else is provided - it will be attempted to be serialised 
to JSON.

Examples:
```
["some", b"other", {{"my": "dictionary"}}]
```

This kind is to be used in combination with sinks blocks and serializers blocks.
Serializer should output value of this kind which shall then be accepted by sink.
"""
SERIALISED_PAYLOADS_KIND = Kind(
    name="serialised_payloads",
    description="Serialised element that is usually accepted by sink",
    docs=SERIALISED_PAYLOADS_KIND_DOCS,
    serialised_data_type="List[Union[str, dict]]",
    internal_data_type="List[Union[str, bytes, dict]]",
)

BYTES_KIND = Kind(
    name="bytes",
    description="This kind represent bytes",
    docs="This kind represent bytes. Default serializer turns bytes into base64-encoded string and "
    "this is the source of different data representation.",
    serialised_data_type="str",
    internal_data_type="bytes",
)


BOOLEAN_KIND_DOCS = """
This kind represents boolean value - `True` or `False`
"""
BOOLEAN_KIND = Kind(
    name="boolean",
    description="Boolean flag",
    docs=BOOLEAN_KIND_DOCS,
    serialised_data_type="bool",
    internal_data_type="bool",
)

INTEGER_KIND_DOCS = """
Examples:
```
1
2
```
"""
INTEGER_KIND = Kind(
    name="integer",
    description="Integer value",
    docs=INTEGER_KIND_DOCS,
    serialised_data_type="int",
    internal_data_type="int",
)
STRING_KIND_DOCS = """
Examples:
```
"my string value"
```
"""
STRING_KIND = Kind(
    name="string",
    description="String value",
    docs=STRING_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

TOP_CLASS_KIND_DOCS = f"""
The kind represent top classes predicted by classification model.

Example:
```
"car"
```
"""
TOP_CLASS_KIND = Kind(
    name="top_class",
    description="String value representing top class predicted by classification model",
    docs=TOP_CLASS_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

FLOAT_KIND_DOCS = """
Example:
```
1.3
2.7
```
"""
FLOAT_KIND = Kind(
    name="float",
    description="Float value",
    docs=FLOAT_KIND_DOCS,
    serialised_data_type="float",
    internal_data_type="float",
)
DICTIONARY_KIND_DOCS = """
This kind represent a value of any Python dict.

Examples:
```
{"my_key", "my_value"}
``` 
"""
DICTIONARY_KIND = Kind(
    name="dictionary",
    description="Dictionary",
    serialised_data_type="dict",
    internal_data_type="dict",
)

CLASSIFICATION_PREDICTION_KIND_DOCS = """
This kind represent predictions from Classification Models.

Examples:
```
# in case of multi-class classification
{
    "image": {"height": 128, "width": 256},
    "predictions": [{"class_name": "A", "class_id": 0, "confidence": 0.3}],
    "top": "A",
    "confidence": 0.3,
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}

# in case of multi-label classification
{
    "image": {"height": 128, "width": 256},
    "predictions": {
        "a": {"confidence": 0.3, "class_id": 0},
        "b": {"confidence": 0.3, "class_id": 1},
    }
    "predicted_classes": ["a", "b"],
    "parent_id": "some",
    "prediction_type": "classification",
    "inference_id": "some",
    "root_parent_id": "some",
}
```
"""
CLASSIFICATION_PREDICTION_KIND = Kind(
    name="classification_prediction",
    description="Predictions from classifier",
    docs=CLASSIFICATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="dict",
)


DETECTION_KIND_DOCS = """
This kind represents single detection in prediction from a model that detects multiple elements
(like object detection or instance segmentation model). It is represented as a tuple
that is created from `sv.Detections(...)` object while iterating over its content. `workflows`
utilises `data` property of `sv.Detections(...)` to keep additional metadata which will be available
in the tuple. Some properties may not always be present. Take a look at documentation of 
`object_detection_prediction`, `instance_segmentation_prediction`, `keypoint_detection_prediction`
kinds to discover which additional metadata are available.

More technical details about 
[iterating over `sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/#supervision.detection.core.Detections)
"""

DETECTION_KIND = Kind(
    name="detection",
    description="Single element of detections-based prediction (like `object_detection_prediction`)",
    docs=DETECTION_KIND_DOCS,
    serialised_data_type="Tuple[list, Optional[list], Optional[float], Optional[float], Optional[int], dict]",
    internal_data_type="Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[float], Optional[int], dict]",
)


POINT_KIND = Kind(
    name="point",
    description="Single point in 2D",
    docs=None,
    serialised_data_type="Tuple[int, int]",
    internal_data_type="Tuple[int, int]",
)

CONTOURS_KIND_DOCS = """
This kind represents a value of a list of numpy arrays where each array represents contour points.

Example:
```
[
    np.array([[10, 10],
              [20, 20],
              [30, 30]], dtype=np.int32),
    np.array([[50, 50],
              [60, 60],
              [70, 70]], dtype=np.int32)
]
```
"""

CONTOURS_KIND = Kind(
    name="contours",
    description="List of numpy arrays where each array represents contour points",
    docs=CONTOURS_KIND_DOCS,
    serialised_data_type="List[list]",
    internal_data_type="List[np.ndarray]",
)

ZONE_KIND = Kind(
    name="zone",
    description="Definition of polygon zone",
    docs="List of points defining polygon zone in format [(x, y)]",
    serialised_data_type="List[Tuple[int, int]]",
    internal_data_type="List[Tuple[int, int]]",
)

NUMPY_ARRAY_KIND = Kind(
    name="numpy_array",
    description="Numpy array",
    docs="Any np.ndarray object",
    serialised_data_type="list",
    internal_data_type="np.ndarray",
)

OBJECT_DETECTION_PREDICTION_KIND_DOCS = """
This kind represents single object detection prediction in form of 
[`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.

Example:
```
sv.Detections(
    xyxy=array([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    mask=None, 
    confidence=array([    0.84955,     0.74344,     0.45636,     0.86537]), 
    class_id=array([2, 7, 2, 0]), 
    tracker_id=None, 
    data={
        'class_name': array(['car', 'truck', 'car', 'car'], dtype='<U13')
        'detection_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'inference_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'prediction_type': array([
            'object-detection', 'object-detection', 
            'object-detection', 'object-detection'
        ], dtype='<U16'),
        'root_parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'root_parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'scaling_relative_to_parent': array([1, 1, 1, 1]),
        'scaling_relative_to_root_parent': array([1, 1, 1, 1]),
    }
)
```   

As you can see, we have extended the standard set of metadata for predictions maintained by `supervision`.
Adding this metadata is needed to ensure compatibility with blocks from `roboflow_core` plugin.

The design of metadata is suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image


**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="object_detection_prediction",
    description="Prediction with detected bounding boxes in form of sv.Detections(...) object",
    docs=OBJECT_DETECTION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="sv.Detections",
)


INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS = """
This kind represents single instance segmentation prediction in form of 
[`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.

Example:
```
sv.Detections(
    xyxy=array([[        127,         189,         322,         303]]), 
    mask=array([
        [[False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        ...,
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False],
        [False, False, False, ..., False, False, False]]
    ]), 
    confidence=array([    0.95898]), 
    class_id=array([6]), 
    tracker_id=None, 
    data={
        'class_name': array(['G'], dtype='<U1'),
        'detection_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'parent_id': array(['image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640]]),
        'inference_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'prediction_type': array(['instance-segmentation'], dtype='<U16'),
        'root_parent_id': array(['image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0]]),
        'root_parent_dimensions': array([[425, 640]]),
        'parent_coordinates': array([[0, 0]]),
        'parent_dimensions': array([[425, 640]]),
        'scaling_relative_to_parent': array([1]),
        'scaling_relative_to_root_parent': array([1]),
    }
)
```

As you can see, we have extended the standard set of metadata for predictions maintained by `supervision`.
Adding this metadata is needed to ensure compatibility with blocks from `roboflow_core` plugin.

The design of metadata is suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [InstanceSegmentationInferenceResponse](https://detect.roboflow.com/docs)
"""
INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="instance_segmentation_prediction",
    description="Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object",
    docs=INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="sv.Detections",
)


KEYPOINT_DETECTION_PREDICTION_KIND_DOCS = """
This kind represents single keypoints prediction in form of 
[`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.

Example:
```
sv.Detections(
    xyxy=array([[        127,         189,         322,         303]]), 
    mask=None, 
    confidence=array([    0.95898]), 
    class_id=array([6]), 
    tracker_id=None, 
    data={
        'class_name': array(['G'], dtype='<U1'),
        'detection_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'parent_id': array(['image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640]]),
        'inference_id': array(['51dfa8d5-261c-4dcb-ab30-9aafe9b52379'], dtype='<U36'),
        'prediction_type': array(['instance-segmentation'], dtype='<U16'),
        'root_parent_id': array(['image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0]]),
        'root_parent_dimensions': array([[425, 640]]),
        'parent_coordinates': array([[0, 0]]),
        'parent_dimensions': array([[425, 640]]),
        'scaling_relative_to_parent': array([1]),
        'scaling_relative_to_root_parent': array([1]),
        'keypoints_class_name': array(),  # variable length array of type object - one 1D array of str for each box
        'keypoints_class_id': array(),  # variable length array of type object - one 1D array of int for each box
        'keypoints_confidence': array(),  # variable length array of type object - one 1D array of float for each box
        'keypoints_xy': array(),  # variable length array of type object - one 2D array for bbox with (x, y) coords
    }
)
```

Prior to [sv.Keypoints(...)](https://supervision.roboflow.com/0.21.0/keypoint/core/) we introduced 
keypoints detection based on [`sv.Detections(...)`](https://supervision.roboflow.com/latest/detection/core/) object.
The decision was suboptimal so we would need to revert in the future, but for now this is the format of
data for keypoints detection. 

The design of metadata is also suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

* `keypoints_class_name` array of variable size 1D arrays of string with key points class names

* `keypoints_class_id` array of variable size 1D arrays of int with key points class ids

* `keypoints_confidence` array of variable size 1D arrays of float with key points confidence

* `keypoints_xy` array of variable size 2D arrays of coordinates of keypoints in `(x, y)` format

**SERIALISATION:**

Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output, but **loosing keypoints details** - which can be recovered if output 
JSON field is parsed. Entity details: [KeypointsDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="keypoint_detection_prediction",
    description="Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object",
    docs=KEYPOINT_DETECTION_PREDICTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="sv.Detections",
)

QR_CODE_DETECTION_KIND_DOCS = """
This kind represents batch of predictions regarding QR codes location and data their provide.

Example:
```
sv.Detections(
    xyxy=array([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    mask=None, 
    confidence=array([    1.0, 1.0, 1.0, 1.0]), 
    class_id=array([2, 7, 2, 0]), 
    tracker_id=None, 
    data={
        'class_name': array(['qr_code', 'qr_code', 'qr_code', 'qr_code'], dtype='<U13')
        'detection_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'inference_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'prediction_type': array([
            'qrcode-detection', 'qrcode-detection', 
            'qrcode-detection', 'qrcode-detection'
        ], dtype='<U16'),
        'root_parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'root_parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'scaling_relative_to_parent': array([1, 1, 1, 1]),
        'scaling_relative_to_root_parent': array([1, 1, 1, 1]),
        'data': np.array(['qr-code-1-data', 'qr-code-2-data', 'qr-code-3-data', 'qr-code-4-data'])
    }
)
```

As you can see, we have extended the standard set of metadata for predictions maintained by `supervision`.
Adding this metadata is needed to ensure compatibility with blocks from `roboflow_core` plugin.

The design of metadata is suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

* `data` - extracted QR code

**SERIALISATION:**
Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
QR_CODE_DETECTION_KIND = Kind(
    name="qr_code_detection",
    description="Prediction with QR code detection",
    docs=QR_CODE_DETECTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="sv.Detections",
)

BAR_CODE_DETECTION_KIND_DOCS = """
This kind represents batch of predictions regarding barcodes location and data their provide.

Example:
```
sv.Detections(
    xyxy=array([
       [        865,       153.5,        1189,       422.5],
       [      192.5,        77.5,       995.5,       722.5],
       [        194,          82,         996,         726],
       [        460,         333,         704,         389]]
    ), 
    mask=None, 
    confidence=array([    1.0, 1.0, 1.0, 1.0]), 
    class_id=array([2, 7, 2, 0]), 
    tracker_id=None, 
    data={
        'class_name': array(['barcode', 'barcode', 'barcode', 'barcode'], dtype='<U13')
        'detection_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'image_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'inference_id': array([
            '51dfa8d5-261c-4dcb-ab30-9aafe9b52379', 'c0c684d1-1e30-4880-aedd-29e67e417264'
            '8cfc543b-9cfe-493b-b5ad-77afed7bee83', 'c0c684d1-1e30-4880-aedd-38e67e441454'
        ], dtype='<U36'),
        'prediction_type': array([
            'barcode-detection', 'barcode-detection', 
            'barcode-detection', 'barcode-detection'
        ], dtype='<U16'),
        'root_parent_id': array(['image.[0]', 'image.[0]', 'image.[0]', 'image.[0]'], dtype='<U9'),
        'root_parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'root_parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'parent_coordinates': array([[0, 0], [0, 0], [0, 0], [0, 0]]),
        'parent_dimensions': array([[425, 640], [425, 640], [425, 640], [425, 640]]),
        'scaling_relative_to_parent': array([1, 1, 1, 1]),
        'scaling_relative_to_root_parent': array([1, 1, 1, 1]),
        'data': np.array(['qr-code-1-data', 'qr-code-2-data', 'qr-code-3-data', 'qr-code-4-data'])
    }
)
```

As you can see, we have extended the standard set of metadata for predictions maintained by `supervision`.
Adding this metadata is needed to ensure compatibility with blocks from `roboflow_core` plugin.

The design of metadata is suboptimal (as metadata regarding whole image is duplicated across all 
bounding boxes and there is no way on how to save metadata for empty predictions). We
have [GH issue](https://github.com/roboflow/inference/issues/567) to communicate around this
problem.

**Details of additional fields:**

* `detection_id` - unique identifier for each detection, to be used for when dependent elements 
are created based on specific detection (example: Dynamic Crop takes this value as parent id for new image)

* `parent_id` - identifier of image that generated prediction (to be fetched from `WorkflowImageData` object)

* `image_dimensions` - dimensions of image that was basis for prediction - format: `(height, width)`

* `inference_id` - identifier of inference request (optional, relevant for Roboflow models)

* `prediction_type` - type of prediction

* `root_parent_id` - identifier of primary Workflow input that was responsible for downstream prediction 
(to be fetched from `WorkflowImageData` object) - usually identifier of Workflow input placeholder 

* `root_parent_coordinates` - offset regarding origin input - format (`offset_x`, `offset_y`)

* `root_parent_dimensions` - dimensions of origin input image `(height, width)`

* `parent_coordinates` - offset regarding parent - format (`offset_x`, `offset_y`)

* `parent_dimensions` - dimensions of parent image `(height, width)`

* `scaling_relative_to_parent` - scaling factor regarding parent image

* `scaling_relative_to_root_parent` - scaling factor regarding origin input image

* `data` - extracted barcode

**SERIALISATION:**
Execution Engine behind API will serialise underlying data once selector of this kind is declared as
Workflow output - serialisation will be executed such that `sv.Detections.from_inference(...)`
can decode the output. Entity details: [ObjectDetectionInferenceResponse](https://detect.roboflow.com/docs)
"""
BAR_CODE_DETECTION_KIND = Kind(
    name="bar_code_detection",
    description="Prediction with barcode detection",
    docs=BAR_CODE_DETECTION_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="sv.Detections",
)
PREDICTION_TYPE_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the type of prediction.

Examples:
```
"object-detection"
"instance-segmentation"
```
"""
PREDICTION_TYPE_KIND = Kind(
    name="prediction_type",
    description="String value with type of prediction",
    docs=PREDICTION_TYPE_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

PARENT_ID_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the context of prediction.
For example - whenever there is a workflow with multiple models - such that first model detect objects 
and then other models make their predictions based on crops from first model detections - `parent_id`
helps to figure out which detection of the first model is associated to which downstream predictions.

Examples:
```
"uuid-1"
"uuid-2"
```
"""
PARENT_ID_KIND = Kind(
    name="parent_id",
    description="Identifier of parent for step output",
    docs=PARENT_ID_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

IMAGE_METADATA_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the image that prediction was made against.

Examples:
```
[{{"width": 1280, "height": 720}}, {{"width": 1920, "height": 1080}}]
[{{"width": 1280, "height": 720}}]
```
"""
IMAGE_METADATA_KIND = Kind(
    name="image_metadata",
    description="Dictionary with image metadata required by supervision",
    docs=IMAGE_METADATA_KIND_DOCS,
    serialised_data_type="dict",
    internal_data_type="dict",
)


LANGUAGE_MODEL_OUTPUT_KIND_DOCS = """
This kind represents output generated by language model. It is Python string, which can be processed 
by blocks transforming LLMs / VLMs output into structured form.

Examples:
```
{"predicted_class": "car", "confidence": 0.7}  # which is example JSON with classification prediction
"The is A."  # which is example unstructured generation for VQA task 
``` 
"""

LANGUAGE_MODEL_OUTPUT_KIND = Kind(
    name="language_model_output",
    description="LLM / VLM output",
    docs=LANGUAGE_MODEL_OUTPUT_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

INFERENCE_ID_KIND_DOCS = """
This kind represents identifier of inference process, which is usually opaque string used as correlation
identifier for external systems (like Roboflow Model Monitoring).

Examples:
```
b1851e3d-a145-4540-a39e-875f21f6cd84
```
"""

INFERENCE_ID_KIND = Kind(
    name="inference_id",
    description="Inference identifier",
    docs=INFERENCE_ID_KIND_DOCS,
    serialised_data_type="str",
    internal_data_type="str",
)

SECRET_KIND = Kind(
    name="secret",
    description="Secret value",
    docs="This kind represents a secret - password or other credential that should remain confidential.",
    serialised_data_type="str",
    internal_data_type="str",
)


STEP_AS_SELECTED_ELEMENT = "step"
STEP_OUTPUT_AS_SELECTED_ELEMENT = "step_output"
BATCH_AS_SELECTED_ELEMENT = "batch"
SCALAR_AS_SELECTED_ELEMENT = "scalar"
ANY_DATA_AS_SELECTED_ELEMENT = "any_data"

StepSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_0-9\-]+"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: STEP_AS_SELECTED_ELEMENT,
        }
    ),
]

ImageInputField = Field(
    title="Image",
    description="The image to infer on.",
    examples=["$inputs.image", "$steps.cropping.crops"],
    validation_alias=AliasChoices("images", "image"),
)

RoboflowModelField = Field(
    title="Model",
    description="Roboflow model identifier.",
    examples=["my_project/3", "$inputs.model"],
)


def StepOutputSelector(kind: Optional[List[Kind]] = None):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        REFERENCE_KEY: True,
        SELECTED_ELEMENT_KEY: STEP_OUTPUT_AS_SELECTED_ELEMENT,
        KIND_KEY: [k.dict() for k in kind],
        SELECTOR_POINTS_TO_BATCH_KEY: True,
    }
    return Annotated[
        str,
        StringConstraints(pattern=r"^\$steps\.[A-Za-z_\-0-9]+\.[A-Za-z_*0-9\-]+$"),
        Field(json_schema_extra=json_schema_extra),
    ]


def WorkflowParameterSelector(kind: Optional[List[Kind]] = None):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        REFERENCE_KEY: True,
        SELECTED_ELEMENT_KEY: "workflow_parameter",
        KIND_KEY: [k.dict() for k in kind],
    }
    return Annotated[
        str,
        StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9\-]+$"),
        Field(json_schema_extra=json_schema_extra),
    ]


WorkflowImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9\-]+$"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: "workflow_image",
            KIND_KEY: [IMAGE_KIND.dict()],
            SELECTOR_POINTS_TO_BATCH_KEY: True,
        }
    ),
]

StepOutputImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_\-0-9]+\.[A-Za-z_*0-9\-]+$"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: STEP_OUTPUT_AS_SELECTED_ELEMENT,
            KIND_KEY: [IMAGE_KIND.dict()],
            SELECTOR_POINTS_TO_BATCH_KEY: True,
        }
    ),
]

FloatZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


WorkflowVideoMetadataSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9\-]+$"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: "workflow_video_metadata",
            KIND_KEY: [VIDEO_METADATA_KIND.dict()],
            SELECTOR_POINTS_TO_BATCH_KEY: True,
        }
    ),
]


def Selector(
    kind: Optional[List[Kind]] = None,
):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        REFERENCE_KEY: True,
        SELECTED_ELEMENT_KEY: ANY_DATA_AS_SELECTED_ELEMENT,
        KIND_KEY: [k.dict() for k in kind],
        SELECTOR_POINTS_TO_BATCH_KEY: "dynamic",
    }
    return Annotated[
        str,
        StringConstraints(
            pattern=r"(^\$steps\.[A-Za-z_\-0-9]+\.[A-Za-z_*0-9\-]+$)|(^\$inputs.[A-Za-z_0-9\-]+$)"
        ),
        Field(json_schema_extra=json_schema_extra),
    ]
