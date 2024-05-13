from typing import List, Optional

from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import Annotated, Literal


class Kind(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    docs: Optional[str] = Field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__() + self.description.__hash__() + self.docs.__hash__()


REFERENCE_KEY = "reference"
SELECTED_ELEMENT_KEY = "selected_element"
KIND_KEY = "kind"

DOCS_NOTE_ABOUT_BATCH = """
**Important note**:

When you see `Batch[<A>]` in a name, it means that each group of data, called a batch, will contain elements 
of type `<A>`. This also implies that if there are multiple inputs or outputs for a batch-wise operation, 
they will maintain the same order of elements within each batch. 
"""

WILDCARD_KIND_DOCS = """
This is a special kind that represents Any value - which is to be used by default if 
kind annotation is not specified. It will not tell the compiler what is to be expected
as a specific value in runtime, but at the same time, it makes it possible to run the 
workflow when we do not know or do not care about types. 

**Important note:** Usage of this kind reduces execution engine capabilities to predict 
problems with workflow and make those problems to be visible while running the workflow.
"""
WILDCARD_KIND = Kind(
    name="*", description="Equivalent of any element", docs=WILDCARD_KIND_DOCS
)

IMAGE_KIND_DOCS = f"""
This is the representation of image batch in `workflows`. The value behind this kind 
is Python list of dictionaries. Each of this dictionary is native `inference` image with
the following keys defined:
```python
{{
    "type": "url",   # there are different types supported, including np arrays and PIL images
    "value": "..."   # value depends on `type`
}}
```
This format makes it possible to use [inference image utils](https://inference.roboflow.com/docs/reference/inference/core/utils/image_utils/)
to operate on the images. 

Some blocks that output images may add additional fields - like "parent_id", which should
not be modified but may be used is specific contexts - for instance when
one needs to tag predictions with identifier of parent image.

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_IMAGES_KIND = Kind(
    name="Batch[image]", description="Image in workflows", docs=IMAGE_KIND_DOCS
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
    description="List of values of any types",
    docs=LIST_OF_VALUES_KIND_DOCS,
)

BATCH_OF_SERIALISED_PAYLOADS_KIND_DOCS = f"""
This value represents list of serialised values. Each serialised value is either
string or bytes - if something else is provided - it will be attempted to be serialised 
to JSON.

Examples:
```
["some", b"other", {{"my": "dictionary"}}]
```

This kind is to be used in combination with sinks blocks and serializers blocks.
Serializer should output value of this kind which shall then be accepted by sink.

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_SERIALISED_PAYLOADS_KIND = Kind(
    name="Batch[serialised_payloads]",
    description="List of serialised elements that can be registered in the sink",
    docs=BATCH_OF_SERIALISED_PAYLOADS_KIND_DOCS,
)

BOOLEAN_KIND_DOCS = """
This kind represents single boolean value - `True` or `False`
"""
BOOLEAN_KIND = Kind(name="boolean", description="Boolean flag", docs=BOOLEAN_KIND_DOCS)
BATCH_OF_BOOLEAN_KIND_DOCS = f"""
This kind represents batch of boolean values. 

Examples:
```
[True, False, False, True]
[True, True]
``` 

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_BOOLEAN_KIND = Kind(
    name="Batch[boolean]",
    description="Boolean values batch",
    docs=BATCH_OF_BOOLEAN_KIND_DOCS,
)

INTEGER_KIND_DOCS = """
Examples:
```
1
2
```
"""
INTEGER_KIND = Kind(name="integer", description="Integer value", docs=INTEGER_KIND_DOCS)
STRING_KIND_DOCS = """
Examples:
```
"my string value"
```
"""
STRING_KIND = Kind(name="string", description="String value", docs=STRING_KIND_DOCS)
BATCH_OF_STRING_KIND_DOCS = f"""
This kind represents batch of string values.

Examples:
```
["a", "b", "c"]
["d", "e"]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_STRING_KIND = Kind(
    name="Batch[string]",
    description="Batch of string values",
    docs=BATCH_OF_STRING_KIND_DOCS,
)
BATCH_OF_TOP_CLASS_KIND_DOCS = f"""
The kind represent top classes predicted by classification model - representing its predictions on batch of images.

Example:
```
["car", "dog", "car", "cat"]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_TOP_CLASS_KIND = Kind(
    name="Batch[top_class]",
    description="Batch of string values representing top class predicted by classification model",
    docs=BATCH_OF_TOP_CLASS_KIND_DOCS,
)
FLOAT_KIND_DOCS = """
Example:
```
1.3
2.7
```
"""
FLOAT_KIND = Kind(name="float", description="Float value", docs=FLOAT_KIND_DOCS)
DICTIONARY_KIND_DOCS = """
This kind represent a value of any Python dict.

Examples:
```
{"my_key", "my_value"}
``` 
"""
DICTIONARY_KIND = Kind(name="dictionary", description="Dictionary")
BATCH_OF_DICTIONARY_KIND_DOCS = f"""
This kind represent a batch of any Python dicts.
Examples:
```
[{{"my_key", "my_value_1"}}, {{"my_key", "my_value_2"}}]
``` 

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_DICTIONARY_KIND = Kind(
    name="Batch[dictionary]",
    description="Batch of dictionaries",
    docs=BATCH_OF_DICTIONARY_KIND_DOCS,
)
BATCH_OF_CLASSIFICATION_PREDICTION_KIND_DOCS = f"""
This kind represent predictions from Roboflow classification model.

Examples:
```
# in case of multi-class classification
[
    {{"class": "A", "class_id": 0, "confidence": 0.3}}, {{"class": "B", "class_id": 1, "confidence": 0.7}},
    {{"class": "A", "class_id": 0, "confidence": 0.7}}, {{"class": "B", "class_id": 1, "confidence": 0.3}},
]
[
    {{"class": "A", "class_id": 0, "confidence": 0.1}}, {{"class": "B", "class_id": 1, "confidence": 0.9}},
    {{"class": "A", "class_id": 0, "confidence": 0.9}}, {{"class": "B", "class_id": 1, "confidence": 0.1}},
]

# in case of multi-label classification
[
    {{
        "class_a": 0.3,
        "class_b": 0.4,
    }},
    {{
        "class_c": 0.3,
        "class_b": 0.4,
    }}
]
[
    {{
        "car": 0.4,
        "truck": 0.5,
    }},
    {{
        "truck": 0.6,
        "bike": 0.4,
    }}
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_CLASSIFICATION_PREDICTION_KIND = Kind(
    name="Batch[classification_prediction]",
    description="`'predictions'` key from Roboflow classifier output",
    docs=BATCH_OF_CLASSIFICATION_PREDICTION_KIND_DOCS,
)
BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND_DOCS = f"""
This kind represents batch of predictions from Roboflow object detection model.

Example:
```
# Each prediction in batch is list of dictionaries that contains detected objects (detections)
[
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid"}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid"}}
    ],
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid"}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid"}}
    ]
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[object_detection_prediction]",
    description="`'predictions'` key from Roboflow object detection model output",
    docs=BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND_DOCS,
)
BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS = f"""
This kind represents batch of predictions from Roboflow instance segmentation model.

Example:
```
# Each prediction in batch is list of dictionaries that contains detected objects (detections) and list of points 
providing object contour,
[
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "points": [{{"x": 300, "y": 200}}]}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "points": [{{"x": 300, "y": 200}}}}
    ],
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "points": [{{"x": 300, "y": 200}}}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "points": [{{"x": 300, "y": 200}}}}
    ]
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="Batch[instance_segmentation_prediction]",
    description="`'predictions'` key from Roboflow instance segmentation model output",
    docs=BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND_DOCS,
)
BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND_DOCS = f"""
This kind represents batch of predictions from Roboflow keypoint detection model.

Example:
```
# Each prediction in batch is list of dictionaries that contains detected objects (detections) and list of points of 
object skeleton. 
[
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "keypoints": [{{"x": 300, "y": 200, "confidence": 0.3, "class_id": 0, "class_name": "tire_center"}}]}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "keypoints": [{{"x": 300, "y": 200, "confidence": 0.3, "class_id": 0, "class_name": "tire_center"}}}}
    ],
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "keypoints": [{{"x": 300, "y": 200, "confidence": 0.3, "class_id": 0, "class_name": "tire_center"}}}},
        {{"x": 600, "y": 900, "width": 100, "height" 50, "confidence": 0.3, "class": "car", "class_id": 0.1, "detection_id": "random-uuid", "keypoints": [{{"x": 300, "y": 200, "confidence": 0.3, "class_id": 0, "class_name": "tire_center"}}}}
    ]
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[keypoint_detection_prediction]",
    description="`'predictions'` key from Roboflow keypoint detection model output",
    docs=BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND_DOCS,
)
BATCH_OF_QR_CODE_DETECTION_KIND_DOCS = f"""
This kind represents batch of predictions regarding QR codes location and data their provide.

Example:
```
# Each prediction in batch is list of dictionaries that contains detected QR codes (detections) and their metadata
[
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "qr_code", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "qr_code", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},    ],
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "qr_code", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "qr_code", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
    ]
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_QR_CODE_DETECTION_KIND = Kind(
    name="Batch[qr_code_detection]",
    description="Roboflow prediction with QR code detection",
    docs=BATCH_OF_QR_CODE_DETECTION_KIND_DOCS,
)
BATCH_OF_BAR_CODE_DETECTION_KIND_DOCS = f"""
This kind represents batch of predictions regarding barcodes location and data their provide.

Example:
```
# Each prediction in batch is list of dictionaries that contains detected barcodes (detections) and their metadata
[
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "barcode", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "barcode", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},    ],
    [
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "barcode", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
        {{"x": 300, "y": 400, "width": 100, "height" 50, "confidence": 1.0, "class": "barcode", "class_id": 0.1, "detection_id": "random-uuid", "data": "<qr-code-data>"}},
    ]
]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_BAR_CODE_DETECTION_KIND = Kind(
    name="Batch[bar_code_detection]",
    description="Roboflow prediction with barcode detection",
    docs=BATCH_OF_BAR_CODE_DETECTION_KIND_DOCS,
)
BATCH_OF_PREDICTION_TYPE_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the type of prediction.

Examples:
```
["object-detection", "object-detection"]
["instance-segmentation", "instance-segmentation"]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_PREDICTION_TYPE_KIND = Kind(
    name="Batch[prediction_type]",
    description="String value with type of prediction",
    docs=BATCH_OF_PREDICTION_TYPE_KIND_DOCS,
)
BATCH_OF_PARENT_ID_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the context of prediction.
For example - whenever there is a workflow with multiple models - such that first model detect objects 
and then other models make their predictions based on crops from first model detections - `parent_id`
helps to figure out which detection of the first model is associated to which downstream predictions.

Examples:
```
["uuid-1", "uuid-1", "uuid-2", "uuid-2"]
["uuid-1", "uuid-1", "uuid-1", "uuid-1"]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_PARENT_ID_KIND = Kind(
    name="Batch[parent_id]",
    description="Identifier of parent for step output",
    docs=BATCH_OF_PARENT_ID_KIND_DOCS,
)
BATCH_OF_IMAGE_METADATA_KIND_DOCS = f"""
This kind represent batch of prediction metadata providing information about the image that prediction was made against.

Examples:
```
[{{"width": 1280, "height": 720}}, {{"width": 1920, "height": 1080}}]
[{{"width": 1280, "height": 720}}]
```

{DOCS_NOTE_ABOUT_BATCH}
"""
BATCH_OF_IMAGE_METADATA_KIND = Kind(
    name="Batch[image_metadata]",
    description="Dictionary with image metadata required by supervision",
    docs=BATCH_OF_IMAGE_METADATA_KIND_DOCS,
)

STEP_AS_SELECTED_ELEMENT = "step"
STEP_OUTPUT_AS_SELECTED_ELEMENT = "step_output"

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


def StepOutputSelector(kind: Optional[List[Kind]] = None):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        REFERENCE_KEY: True,
        SELECTED_ELEMENT_KEY: STEP_OUTPUT_AS_SELECTED_ELEMENT,
        KIND_KEY: [k.dict() for k in kind],
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
            KIND_KEY: [BATCH_OF_IMAGES_KIND.dict()],
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
            KIND_KEY: [BATCH_OF_IMAGES_KIND.dict()],
        }
    ),
]

FloatZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


class FlowControl(BaseModel):
    mode: Literal["pass", "terminate_branch", "select_step"]
    context: Optional[str] = Field(default=None)
