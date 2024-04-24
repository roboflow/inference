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

IMAGE_KIND_DOCS = """
This is the representation of image batch in `workflows`. The value behind this kind 
is Python list of dictionaries. Each of this dictionary is native `inference` image with
the following keys defined:
```python
{
    "type": "url",   # there are different types supported, including np arrays and PIL images
    "value": "..."   # value depends on `type`
}
```
This format makes it possible to use [inference image utils](https://inference.roboflow.com/docs/reference/inference/core/utils/image_utils/)
to operate on the images. 

Some blocks that outputs images may add additional fields - like "parent_id", which should
not be modified but may be used is specific contexts - for instance when
one needs to tag predictions with identifier of parent image.
"""
IMAGE_KIND = Kind(
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

BATCH_OF_SERIALISED_PAYLOADS_KIND_DOCS = """
This value represents list of serialised values. Each serialised value is either
string or bytes - if something else is provided - it will be attempted to be serialised 
to JSON.

Example:
```
["some", b"other", {"my": "dictionary"}]
```

This kind is to be used in combination with sinks blocks and serializers blocks.
Serializer should output value of this kind which shall then be accepted by sink.
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
BATCH_OF_BOOLEAN_KIND = Kind(name="Batch[boolean]", description="Boolean flag batch")
INTEGER_KIND = Kind(name="integer", description="Integer value")
STRING_KIND = Kind(name="string", description="String value")
BATCH_OF_STRING_KIND = Kind(name="Batch[string]", description="String value")
TOP_CLASS_KIND = Kind(
    name="Batch[top_class]",
    description="String value representing top class predicted by classification model",
)
FLOAT_KIND = Kind(name="float", description="Float value")
DICTIONARY_KIND = Kind(name="dictionary", description="Dictionary")
BATCH_OF_DICTIONARY_KIND = Kind(name="Batch[dictionary]", description="Dictionary")
CLASSIFICATION_PREDICTION_KIND = Kind(
    name="Batch[classification_prediction]",
    description="`'predictions'` key from Roboflow classifier output",
)
OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[object_detection_prediction]",
    description="`'predictions'` key from Roboflow object detection model output",
)
INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="Batch[instance_segmentation_prediction]",
    description="`'predictions'` key from Roboflow instance segmentation model output",
)
KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[keypoint_detection_prediction]",
    description="`'predictions'` key from Roboflow keypoint detection model output",
)
QR_CODE_DETECTION_KIND = Kind(
    name="Batch[qr_code_detection]",
    description="Roboflow prediction with QR code detection",
)
BAR_CODE_DETECTION_KIND = Kind(
    name="Batch[bar_code_detection]",
    description="Roboflow prediction with barcode detection",
)

PREDICTION_TYPE_KIND = Kind(
    name="Batch[prediction_type]", description="String value with type of prediction"
)
PARENT_ID_KIND = Kind(
    name="Batch[parent_id]", description="Identifier of parent for step output"
)
IMAGE_METADATA_KIND = Kind(
    name="Batch[image_metadata]",
    description="Dictionary with image metadata required by supervision",
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


def InferenceParameterSelector(kind: Optional[List[Kind]] = None):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        REFERENCE_KEY: True,
        SELECTED_ELEMENT_KEY: "inference_parameter",
        KIND_KEY: [k.dict() for k in kind],
    }
    return Annotated[
        str,
        StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9\-]+$"),
        Field(json_schema_extra=json_schema_extra),
    ]


InferenceImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9\-]+$"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: "inference_image",
            KIND_KEY: [IMAGE_KIND.dict()],
        }
    ),
]

OutputStepImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_\-0-9]+\.[A-Za-z_*0-9\-]+$"),
    Field(
        json_schema_extra={
            REFERENCE_KEY: True,
            SELECTED_ELEMENT_KEY: STEP_OUTPUT_AS_SELECTED_ELEMENT,
            KIND_KEY: [IMAGE_KIND.dict()],
        }
    ),
]

FloatZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


class FlowControl(BaseModel):
    mode: Literal["pass", "terminate_branch", "select_step"]
    context: Optional[str] = Field(default=None)
