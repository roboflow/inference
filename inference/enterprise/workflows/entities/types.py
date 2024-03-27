from typing import Annotated, List, Optional

from pydantic import Field, StringConstraints

WILDCARD_KIND = "*"
IMAGE_KIND = "image"
ROBOFLOW_MODEL_ID_KIND = "roboflow_model_id"
ROBOFLOW_PROJECT = "roboflow_project"
FLOAT_ZERO_TO_ONE_KIND = "float_zero_to_one"
LIST_OF_VALUES_KIND = "list_of_values"
BOOLEAN_KIND = "boolean"
INTEGER_KIND = "integer"
STRING_KIND = "string"
DICTIONARY_KIND = "dictionary"
CLASSIFICATION_PREDICTION_KIND = "classification_prediction"
OBJECT_DETECTION_PREDICTION_KIND = "object_detection_prediction"
INSTANCE_SEGMENTATION_PREDICTION_KIND = "instance_segmentation_prediction"
KEYPOINT_DETECTION_PREDICTION_KIND = "keypoint_detection_prediction"
QR_CODE_DETECTION_KIND = "qr_code_detection"
BAR_CODE_DETECTION_KIND = "bar_code_detection"

PREDICTION_TYPE_KIND = "prediction_type"
PARENT_ID_KIND = "parent_id"
IMAGE_METADATA_KIND = "image_metadata"

StepSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_0-9]+"),
    Field(
        json_schema_extra={
            "reference": True,
            "selected_element": "step",
        }
    ),
]


def StepOutputSelector(kinds: Optional[List[str]] = None):
    if kinds is None:
        kinds = [WILDCARD_KIND]
    json_schema_extra = {
        "reference": True,
        "selected_element": "step_output",
        "kind": kinds,
    }
    return Annotated[
        str,
        StringConstraints(pattern=r"^\$steps\.[A-Za-z_0-9]+\.[A-Za-z_*0-9]+$"),
        Field(json_schema_extra=json_schema_extra),
    ]


def InferenceParameterSelector(kinds: Optional[List[str]] = None):
    if kinds is None:
        kinds = [WILDCARD_KIND]
    json_schema_extra = {
        "reference": True,
        "selected_element": "inference_parameter",
        "kind": kinds,
    }
    return Annotated[
        str,
        StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9]+$"),
        Field(json_schema_extra=json_schema_extra),
    ]


InferenceImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$inputs.[A-Za-z_0-9]+$"),
    Field(
        json_schema_extra={
            "reference": True,
            "selected_element": "inference_image",
            "kind": [IMAGE_KIND],
        }
    ),
]

OutputStepImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_0-9]+\.[A-Za-z_*0-9]+$"),
    Field(
        json_schema_extra={
            "reference": True,
            "selected_element": "output_step",
            "kind": [IMAGE_KIND],
        }
    ),
]

FloatZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]
