from typing import List, Optional

from pydantic import BaseModel, Field, StringConstraints
from typing_extensions import Annotated, Literal


class Kind(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)

    def __hash__(self) -> int:
        return self.name.__hash__() + self.description.__hash__()


WILDCARD_KIND = Kind(name="*", description="Equivalent of any element")
IMAGE_KIND = Kind(name="Batch[image]", description="Image in workflows")
ROBOFLOW_MODEL_ID_KIND = Kind(name="roboflow_model_id", description="Roboflow model id")
ROBOFLOW_PROJECT_KIND = Kind(
    name="roboflow_project", description="Roboflow project name"
)
ROBOFLOW_API_KEY_KIND = Kind(name="roboflow_api_key", description="Roboflow API key")
FLOAT_ZERO_TO_ONE_KIND = Kind(
    name="float_zero_to_one", description="Float value in range [0.0, 1.0]"
)
LIST_OF_VALUES_KIND = Kind(
    name="list_of_values", description="List of values of any types"
)
BATCH_OF_SERIALISED_PAYLOADS_KIND = Kind(
    name="Batch[serialised_payloads]",
    description="List of serialised elements that can be registered in the sink",
)
BOOLEAN_KIND = Kind(name="boolean", description="Boolean flag")
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
    description="'predictions' key from Roboflow classifier output",
)
OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[object_detection_prediction]",
    description="'predictions' key from Roboflow object detection model output",
)
INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="Batch[instance_segmentation_prediction]",
    description="'predictions' key from Roboflow instance segmentation model output",
)
KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="Batch[keypoint_detection_prediction]",
    description="'predictions' key from Roboflow keypoint detection model output",
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

StepSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_0-9\-]+"),
    Field(
        json_schema_extra={
            "reference": True,
            "selected_element": STEP_AS_SELECTED_ELEMENT,
        }
    ),
]


def StepOutputSelector(kind: Optional[List[Kind]] = None):
    if kind is None:
        kind = [WILDCARD_KIND]
    json_schema_extra = {
        "reference": True,
        "selected_element": "step_output",
        "kind": [k.dict() for k in kind],
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
        "reference": True,
        "selected_element": "inference_parameter",
        "kind": [k.dict() for k in kind],
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
            "reference": True,
            "selected_element": "inference_image",
            "kind": [IMAGE_KIND.dict()],
        }
    ),
]

OutputStepImageSelector = Annotated[
    str,
    StringConstraints(pattern=r"^\$steps\.[A-Za-z_\-0-9]+\.[A-Za-z_*0-9\-]+$"),
    Field(
        json_schema_extra={
            "reference": True,
            "selected_element": "step_output",
            "kind": [IMAGE_KIND.dict()],
        }
    ),
]

FloatZeroToOne = Annotated[float, Field(ge=0.0, le=1.0)]


class FlowControl(BaseModel):
    mode: Literal["pass", "terminate_branch", "select_step"]
    context: Optional[str] = Field(default=None)
