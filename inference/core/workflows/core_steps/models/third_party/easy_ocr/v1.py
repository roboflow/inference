from typing import Dict, List, Literal, Optional, Type, Union

import easyocr
import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from uuid import uuid4

from supervision.config import CLASS_NAME_DATA_FIELD
from inference.core.workflows.core_steps.common.entities import StepExecutionMode

from inference.core.workflows.core_steps.common.utils import attach_parents_coordinates_to_sv_detections
from inference.core.workflows.execution_engine.constants import (
    DETECTED_CODE_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PREDICTION_TYPE_KEY,
)

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
 Retrieve the characters in an image using EasyOCR Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e.
ObjectDetectionBlock). An object detection model could isolate specific regions from an
image (i.e. a shipping container ID in a logistics use case) for further processing.
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis
on particular regions of an image.
"""

EXPECTED_OUTPUT_KEYS = {"result", "parent_id", "root_parent_id", "prediction_type"}

CHARACTER_SETS = Literal[
    "Abaza",
    "Adyghe",
    "Afrikaans",
    "Angika",
    "Arabic",
    "Assamese",
    "Avar",
    "Azerbaijani",
    "Belarusian",
    "Bulgarian",
    "Bihari",
    "Bhojpuri",
    "Bengali",
    "Bosnian",
    "SimplifiedChinese",
    "TraditionalChinese",
    "Chechen",
    "Czech",
    "Welsh",
    "Danish",
    "German",
    "English",
    "Spanish",
    "Estonian",
    "Persian",
    "Finnish",
    "French",
    "Irish",
    "GoanKonkani",
    "Hindi",
    "Croatian",
    "Hungarian",
    "Indonesian",
    "Ingush",
    "Icelandic",
    "Italian",
    "Japanese",
    "Kabardian",
    "Kannada",
    "Korean",
    "Kurdish",
    "Latin",
    "Lak",
    "Lezghian",
    "Lithuanian",
    "Latvian",
    "Magahi",
    "Maithili",
    "Maori",
    "Mongolian",
    "Marathi",
    "Malay",
    "Maltese",
    "Nepali",
    "Newari",
    "Dutch",
    "Norwegian",
    "Occitan",
    "Pali",
    "Polish",
    "Portuguese",
    "Romanian",
    "Russian",
    "Serbian",
    "Serbian",
    "Nagpuri",
    "Slovak",
    "Slovenian",
    "Albanian",
    "Swedish",
    "Swahili",
    "Tamil",
    "Tabassaran",
    "Telugu",
    "Thai",
    "Tajik",
    "Tagalog",
    "Turkish",
    "Uyghur",
    "Ukranian",
    "Urdu",
    "Uzbek",
    "Vietnamese",
]

LANGUAGE_FILE_PREFIXES: Dict[str, str] = {
    "Abaza": "abq",
    "Adyghe": "ady",
    "Afrikaans": "af",
    "Angika": "ang",
    "Arabic": "ar",
    "Assamese": "as",
    "Avar": "ava",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bihari": "bh",
    "Bhojpuri": "bho",
    "Bengali": "bn",
    "Bosnian": "bs",
    "SimplifiedChinese": "ch_sim",
    "TraditionalChinese": "ch_tra",
    "Chechen": "che",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian": "fa",
    "Finnish": "fi",
    "French": "fr",
    "Irish": "ga",
    "GoanKonkani": "gom",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Ingush": "inh",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Kabardian": "kbd",
    "Kannada": "kn",
    "Korean": "ko",
    "Kurdish": "ku",
    "Latin": "la",
    "Lak": "lbe",
    "Lezghian": "lez",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Magahi": "mah",
    "Maithili": "mai",
    "Maori": "mi",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Maltese": "mt",
    "Nepali": "ne",
    "Newari": "new",
    "Dutch": "nl",
    "Norwegian": "no",
    "Occitan": "oc",
    "Pali": "pi",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Serbian": "rs_cyrillic",
    "Serbian": "rs_latin",
    "Nagpuri": "sck",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Albanian": "sq",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Tabassaran": "tab",
    "Telugu": "te",
    "Thai": "th",
    "Tajik": "tjk",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Uyghur": "ug",
    "Ukranian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
}


class BlockManifest(WorkflowBlockManifest):

    model_config = ConfigDict(
        json_schema_extra={
            "name": "EasyOCR",
            "version": "v1",
            "short_description": "Extract text from an image using EasyOCR.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "ui_manifest": {
                "section": "model",
                "icon": "far fa-text",
                "blockPriority": 11,
                "inDevelopment": False,
                "inference": True,
            },
        }
    )
    type: Literal["roboflow_core/easy_ocr@v1", "EasyOCR"]
    name: str = Field(description="Unique name of step in workflows")
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image for this step.",
        examples=["$inputs.image"],
    )
    character_set: CHARACTER_SETS = Field(
        title="Character Set",
        description="Character set to use for OCR",
        default="English",
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="ocr_result", kind=[OBJECT_DETECTION_PREDICTION_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


def ocr_result_to_detections(image: WorkflowImageData, result: List[Union[List, str]]) -> sv.Detections:
    # Prepare lists for bounding boxes, confidences, class IDs, and labels
    xyxy, confidences, class_ids, label, class_names = [], [], [], [], []

    # Extract data from OCR result
    for detection in result:
        bbox, text, confidence = detection[0], detection[1], detection[2]

        # Convert bounding box format
        x_min = int(min([point[0] for point in bbox]))
        y_min = int(min([point[1] for point in bbox]))
        x_max = int(max([point[0] for point in bbox]))
        y_max = int(max([point[1] for point in bbox]))

        # Append data to lists
        xyxy.append([x_min, y_min, x_max, y_max])
        label.append(text)
        confidences.append(confidence)
        class_ids.append(0)
        class_names.append("ocr_result")

    # Convert to NumPy arrays
    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidences),
        class_id=np.array(class_ids),
        data={CLASS_NAME_DATA_FIELD: class_names},
    )
    detections[DETECTION_ID_KEY] = np.array([uuid4() for _ in range(len(detections))])
    detections[PREDICTION_TYPE_KEY] = np.array(["barcode-detection"] * len(detections))
    detections[DETECTED_CODE_KEY] = np.array(label)
    img_height, img_width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array(
        [[img_height, img_width]] * len(detections)
    )
    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )
    return detections


class EasyOCRBlockV1(WorkflowBlock):
    reader: Optional[easyocr.Reader] = None

    def __init__(
        self,
        step_execution_mode: StepExecutionMode,
    ):
        self._step_execution_mode = step_execution_mode

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["step_execution_mode"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(self, image: WorkflowImageData, character_set: str) -> BlockResult:
        if self._step_execution_mode is StepExecutionMode.LOCAL:
            reader_char_set = LANGUAGE_FILE_PREFIXES.get(character_set)
            if reader_char_set is None:
                raise ValueError(f"Unsupported character set: {character_set}")
            if not self.reader:
                self.reader = easyocr.Reader([reader_char_set])
            result = self.reader.readtext(image.numpy_image)
            return {"ocr_result": ocr_result_to_detections(image, result)}
        elif self._step_execution_mode == StepExecutionMode.REMOTE:
            raise NotImplementedError(
                "Remote execution is not supported for EasyOCR. Please use a local or dedicated inference server."
            )
        else:
            raise ValueError(
                f"Unknown step execution mode: {self._step_execution_mode}"
            )
