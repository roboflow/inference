from typing import List, Optional, Union

from pydantic import Field, model_validator

from inference.core.entities.requests.inference import (
    BaseRequest,
    InferenceRequestImage,
)

PP_OCR_STAGE_VALUES = {"none", "tiny", "small", "medium"}

# Sentinel marking a stage field that was omitted from the request, so we can
# tell it apart from an explicit ``None`` (which disables the stage). An omitted
# stage defaults to "small" (or is derived from pp_ocr_version_id).
_STAGE_UNSET = "__unset__"


class PPOCRInferenceRequest(BaseRequest):
    """
    PP-OCR inference request.

    Attributes:
        api_key (Optional[str]): Roboflow API Key.
    """

    image: Union[List[InferenceRequestImage], InferenceRequestImage]
    text_detection: Optional[str] = _STAGE_UNSET
    text_recognition: Optional[str] = _STAGE_UNSET
    pp_ocr_version_id: Optional[str] = Field(None)
    model_id: Optional[str] = Field(None)

    @model_validator(mode="after")
    def resolve_stages_and_model_id(self) -> "PPOCRInferenceRequest":
        text_detection = self.text_detection
        text_recognition = self.text_recognition
        # When neither stage is explicitly provided but a pp_ocr_version_id is
        # (e.g. the SDK/HTTP/remote path transmits the selection only as
        # pp_ocr_version_id), derive det/rec from it. A single token applies to
        # both stages.
        if text_detection is _STAGE_UNSET and text_recognition is _STAGE_UNSET:
            if self.pp_ocr_version_id:
                parts = self.pp_ocr_version_id.split("-")
                if len(parts) == 1:
                    text_detection = text_recognition = parts[0]
                elif len(parts) == 2:
                    text_detection, text_recognition = parts
                else:
                    raise ValueError(
                        f"Invalid PP-OCR pp_ocr_version_id value: {self.pp_ocr_version_id}"
                    )
        # An omitted stage defaults to "small"; an explicit ``None`` disables it.
        det = ("small" if text_detection is _STAGE_UNSET else text_detection) or "none"
        rec = (
            "small" if text_recognition is _STAGE_UNSET else text_recognition
        ) or "none"
        det = det.lower()
        rec = rec.lower()
        if det not in PP_OCR_STAGE_VALUES:
            raise ValueError(f"Invalid PP-OCR text_detection value: {det}")
        if rec not in PP_OCR_STAGE_VALUES:
            raise ValueError(f"Invalid PP-OCR text_recognition value: {rec}")
        if det == "none" and rec == "none":
            raise ValueError("PP-OCR requires at least one of detection or recognition")
        self.text_detection = det
        self.text_recognition = rec
        self.pp_ocr_version_id = f"{det}-{rec}"
        self.model_id = f"pp_ocr/{det}-{rec}"
        return self
