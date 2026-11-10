"""Read text with a PP-OCRv6 detector and recognizer from ``models``.

Canonical type: ``roboflow_core/pp_ocr@v1``.

PP-OCR has no phases because its two models compose inside one public
pipeline call. With ``block_execution="phases"`` the engine calls its ``run``,
so its errors are plain ``ValueError`` and ``LookupError``, not the
``PhaseFailure`` of the phased DocTR and EasyOCR blocks.

Differences from V1, all deliberate:

- The pipeline runs once per batch (V1: once per image). Inside that call
  the detector runs once over the batch and the recognizer once per image
  over that image's line crops; without detection the recognizer runs once
  over the batch. Rows keep the detector's ``"polygon"`` metadata. V1's
  ``detection_id`` is replaced by V2 row identity.
- The block fails when the recognizer returns another number of line texts
  than detected lines. V1 named the missing lines ``""`` and dropped extra
  texts.
- Package ids are literal strings only. Workflow selector prefixes
  (``$inputs.``, ``$steps.``, ``$sources.``, ``$operators.``) fail
  compilation; supply the actual package id instead.
"""

from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple, Union

import torch
from inference_models.models.base.object_detection import Detections
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline
from pydantic import Field, StrictStr, ValidationInfo, field_validator, model_validator
from roboflow_workflows.v2.blocks.models._shared.ocr import (
    PREDICTION_TYPE,
    TEXT_KEY,
    _ocr_result,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    models_cache,
    one_per_call,
    require_rgb,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAME_KEY
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)


PP_OCR_DETECTION_MODEL = "pp-ocrv6-det/{size}"

PP_OCR_RECOGNITION_MODEL = "pp-ocrv6-rec/{size}"

PP_OCR_STAGE_REQUIRED = (
    "PP-OCR requires at least one of text detection or text recognition to be "
    "enabled"
)

# Rows are named by their text; this only names class id 0.
TEXT_REGION_CLASS_NAMES = ("text-region",)


class PPOCRBlock(Block):
    """Read text with a PP-OCRv6 detector and recognizer from ``models``.

    The two models compose through ``PPOCRv6Pipeline.with_models``; either
    stage may be ``"none"``, not both. With detection, each row is one text
    line in reading order, named by its recognised text (``""`` without
    recognition) and keeping the detector's ``"polygon"``. Without detection
    every image is read as one pre-cropped line and has no rows.
    """

    type = "roboflow_core/pp_ocr@v1"
    metadata = {
        "name": "PP-OCR",
        "port_of": "roboflow_core/pp_ocr@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "paddle", "paddleocr"],
    }
    outputs = {
        "result": Output(
            STRING_KIND,
            source="image",
            description="Recognised text of the whole image.",
        ),
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=(
                "Native Detections of text lines in reading order, each named by its "
                "recognised text."
            ),
        ),
        "model_id": Output(
            STRING_KIND,
            source="image",
            description=(
                "Id of the detection model that served this image; the recognition "
                "model's id when text detection is 'none'."
            ),
        ),
        "inference_id": Output(
            STRING_KIND,
            source="image",
            description=(
                "Fresh id of this image's result; for results with rows, also the "
                "namespace of their row identity."
            ),
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to read text from; delivered as one batch."
        )
        text_detection: Literal["none", "tiny", "small", "medium"] = Field(
            default="small",
            title="Text Detection Model Size",
            description=(
                "Text detection model size (model id pp-ocrv6-det/<size>). 'none' "
                "disables detection: each image is read as one pre-cropped text "
                "line."
            ),
            examples=["small", "none"],
        )
        text_recognition: Literal["none", "tiny", "small", "medium"] = Field(
            default="small",
            title="Text Recognition Model Size",
            description=(
                "Text recognition model size (model id pp-ocrv6-rec/<size>). "
                "'none' disables recognition: only text boxes are returned."
            ),
            examples=["small", "none"],
        )
        text_detection_package_id: Optional[StrictStr] = Field(
            default=None,
            description=(
                "Exact package of the detection model, as a literal string; a "
                "workflow selector is rejected at compile time. Supply the actual "
                "package id, or None to let the loader choose."
            ),
            examples=[None],
        )
        text_recognition_package_id: Optional[StrictStr] = Field(
            default=None,
            description=(
                "Exact package of the recognition model, as a literal string; a "
                "workflow selector is rejected at compile time. Supply the actual "
                "package id, or None to let the loader choose."
            ),
            examples=[None],
        )

        @field_validator("text_detection_package_id", "text_recognition_package_id")
        @classmethod
        def _validate_literal_package_id(
            cls, value: Optional[str], info: ValidationInfo
        ) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")
            # Match the engine's reserved selector prefixes, including malformed
            # references. Other dollar signs in ordinary package ids are literal.
            if value is not None and value.startswith(
                ("$inputs.", "$steps.", "$sources.", "$operators.")
            ):
                raise ValueError(
                    f"{info.field_name} accepts literal-only package IDs; workflow "
                    f"selectors such as {value!r} are unsupported. Supply the actual "
                    "package ID as a literal string, or None to let the loader choose."
                )

            return value

        @model_validator(mode="after")
        def _require_a_stage(self) -> "PPOCRBlock.Params":
            if self.text_detection == "none" and self.text_recognition == "none":
                raise ValueError(PP_OCR_STAGE_REQUIRED)

            return self

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the enabled stages' models this step needs from ``models``."""
        requests = _pp_ocr_requests(
            text_detection=params.text_detection,
            text_recognition=params.text_recognition,
            text_detection_package_id=params.text_detection_package_id,
            text_recognition_package_id=params.text_recognition_package_id,
        )
        resources = [
            DependentResource(
                resource_type="model",
                identifier=request.model_id,
                details={
                    "resource": "models",
                    "task_type": PREDICTION_TYPE,
                    "model_package_id": request.model_package_id,
                },
            )
            for request in requests
            if request is not None
        ]

        return resources

    def run(
        self,
        *,
        image: Batch,
        text_detection: str = "small",
        text_recognition: str = "small",
        text_detection_package_id: Optional[str] = None,
        text_recognition_package_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Read the text of every image of the batch in one pipeline call.

        In that call the detector reads the whole batch once and the recognizer
        reads each image's line crops once per image; without detection the
        recognizer reads the whole batch once.

        Args:
            image: The batch of RGB images of one workflow run.
            text_detection: Detection model size, or ``"none"``.
            text_recognition: Recognition model size, or ``"none"``.
            text_detection_package_id: Package of the detection model;
                ``None`` lets the loader choose.
            text_recognition_package_id: Package of the recognition model;
                ``None`` lets the loader choose.

        Returns:
            One ``{"result", "predictions", "model_id", "inference_id"}`` per
            image, in batch order. ``model_id`` names the detection model, or
            the recognition model when detection is disabled.

        Raises:
            ValueError: When both stages are ``"none"``, a setting is a
                ``Batch`` (one value per call), an image is not RGB, the
                pipeline returns another number of results than images,
                or the recognizer returns another number of line texts than
                detected lines.
            LookupError: When a model is missing from or conflicts with the
                supplied models.
        """
        members = image.content
        for member in members:
            require_rgb(member, block=self.type)
        for name, value in (
            ("text_detection", text_detection),
            ("text_recognition", text_recognition),
            ("text_detection_package_id", text_detection_package_id),
            ("text_recognition_package_id", text_recognition_package_id),
        ):
            one_per_call(value, name=name, block=self.type)
        detection, recognition = _pp_ocr_requests(
            text_detection=text_detection,
            text_recognition=text_recognition,
            text_detection_package_id=text_detection_package_id,
            text_recognition_package_id=text_recognition_package_id,
        )
        # An empty batch loads no model and builds no pipeline.
        if not members:
            return []

        # The one model_id output names the detector, else the recognizer.
        reported = detection or recognition
        pipeline = PPOCRv6Pipeline.with_models(
            [self._model(detection), self._model(recognition)]
        )

        outputs = pipeline.infer(
            [member.tensor_image for member in members], input_color_format="rgb"
        )
        if len(outputs) != len(members):
            raise ValueError(
                f"{self.type}: the PP-OCR pipeline returned {len(outputs)} results "
                f"for {len(members)} images"
            )
        results = [
            _ocr_result(
                output.text,
                _pp_ocr_detections(
                    output,
                    recognition_enabled=recognition is not None,
                    device=member.device,
                    block=self.type,
                ),
                member,
                class_names=TEXT_REGION_CLASS_NAMES,
                model_id=reported.model_id,
            )
            for output, member in zip(outputs, members)
        ]

        return results

    def _model(self, request: Optional[ModelRequest]) -> Any:
        model = None if request is None else self._models.get(request)

        return model


def _pp_ocr_requests(
    *,
    text_detection: str,
    text_recognition: str,
    text_detection_package_id: Optional[str],
    text_recognition_package_id: Optional[str],
) -> Tuple[Optional[ModelRequest], Optional[ModelRequest]]:
    """``(detection, recognition)`` requests; ``None`` for a disabled stage."""
    # Params reject this at compile time; direct run() calls reach this check.
    if text_detection == "none" and text_recognition == "none":
        raise ValueError(PP_OCR_STAGE_REQUIRED)

    detection = (
        None
        if text_detection == "none"
        else ModelRequest(
            PP_OCR_DETECTION_MODEL.format(size=text_detection),
            text_detection_package_id,
        )
    )
    recognition = (
        None
        if text_recognition == "none"
        else ModelRequest(
            PP_OCR_RECOGNITION_MODEL.format(size=text_recognition),
            text_recognition_package_id,
        )
    )

    return detection, recognition


def _pp_ocr_detections(
    output: Any, *, recognition_enabled: bool, device: torch.device, block: str
) -> Detections:
    """The pipeline's line boxes, each row with its ``text`` and ``class``.

    Without detection there are no boxes. Without recognition every line's
    text is ``""`` (V1 parity). With recognition the pipeline must return
    exactly one text per detected line.

    Raises:
        ValueError: When recognition ran and the number of line texts differs
            from the number of detected lines.
    """
    detections = output.detections
    if detections is None:
        empty = Detections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32, device=device),
            class_id=torch.zeros((0,), dtype=torch.int64, device=device),
            confidence=torch.zeros((0,), dtype=torch.float32, device=device),
            bboxes_metadata=[],
        )
        return empty

    if not recognition_enabled:
        line_texts = [""] * len(detections)
    elif len(output.line_texts) == len(detections):
        line_texts = output.line_texts
    else:
        raise ValueError(
            f"{block}: the PP-OCR recognizer returned {len(output.line_texts)} "
            f"line texts for {len(detections)} detected lines"
        )
    # Row metadata is never None here: the PP-OCRv6 detector sets one
    # {"polygon": ...} entry per box ([] without boxes), and the pipeline
    # indexes those entries to put the lines in reading order.
    detections.bboxes_metadata = [
        {**row, TEXT_KEY: text, CLASS_NAME_KEY: text}
        for row, text in zip(detections.bboxes_metadata, line_texts)
    ]

    return detections
