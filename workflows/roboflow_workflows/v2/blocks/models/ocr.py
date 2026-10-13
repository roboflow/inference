"""Optical character recognition with host-owned ``inference_models`` models.

Ports of the latest V1 ``roboflow_core/ocr_model@v1`` (DocTR),
``roboflow_core/easy_ocr@v1`` and ``roboflow_core/pp_ocr@v1``. Every image
gets the same four outputs::

    result        recognised text of the whole image (str)
    predictions   native Detections in the image's pixel coordinates, one row
                  per text region; each row's metadata holds its "text"
    model_id      model id that served the image; PP-OCR reports its
                  detection model, else its recognition model
    inference_id  fresh id; namespace of the rows' identity

    block            models from the 'models' resource      model calls
    v2/ocr_model     model_id, default "doctr/default"      pre_process, forward,
    v2/easy_ocr      "easy_ocr/<language model>"            post_process: one call
                                                             per phase and request
    v2/pp_ocr        "pp-ocrv6-det/<size>" and              PPOCRv6Pipeline
                     "pp-ocrv6-rec/<size>"                  .with_models(...).infer,
                                                             once per batch; run only

PP-OCR has no phases because its two models compose inside one public
pipeline call. With ``block_execution="phases"`` the engine calls its ``run``,
so its errors are plain ``ValueError`` and ``LookupError``, not the
``PhaseFailure`` of the phased DocTR and EasyOCR blocks.

Row names follow V1: a DocTR row is a ``block``, ``line`` or ``word`` (class
ids 0..2); an EasyOCR or PP-OCR row is named by its own recognised text (row
metadata ``"class"``), with ``text-region`` as the fallback class name.

Differences from V1, all deliberate:

- No remote execution mode. The host passes the models, as for every V2
  model block (see ``detection``); V1's HTTP path is not ported.
- ``parent_id``, ``root_parent_id`` and ``prediction_type`` are no outputs:
  the image's frame record and ``prediction_type`` ("ocr") live in
  ``predictions.image_metadata``. ``model_id`` and ``inference_id`` are new.
- EasyOCR has no ``quantize`` and sends no language codes. V1 sent both only
  to the remote HTTP API; its local model ignored them. ``language`` selects
  the model, which carries its own languages.
- PP-OCR calls its pipeline once per batch (V1: once per image). Inside that
  call the detector runs once over the batch and the recognizer once per
  image over that image's line crops; without detection the recognizer runs
  once over the batch. Rows keep the detector's ``"polygon"`` metadata. V1's
  ``detection_id`` is replaced by V2 row identity.
- PP-OCR fails when the recognizer returns another number of line texts than
  detected lines. V1 named the missing lines ``""`` and dropped extra texts.
- PP-OCR package ids are literal strings only. Workflow selector prefixes
  (``$inputs.``, ``$steps.``, ``$sources.``, ``$operators.``) fail compilation;
  supply the actual package id instead. DocTR and EasyOCR package ids also
  accept selectors.
- EasyOCR's empty result carries float class ids; the block makes them
  integer so that a text-free image is a valid detection prediction.

Compilation never loads a model and nothing here closes one; models must
support concurrent calls as the ``inference_models`` OCR backends do (DocTR
and EasyOCR lock their ``forward``).
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import torch
from inference_models.models.base.object_detection import Detections
from inference_models.models.pp_ocrv6.pp_ocrv6_pipeline import PPOCRv6Pipeline
from pydantic import Field, StrictStr, ValidationInfo, field_validator, model_validator
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.common import (
    INFERENCE_ID_OUTPUT,
    MODEL_ID_OUTPUT,
    MODELS_RESOURCE,
    PACKAGE_ID_DESCRIPTION,
    ModelGroup,
    ModelIdField,
    ModelOption,
    ModelSelectionParams,
    PackageIdField,
    RequestField,
    checked_count,
    forward_groups,
    identified_result,
    model_dependency,
    models_cache,
    prediction_metadata,
    preprocess_groups,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
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
from roboflow_workflows.v2.engine.phases import phase

PREDICTION_TYPE = "ocr"
TEXT_KEY = "text"
TEXT_REGION_CLASS_NAMES = ("text-region",)
DOCTR_MODEL_ID = "doctr/default"

# Displayed language -> EasyOCR model id (V1 ``MODELS``, without the language
# codes only its remote API read).
EASY_OCR_MODELS: Mapping[str, str] = {
    "English": "easy_ocr/english_g2",
    "Japanese": "easy_ocr/japanese_g2",
    "Kannada": "easy_ocr/kannada_g2",
    "Korean": "easy_ocr/korean_g2",
    "Latin": "easy_ocr/latin_g2",
    "Telugu": "easy_ocr/telugu_g2",
    "Simplified Chinese": "easy_ocr/zh_sim_g2",
}
EasyOCRLanguage = Literal[
    "English",
    "Japanese",
    "Kannada",
    "Korean",
    "Latin",
    "Telugu",
    "Simplified Chinese",
]
# V1 passes confidence 0.0: EasyOCR keeps every region it reads.
EASY_OCR_OPTIONS = (
    ModelOption(param="confidence", keyword="confidence", value=None, default=0.0),
)

PPOCRStage = Literal["none", "tiny", "small", "medium"]
PP_OCR_DETECTION_MODEL = "pp-ocrv6-det/{size}"
PP_OCR_RECOGNITION_MODEL = "pp-ocrv6-rec/{size}"
PP_OCR_STAGE_REQUIRED = (
    "PP-OCR requires at least one of text detection or text recognition to be "
    "enabled"
)
PP_OCR_MODEL_ID_OUTPUT = Output(
    STRING_KIND,
    source="image",
    description=(
        "Id of the detection model that served this image; the recognition "
        "model's id when text detection is 'none'."
    ),
)

RESULT_OUTPUT = Output(
    STRING_KIND, source="image", description="Recognised text of the whole image."
)
IMAGE_DESCRIPTION = "RGB images to read text from; delivered as one batch."


def _ocr_outputs(
    predictions_description: str, *, model_id: Output = MODEL_ID_OUTPUT
) -> Dict[str, Output]:
    outputs = {
        "result": RESULT_OUTPUT,
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description=predictions_description,
        ),
        "model_id": model_id,
        "inference_id": INFERENCE_ID_OUTPUT,
    }

    return outputs


class OCRModelBlock(Block):
    """Read text with DocTR models (``StructuredOCRModel``) from ``models``.

    The phases are the model's ``pre_process``, ``forward`` and
    ``post_process``, each called once per distinct model request; ``run``
    composes the same three. ``predictions`` hold the model's word, line and
    block boxes with their ``"text"``; the model's ``class_names`` name them.
    """

    type = "v2/ocr_model@v1"
    metadata = {
        "name": "OCR Model",
        "port_of": "roboflow_core/ocr_model@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "doctr"],
    }
    outputs = _ocr_outputs(
        "Native Detections of words, lines and blocks with their 'text'."
    )

    class Params(ModelSelectionParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description=IMAGE_DESCRIPTION
        )
        model_id: ModelIdField = Field(
            default=DOCTR_MODEL_ID,
            description=(
                "Model id (or local path) of a DocTR model in the 'models' "
                "resource; a selector may choose it per run or per image."
            ),
            examples=[DOCTR_MODEL_ID, "$inputs.model_id"],
        )

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource."""
        resources = model_dependency(params, task_type=PREDICTION_TYPE)

        return resources

    @phase
    def network_input(
        self, image: Batch, model_id: RequestField, model_package_id: RequestField
    ) -> Tuple[ModelGroup, ...]:
        """Resolve each row's model and preprocess each group."""
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=model_id,
            model_package_id=model_package_id,
            options=(),
            block=self.type,
        )

        return groups

    @phase
    def raw_output(self, network_input: Tuple[ModelGroup, ...]) -> Tuple[Any, ...]:
        """Run each group's model once on its rows (``forward``)."""
        raw_output = forward_groups(network_input)

        return raw_output

    @phase
    def result(
        self,
        raw_output: Tuple[Any, ...],
        network_input: Tuple[ModelGroup, ...],
        image: Batch,
    ) -> List[Dict[str, Any]]:
        """Decode text and boxes (``post_process``); one result per image."""
        results = _structured_results(raw_output, network_input, image, block=self.type)

        return results

    def run(
        self,
        *,
        image: Batch,
        model_id: RequestField = DOCTR_MODEL_ID,
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Read the text of every image of the batch.

        Args:
            image: The batch of RGB images of one workflow run.
            model_id: Model id for every image, or a ``Batch`` with one per
                image.
            model_package_id: Package id for every image, a ``Batch`` with one
                per image, or ``None`` to let the loader choose.

        Returns:
            One ``{"result", "predictions", "model_id", "inference_id"}`` per
            image, in batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when a model cannot be
                resolved or an image is not RGB; phase ``result`` when a model
                returns another number of results than images.
        """
        network_input = self.network_input(image, model_id, model_package_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results


class EasyOCRBlock(Block):
    """Read text with the EasyOCR model of one language from ``models``.

    ``language`` selects the model id (``EASY_OCR_MODELS``). Phases as in
    ``OCRModelBlock``; the model keeps every region it reads (confidence 0,
    as V1). Each row is named by its own recognised text.
    """

    type = "v2/easy_ocr@v1"
    metadata = {
        "name": "EasyOCR",
        "port_of": "roboflow_core/easy_ocr@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "easyocr"],
    }
    outputs = _ocr_outputs(
        "Native Detections of text regions, each named by its recognised text."
    )

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description=IMAGE_DESCRIPTION
        )
        language: EasyOCRLanguage = Field(
            default="English",
            title="Language",
            description=(
                "Language of the EasyOCR model to use; each language has its own "
                "model id in the 'models' resource (e.g. English -> "
                "easy_ocr/english_g2)."
            ),
            examples=["English", "Japanese"],
        )
        model_package_id: PackageIdField = Field(
            default=None,
            description=PACKAGE_ID_DESCRIPTION,
            examples=[None, "$inputs.model_package_id"],
        )

        @field_validator("model_package_id")
        @classmethod
        def _reject_empty(cls, value: Optional[str]) -> Optional[str]:
            if value == "":
                raise ValueError("must be a non-empty string")

            return value

    def __init__(self, *, models: Union[Mapping[Any, Any], ModelCache]):
        self._models = models_cache(models, block=self.type)

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the language's model this step needs from ``models``."""
        resources = [
            _model_resource(
                EASY_OCR_MODELS[params.language],
                model_package_id=params.model_package_id,
            )
        ]

        return resources

    @phase
    def network_input(
        self, image: Batch, language: str, model_package_id: RequestField
    ) -> Tuple[ModelGroup, ...]:
        """Resolve the language's model and preprocess each group."""
        groups = preprocess_groups(
            image,
            cache=self._models,
            model_id=EASY_OCR_MODELS[language],
            model_package_id=model_package_id,
            options=EASY_OCR_OPTIONS,
            block=self.type,
        )

        return groups

    @phase
    def raw_output(self, network_input: Tuple[ModelGroup, ...]) -> Tuple[Any, ...]:
        """Run each group's model once on its rows (``forward``)."""
        raw_output = forward_groups(network_input)

        return raw_output

    @phase
    def result(
        self,
        raw_output: Tuple[Any, ...],
        network_input: Tuple[ModelGroup, ...],
        image: Batch,
    ) -> List[Dict[str, Any]]:
        """Decode text and boxes (``post_process``); one result per image."""
        results = _structured_results(
            raw_output,
            network_input,
            image,
            block=self.type,
            adjust_detections=_easy_ocr_detections,
        )

        return results

    def run(
        self,
        *,
        image: Batch,
        language: str = "English",
        model_package_id: RequestField = None,
    ) -> List[Dict[str, Any]]:
        """Read the text of every image of the batch.

        Args:
            image: The batch of RGB images of one workflow run.
            language: Language whose model reads every image.
            model_package_id: Package id for every image, a ``Batch`` with one
                per image, or ``None`` to let the loader choose.

        Returns:
            One ``{"result", "predictions", "model_id", "inference_id"}`` per
            image, in batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when the model cannot be
                resolved or an image is not RGB; phase ``result`` when the
                model returns another number of results than images.
        """
        network_input = self.network_input(image, language, model_package_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image)

        return results


class PPOCRBlock(Block):
    """Read text with a PP-OCRv6 detector and recognizer from ``models``.

    The two models compose through ``PPOCRv6Pipeline.with_models``; either
    stage may be ``"none"``, not both. With detection, each row is one text
    line in reading order, named by its recognised text (``""`` without
    recognition) and keeping the detector's ``"polygon"``. Without detection
    every image is read as one pre-cropped line and has no rows.
    """

    type = "v2/pp_ocr@v1"
    metadata = {
        "name": "PP-OCR",
        "port_of": "roboflow_core/pp_ocr@v1",
        "section": "model",
        "search_keywords": ["ocr", "text", "paddle", "paddleocr"],
    }
    outputs = _ocr_outputs(
        "Native Detections of text lines in reading order, each named by its "
        "recognised text.",
        model_id=PP_OCR_MODEL_ID_OUTPUT,
    )

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description=IMAGE_DESCRIPTION
        )
        text_detection: PPOCRStage = Field(
            default="small",
            title="Text Detection Model Size",
            description=(
                "Text detection model size (model id pp-ocrv6-det/<size>). 'none' "
                "disables detection: each image is read as one pre-cropped text "
                "line."
            ),
            examples=["small", "none"],
        )
        text_recognition: PPOCRStage = Field(
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
            _model_resource(request.model_id, model_package_id=request.model_package_id)
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
            ValueError: When both stages are ``"none"``, an image is not RGB,
                the pipeline returns another number of results than images,
                or the recognizer returns another number of line texts than
                detected lines.
            LookupError: When a model is missing from or conflicts with the
                supplied models.
        """
        members = image.content
        for member in members:
            _require_rgb(member, block=self.type)
        detection, recognition = _pp_ocr_requests(
            text_detection=text_detection,
            text_recognition=text_recognition,
            text_detection_package_id=text_detection_package_id,
            text_recognition_package_id=text_recognition_package_id,
        )
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


def _structured_results(
    raw_output: Tuple[Any, ...],
    network_input: Tuple[ModelGroup, ...],
    image: Batch,
    *,
    block: str,
    adjust_detections: Optional[Callable[[Detections], Detections]] = None,
) -> List[Dict[str, Any]]:
    """Post-process each group of a ``StructuredOCRModel``; rows in batch order.

    ``adjust_detections``, if given, fixes up one image's detections first.
    """
    members = image.content
    results: List[Any] = [None] * len(members)
    for group, group_output in zip(network_input, raw_output):
        texts, detections_per_image = group.model.post_process(
            group_output, group.metadata, **group.options
        )
        checked_count(detections_per_image, group, block=block)
        checked_count(texts, group, block=block)

        for text, detections, row in zip(texts, detections_per_image, group.rows):
            if adjust_detections is not None:
                detections = adjust_detections(detections)
            results[row] = _ocr_result(
                text,
                detections,
                members[row],
                class_names=group.model.class_names,
                model_id=group.request.model_id,
            )

    return results


def _ocr_result(
    text: str,
    detections: Detections,
    image: ImageData,
    *,
    class_names: Tuple[str, ...],
    model_id: str,
) -> Dict[str, Any]:
    detections.image_metadata = prediction_metadata(
        detections.image_metadata,
        image,
        class_names=class_names,
        prediction_type=PREDICTION_TYPE,
    )
    result = {"result": text, **identified_result(detections, model_id=model_id)}

    return result


def _easy_ocr_detections(detections: Detections) -> Detections:
    """Name each row by its recognised text and give empty class ids an int type."""
    # V1 parity: an EasyOCR row's class is its recognised text.
    for row in detections.bboxes_metadata or ():
        row[CLASS_NAME_KEY] = str(row.get(TEXT_KEY, ""))
    # EasyOCR builds class ids with torch.tensor(list): no text gives float32.
    if detections.class_id.is_floating_point() and not detections.class_id.numel():
        detections.class_id = detections.class_id.to(torch.int64)

    return detections


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


def _model_resource(
    model_id: str, *, model_package_id: Optional[str]
) -> DependentResource:
    resource = DependentResource(
        resource_type="model",
        identifier=model_id,
        details={
            "resource": MODELS_RESOURCE,
            "task_type": PREDICTION_TYPE,
            "model_package_id": model_package_id,
        },
    )

    return resource


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")
