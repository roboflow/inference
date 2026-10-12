"""Batched object detection with a caller-loaded ``inference_models`` model.

One call per workflow run, one model call per phase::

    image (Batch of N, CHW RGB) ── network_input: model.pre_process([N images])
                                   raw_output:    model.forward(N x 3 x H x W)
                                   result:        model.post_process -> N Detections

The host loads the models and passes them as the ``models`` resource, keyed by
model id. A step names its model with the literal ``model_id``; the step's
``discover_dependent_resources`` reports that id so the host knows what to
load. For example: ``plan.create_session({"models": {"rfdetr-nano": model}})``.
Nothing here downloads, looks up or closes a model.

The block copies the ``models`` mapping when it is constructed, not the model
objects. Changing the host's dictionary afterwards does not change a running
block: bind a different model with a new mapping and a processing reset. Each
call resolves its model once, in ``network_input``, and the later phases use
that same object through the phase values.
"""

from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple

from pydantic import Field, StrictFloat, StrictStr, field_validator
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    DependentResource,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase

PREDICTION_TYPE = "object-detection"
DEFAULT_CONFIDENCE = 0.4
MODELS_RESOURCE = "models"


class _NetworkInput(NamedTuple):
    model: Any
    batch: Any
    metadata: Any


class ObjectDetectionBlock(Block):
    """Detect objects in a batch of images with one caller-loaded model.

    The phases are the model's own ``pre_process``, ``forward`` and
    ``post_process``, each called once for the whole batch; ``run`` composes
    the same three. ``predictions`` are the model's native ``Detections`` in
    each image's pixel coordinates. Their ``image_metadata`` keeps what the
    model wrote and adds the image's ``prediction_metadata()``, the model's
    ``class_names`` and ``prediction_type``.
    """

    type = "v2/object_detection@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native Detections in this image's pixel coordinates.",
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to detect objects in; delivered as one batch."
        )
        model_id: StrictStr = Field(
            min_length=1,
            description=(
                "Key of the caller-loaded model in the 'models' resource. "
                "Literal only; the host loads the model before the session."
            ),
            examples=["rfdetr-nano", "yolov8n-640"],
        )
        confidence: StrictFloat | Ref(FLOAT_KIND) = Field(
            default=DEFAULT_CONFIDENCE,
            ge=0,
            le=1,
            description="Minimum detection confidence, 0..1.",
            examples=[0.4],
        )

        @field_validator("model_id")
        @classmethod
        def _reject_selector(cls, model_id: str) -> str:
            # Without this, "$inputs.model" would compile as a literal id and
            # only fail at run time as a missing model.
            if model_id.startswith("$"):
                raise ValueError(
                    "model_id is literal only; a selector cannot choose the model"
                )

            return model_id

    def __init__(self, *, models: Mapping[str, Any]):
        if not isinstance(models, Mapping):
            raise TypeError(
                f"{self.type} needs the '{MODELS_RESOURCE}' resource as a mapping "
                f"of model id to loaded model, got {type(models).__name__}"
            )

        self._models: Mapping[str, Any] = MappingProxyType(dict(models))

    @classmethod
    def discover_dependent_resources(
        cls, params: BlockParams
    ) -> List[DependentResource]:
        """Report the model this step needs from the ``models`` resource.

        Args:
            params: Validated step parameters.

        Returns:
            One ``model`` resource named by ``model_id``.
        """
        resources = [
            DependentResource(
                resource_type="model",
                identifier=params.model_id,
                details={"resource": MODELS_RESOURCE, "task_type": PREDICTION_TYPE},
            )
        ]

        return resources

    @phase
    def network_input(self, image: Batch, model_id: str) -> _NetworkInput:
        """Resolve the model and preprocess every image (``pre_process``)."""
        model = self._model(model_id)
        for member in image:
            _require_rgb(member, block=self.type)

        batch, metadata = model.pre_process(
            [member.tensor_image for member in image], input_color_format="rgb"
        )
        network_input = _NetworkInput(model=model, batch=batch, metadata=metadata)

        return network_input

    @phase
    def raw_output(self, network_input: _NetworkInput) -> Any:
        """Run the network once on the whole batch (``forward``)."""
        raw_output = network_input.model.forward(network_input.batch)

        return raw_output

    @phase
    def result(
        self,
        raw_output: Any,
        network_input: _NetworkInput,
        image: Batch,
        confidence: float,
    ) -> List[Dict[str, Any]]:
        """Decode boxes (``post_process``); one result per image, in batch order."""
        model = network_input.model
        batch = model.post_process(
            raw_output, network_input.metadata, confidence=confidence
        )
        if len(batch) != len(image):
            raise ValueError(
                f"{self.type}: post_process returned {len(batch)} predictions "
                f"for {len(image)} images"
            )

        class_names = dict(enumerate(model.class_names))
        for predictions, member in zip(batch, image):
            predictions.image_metadata = {
                **(predictions.image_metadata or {}),
                **member.prediction_metadata(),
                "class_names": class_names,
                "prediction_type": PREDICTION_TYPE,
            }
        results = [{"predictions": predictions} for predictions in batch]

        return results

    def run(
        self, *, image: Batch, model_id: str, confidence: float
    ) -> List[Dict[str, Any]]:
        """Detect objects in every image of the batch.

        The same three phases a pipelined run executes as separate stages.

        Args:
            image: The batch of RGB images of one workflow run.
            model_id: Key of the model in the ``models`` resource.
            confidence: Minimum detection confidence.

        Returns:
            One ``{"predictions": Detections}`` per image, in batch order.

        Raises:
            PhaseFailure: Phase ``network_input`` when ``model_id`` is not in
                the ``models`` resource or an image is not RGB; phase
                ``result`` when the model returns another number of predictions.
        """
        network_input = self.network_input(image, model_id)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image, confidence)

        return results

    def _model(self, model_id: str) -> Any:
        model = self._models.get(model_id)
        if model is None:
            available = sorted(self._models)
            raise LookupError(
                f"{self.type}: model_id {model_id!r} is not in the "
                f"'{MODELS_RESOURCE}' resource (available: {available}); load it "
                "and pass it in that mapping when creating the session"
            )

        return model


def _require_rgb(image: ImageData, *, block: str) -> None:
    if image.channels != 3:
        raise ValueError(f"{block} needs a 3-channel RGB image, got {image!r}")
