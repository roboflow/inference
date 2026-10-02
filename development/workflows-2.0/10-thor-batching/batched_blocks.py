"""A batched V2 detector next to the 08 per-image detector and the 09 painters.

    images (Batch of N, CUDA CHW RGB) ─┬─ detector: ONE call per batch ── N predictions ─┐
                                       │    network_input  pre_process([N images])        │
                                       │    raw_output     forward(N x 3 x H x W)         │
                                       │    result         post_process -> N Detections   │
                                       └─ boxes (09 painter, one call per image) ── labels (one call per image)

The detector declares ``image`` with ``batch="always"``, so the engine calls it
once with every image of a run (``PlannedStep.delivers_batches``) and expects a
list of per-image results back, in input order. A run with a
``WorkflowBatchInput`` of N images is therefore one physical model batch.

Authoring note. One ``batch="always"`` block is enough: in a workflow with a
single ``WorkflowParameter`` image it runs as a batch of one. The 08 per-image
detector stays in the catalogue only to reproduce the 09 reference workflow
(``batched_backend.v2_workflow(stages, batched=False)``). Subclassing 08 is demo
reuse, not the recommended shape for a batched block. From 08 the class inherits
``__init__`` (the ``detection_model`` resource), the ``confidence`` parameter and
the ``raw_output`` phase, which already handles N because ``forward`` takes one
stacked batch. A new batched block would declare these itself.

The painters are the 09 ``gpu_blocks`` classes, unchanged. They stay per-image
blocks: a batch of N costs N box calls and N label calls, serially, inside their
own stage. That work is visible in each row's ``boxes_ms`` / ``labels_ms``.

Readiness is the 09 contract: the TRT model synchronizes its own streams inside
``pre_process``, ``forward`` and ``post_process``, so each phase returns ready
values, and the painters wait for their own GPU work before returning.
"""

from time import perf_counter
from typing import Any, Dict, List

import thor_imports  # noqa: F401 - installs the 08 and 09 search paths
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import Output, Ref
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase

# isort: split

from detection_blocks import (
    PREDICTION_TYPE,
    ObjectDetector,
    _elapsed_ms,
    _NetworkInput,
    _RawOutput,
    _require_rgb,
)
from gpu_blocks import GpuBoxVisualization, GpuLabelVisualization


class BatchedObjectDetector(ObjectDetector):
    """The 08 detector, called once per batch of images instead of once per image.

    ``run`` and the phases take the whole batch: ``network_input`` and
    ``result`` work on lists; ``raw_output`` (inherited) runs ``forward`` once
    on the stacked batch. The ``*_ms`` outputs are the wall times of the whole
    batch's phase, repeated in every image's row.
    """

    type = "thor_batching/object_detector@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Native Detections in this image's pixel coordinates.",
        ),
        "pre_ms": Output(
            FLOAT_KIND,
            source="image",
            description="pre_process wall time of the whole batch, ms.",
        ),
        "model_ms": Output(
            FLOAT_KIND,
            source="image",
            description="forward wall time of the whole batch, ms.",
        ),
        "post_ms": Output(
            FLOAT_KIND,
            source="image",
            description="post_process and metadata wall time of the whole batch, ms.",
        ),
    }

    class Params(ObjectDetector.Params):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to detect objects in; delivered as one batch."
        )

    @phase
    def network_input(self, image: Batch) -> _NetworkInput:
        """Resize and normalize every image of the batch (``pre_process``)."""
        for member in image:
            _require_rgb(member, block=self.type)

        started = perf_counter()
        batch, metadata = self.model.pre_process(
            [member.tensor_image for member in image], input_color_format="rgb"
        )
        network_input = _NetworkInput(batch, metadata, _elapsed_ms(started))

        return network_input

    @phase
    def result(
        self,
        raw_output: _RawOutput,
        network_input: _NetworkInput,
        image: Batch,
        confidence: float,
    ) -> List[Dict[str, Any]]:
        """Decode boxes (``post_process``); one result per image, in batch order."""
        started = perf_counter()
        batch = self.model.post_process(
            raw_output.output, network_input.metadata, confidence=confidence
        )
        if len(batch) != len(image):
            raise ValueError(
                f"post_process returned {len(batch)} predictions for {len(image)} images"
            )

        class_names = dict(enumerate(self.model.class_names))
        for predictions, member in zip(batch, image):
            predictions.image_metadata = {
                **(predictions.image_metadata or {}),
                **member.prediction_metadata(),
                "class_names": class_names,
                "prediction_type": PREDICTION_TYPE,
            }
        post_ms = _elapsed_ms(started)
        results = [
            {
                "predictions": predictions,
                "pre_ms": network_input.pre_ms,
                "model_ms": raw_output.model_ms,
                "post_ms": post_ms,
            }
            for predictions in batch
        ]

        return results

    def run(self, *, image: Batch, confidence: float) -> List[Dict[str, Any]]:
        """Detect objects in every image of the batch.

        The same three phases a pipelined run executes as separate stages:
        one ``pre_process`` of all images, one ``forward``, one
        ``post_process``.

        Args:
            image: The batch of RGB images of one workflow run.
            confidence: Minimum detection confidence.

        Returns:
            One result per image, in batch order: ``predictions`` and the
            whole-batch ``pre_ms``, ``model_ms`` and ``post_ms``.

        Raises:
            PhaseFailure: Phase ``network_input`` on a non-RGB image; phase
                ``result`` when the model returns another number of predictions.
        """
        network_input = self.network_input(image)
        raw_output = self.raw_output(network_input)
        results = self.result(raw_output, network_input, image, confidence)

        return results


BLOCKS = (
    BatchedObjectDetector,
    ObjectDetector,
    GpuBoxVisualization,
    GpuLabelVisualization,
)


def create_catalogue() -> Catalogue:
    """Collect ``BLOCKS``: both detectors and the two 09 painters.

    Compiles either variant of ``batched_backend.v2_workflow``.

    Returns:
        A catalogue with ``thor_batching/object_detector@v1``,
        ``live_detection/object_detector@v1``,
        ``thor_detection/box_visualization@v1`` and
        ``thor_detection/label_visualization@v1``.
    """
    catalogue = Catalogue(
        BLOCKS,
        kinds=[IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, FLOAT_KIND],
    )

    return catalogue
