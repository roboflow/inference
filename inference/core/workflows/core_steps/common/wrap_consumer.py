"""Generic wrap-and-delegate factory for tensor-mode consumer block
siblings.

Most consumer blocks (visualizers, filters, mutators that don't trivially
port to inference_models native shape, sinks, classical CV helpers) have
heavily sv.Detections-shaped internals. Reimplementing each natively is
expensive and orthogonal to the workflow tensor-data-representation goal
— what matters at the workflow-engine level is the type contract at
block boundaries.

`make_tensor_wrapper_block(NumpyBlockClass)` returns a subclass that
materialises any inference_models native input (`Detections`,
`InstanceDetections`, `KeyPoints`) to sv at the boundary, delegates to
the wrapped numpy `run`, and returns the result as-is. For consumers
whose output is an image / a sink-side effect / sv-shaped predictions,
this is sufficient.

For consumers that emit predictions and need the output back in
inference_models native form for downstream tensor consumers, write a
custom tensor sibling (see
`fusion/detections_classes_replacement/v1_tensor.py`) using the
to_supervision helpers directly.

The materialisation cost is paid by the block, not the engine
(per `[ITERATE PRED.6]`). The class returned shares the numpy block's
`__name__` / `__qualname__` so the loader's `if/else` swap binds the
same identifier in both branches.
"""
from typing import Any, List, Type

import supervision as sv

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

from inference.core.workflows.core_steps.common.to_supervision import (
    classification_prediction_to_dict_per_image,
    multi_label_classification_to_dict,
    sv_detections_to_inference_models_detections,
    to_supervision_with_metadata,
)
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.prototypes.block import WorkflowBlock


def make_tensor_wrapper_block(numpy_block_cls: Type[WorkflowBlock]) -> Type[WorkflowBlock]:
    """Build a tensor-mode sibling for a numpy consumer block.

    The returned class:
    - shares the wrapped class's name (loader if/else binds the same
      identifier in both branches)
    - inherits the wrapped class's manifest, init parameters, signature
    - overrides `run` to materialise inference_models native inputs to sv
      at the boundary before delegating to `super().run(...)`
    """
    class _TensorWrapperBlock(numpy_block_cls):  # type: ignore[misc]
        def run(self, *args: Any, **kwargs: Any) -> Any:
            args = tuple(_maybe_materialise(a) for a in args)
            kwargs = {k: _maybe_materialise(v) for k, v in kwargs.items()}
            result = super().run(*args, **kwargs)
            # Convert any sv.Detections in the output back to
            # inference_models.Detections so downstream tensor-native
            # consumers see a consistent type. Sinks/visualizers whose
            # outputs are images / status dicts are unaffected (passthrough).
            return _maybe_unmaterialise(result)

    _TensorWrapperBlock.__name__ = numpy_block_cls.__name__
    _TensorWrapperBlock.__qualname__ = numpy_block_cls.__qualname__
    _TensorWrapperBlock.__module__ = numpy_block_cls.__module__
    _TensorWrapperBlock.__doc__ = (
        f"Tensor-mode sibling of {numpy_block_cls.__name__}, wrapping the "
        f"numpy implementation. Materialises inference_models native "
        f"prediction inputs to sv.Detections at the boundary before "
        f"delegating to the wrapped block."
    )
    return _TensorWrapperBlock


def _maybe_materialise(value: Any) -> Any:
    if isinstance(value, (Detections, InstanceDetections, KeyPoints)):
        return to_supervision_with_metadata(value)
    if isinstance(value, ClassificationPrediction):
        # Single-label batch-shaped — most consumers handle one dict per
        # image. Returning the list keeps the consumer's expectation
        # intact. Wrap in a Batch if upstream was a Batch.
        return classification_prediction_to_dict_per_image(value)
    if isinstance(value, MultiLabelClassificationPrediction):
        return multi_label_classification_to_dict(value)
    if isinstance(value, Batch):
        return Batch(
            content=[_maybe_materialise(v) for v in value._content],
            indices=value._indices,
        )
    if isinstance(value, list):
        return [_maybe_materialise(v) for v in value]
    return value


def _maybe_unmaterialise(value: Any) -> Any:
    """Reverse direction at the output boundary: convert sv-shaped values
    back to inference_models native so downstream consumers see a
    consistent type. Recurses through dicts and lists (BlockResult shape).
    Passthrough for everything else (images, status payloads, scalars)."""
    if isinstance(value, sv.Detections):
        return sv_detections_to_inference_models_detections(value)
    if isinstance(value, dict):
        return {k: _maybe_unmaterialise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_maybe_unmaterialise(v) for v in value]
    return value
