"""Turn OCR model output into the four outputs every OCR block returns.

The DocTR (``roboflow_core/ocr_model@v1``), EasyOCR
(``roboflow_core/easy_ocr@v1``) and PP-OCR (``roboflow_core/pp_ocr@v1``)
blocks return the same outputs for each image::

    result        recognised text of the whole image (str)
    predictions   native Detections in the image's pixel coordinates, one row
                  per text region; each row's metadata holds its "text"
    model_id      model id that served the image
    inference_id  fresh id; namespace of the rows' identity

Differences from V1 shared by these blocks, all deliberate:

- No remote execution mode. The host passes the models, as for every V2
  model block; V1's HTTP path is not ported.
- ``parent_id``, ``root_parent_id`` and ``prediction_type`` are no outputs:
  the image's frame record and ``prediction_type`` ("ocr") live in
  ``predictions.image_metadata``. ``model_id`` and ``inference_id`` are new.

Compilation never loads a model and nothing here closes one; models must
support concurrent calls as the ``inference_models`` OCR backends do (DocTR
and EasyOCR lock their ``forward``).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from inference_models.models.base.object_detection import Detections
from roboflow_workflows.v2.blocks.models.common import (
    ModelInput,
    checked_count,
    identified_result,
    prediction_metadata,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData


PREDICTION_TYPE = "ocr"

TEXT_KEY = "text"


def _structured_results(
    raw_output: Any,
    network_input: ModelInput,
    image: Batch,
    *,
    block: str,
    adjust_detections: Optional[Callable[[Detections], Detections]] = None,
) -> List[Dict[str, Any]]:
    """Post-process a ``StructuredOCRModel``'s output; results in batch order.

    ``adjust_detections``, if given, fixes up one image's detections first.
    """
    model = network_input.model
    texts, detections_per_image = model.post_process(
        raw_output, network_input.metadata, **network_input.options
    )
    for outputs in (detections_per_image, texts):
        checked_count(
            outputs,
            image_count=len(image),
            request=network_input.request,
            block=block,
        )

    results = []
    for text, detections, member in zip(texts, detections_per_image, image):
        if adjust_detections is not None:
            detections = adjust_detections(detections)
        results.append(
            _ocr_result(
                text,
                detections,
                member,
                class_names=model.class_names,
                model_id=network_input.request.model_id,
            )
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
