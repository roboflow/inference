"""Shared decoding of VLM answers into workflow predictions.

VLM blocks used to emit a raw string that a separate "VLM as Detector" /
"VLM as Classifier" formatter block parsed. This package moves the decoding
into the VLM blocks themselves and keys it by the BOX COORDINATE FORMAT the
prompt asked for rather than by model vendor, so vendors sharing a
coordinate contract share one parser and one prompt wording.
"""

from inference.core.workflows.core_steps.common.vlm_decoding.classification import (
    decode_classification,
    parse_multi_class_classification_results,
    parse_multi_label_classification_results,
)
from inference.core.workflows.core_steps.common.vlm_decoding.detection_formats import (
    BOX_2D_KEYS,
    DETECTION_BOX_FORMATS,
    LABEL_KEYS,
    NAMED_0_1000_PROMPT_TEMPLATE,
    NAMED_BOX_FIELDS,
    XYXY_0_1000_PROMPT_TEMPLATE,
    XYXY_ABSOLUTE_PROMPT_TEMPLATE,
    XYXY_PERCENT_PROMPT_TEMPLATE,
    YXYX_0_1000_PROMPT_TEMPLATE,
    BoxFormatName,
    DetectionBoxFormat,
    build_object_detection_prompt,
    extract_detection_entries,
    get_detection_box_format,
    get_detection_class_name,
    get_detection_confidence,
)
from inference.core.workflows.core_steps.common.vlm_decoding.detections import (
    build_detections,
    decode_object_detections,
)
from inference.core.workflows.core_steps.common.vlm_decoding.json_extraction import (
    JSON_MARKDOWN_BLOCK_PATTERN,
    extract_flat_object_entries,
    extract_json,
    extract_zai_json_array,
    try_parse_json,
)
from inference.core.workflows.core_steps.common.vlm_decoding.outputs import (
    CLASSIFICATION_TASKS,
    DETECTION_TASKS,
    actual_vlm_prediction_outputs,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
)
from inference.core.workflows.core_steps.common.vlm_decoding.tensor_native import (
    tensor_native_carriers_enabled,
    to_tensor_native_predictions,
)
from inference.core.workflows.core_steps.common.vlm_decoding.utils import (
    create_classes_index,
    scale_confidence,
)

__all__ = [
    "BOX_2D_KEYS",
    "CLASSIFICATION_TASKS",
    "DETECTION_BOX_FORMATS",
    "DETECTION_TASKS",
    "JSON_MARKDOWN_BLOCK_PATTERN",
    "LABEL_KEYS",
    "NAMED_0_1000_PROMPT_TEMPLATE",
    "NAMED_BOX_FIELDS",
    "XYXY_0_1000_PROMPT_TEMPLATE",
    "XYXY_ABSOLUTE_PROMPT_TEMPLATE",
    "XYXY_PERCENT_PROMPT_TEMPLATE",
    "YXYX_0_1000_PROMPT_TEMPLATE",
    "BoxFormatName",
    "DetectionBoxFormat",
    "actual_vlm_prediction_outputs",
    "build_detections",
    "build_object_detection_prompt",
    "create_classes_index",
    "decode_classification",
    "decode_object_detections",
    "decode_vlm_output",
    "describe_vlm_prediction_outputs",
    "extract_detection_entries",
    "extract_flat_object_entries",
    "extract_json",
    "extract_zai_json_array",
    "get_detection_box_format",
    "get_detection_class_name",
    "get_detection_confidence",
    "parse_multi_class_classification_results",
    "parse_multi_label_classification_results",
    "scale_confidence",
    "tensor_native_carriers_enabled",
    "to_tensor_native_predictions",
    "try_parse_json",
]
