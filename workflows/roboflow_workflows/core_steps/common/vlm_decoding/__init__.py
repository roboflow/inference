"""Shared decoding of VLM answers into workflow predictions.

VLM blocks used to emit a raw string that a separate "VLM as Detector" /
"VLM as Classifier" formatter block parsed. This package moves the decoding
into the VLM blocks themselves and keys it by the BOX COORDINATE FORMAT the
prompt asked for rather than by model vendor, so vendors sharing a
coordinate contract share one parser and one prompt wording.
"""

from roboflow_workflows.core_steps.common.vlm_decoding.classification import (
    decode_classification,
    parse_multi_class_classification_results,
    parse_multi_label_classification_results,
)
from roboflow_workflows.core_steps.common.vlm_decoding.detection_formats import (
    BBOX_KEYS,
    BOX_2D_KEYS,
    DETECTION_BOX_FORMATS,
    LABEL_KEYS,
    NAMED_0_1000_PROMPT_TEMPLATE,
    NAMED_BOX_FIELDS,
    XYXY_0_1000_PROMPT_TEMPLATE,
    XYXY_ABSOLUTE_BBOX_PROMPT_TEMPLATE,
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
from roboflow_workflows.core_steps.common.vlm_decoding.detections import (
    build_detections,
    decode_object_detections,
)
from roboflow_workflows.core_steps.common.vlm_decoding.json_extraction import (
    extract_flat_object_entries,
    extract_json,
)
from roboflow_workflows.core_steps.common.vlm_decoding.outputs import (
    CLASSIFICATION_TASKS,
    DETECTION_TASKS,
    LEGACY_PREDICTION_KINDS_UNION,
    SEGMENTATION_TASKS,
    actual_vlm_prediction_outputs,
    decode_vlm_output,
    describe_vlm_prediction_outputs,
    prediction_kinds_for_tasks,
)
from roboflow_workflows.core_steps.common.vlm_decoding.segmentation import (
    INSTANCE_SEGMENTATION_PROMPT_TEMPLATE,
    build_instance_segmentation_prompt,
    build_instance_segmentations,
    decode_instance_segmentations,
    extract_segmentation_entries,
    polygon_to_rle,
    read_polygon,
)
from roboflow_workflows.core_steps.common.vlm_decoding.tensor_native import (
    tensor_native_carriers_enabled,
    to_tensor_native_predictions,
)
from roboflow_workflows.core_steps.common.vlm_decoding.utils import (
    create_classes_index,
    scale_confidence,
)

__all__ = [
    "BBOX_KEYS",
    "BOX_2D_KEYS",
    "CLASSIFICATION_TASKS",
    "DETECTION_BOX_FORMATS",
    "DETECTION_TASKS",
    "INSTANCE_SEGMENTATION_PROMPT_TEMPLATE",
    "LABEL_KEYS",
    "NAMED_0_1000_PROMPT_TEMPLATE",
    "LEGACY_PREDICTION_KINDS_UNION",
    "NAMED_BOX_FIELDS",
    "SEGMENTATION_TASKS",
    "XYXY_0_1000_PROMPT_TEMPLATE",
    "XYXY_ABSOLUTE_BBOX_PROMPT_TEMPLATE",
    "XYXY_ABSOLUTE_PROMPT_TEMPLATE",
    "XYXY_PERCENT_PROMPT_TEMPLATE",
    "YXYX_0_1000_PROMPT_TEMPLATE",
    "BoxFormatName",
    "DetectionBoxFormat",
    "actual_vlm_prediction_outputs",
    "build_detections",
    "build_instance_segmentation_prompt",
    "build_instance_segmentations",
    "build_object_detection_prompt",
    "create_classes_index",
    "decode_classification",
    "decode_instance_segmentations",
    "decode_object_detections",
    "decode_vlm_output",
    "describe_vlm_prediction_outputs",
    "extract_detection_entries",
    "extract_flat_object_entries",
    "extract_json",
    "extract_segmentation_entries",
    "get_detection_box_format",
    "get_detection_class_name",
    "get_detection_confidence",
    "parse_multi_class_classification_results",
    "parse_multi_label_classification_results",
    "polygon_to_rle",
    "prediction_kinds_for_tasks",
    "read_polygon",
    "scale_confidence",
    "tensor_native_carriers_enabled",
    "to_tensor_native_predictions",
]
