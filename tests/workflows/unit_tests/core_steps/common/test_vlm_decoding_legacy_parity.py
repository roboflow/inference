"""Parity between in-block VLM decoding and the legacy ``vlm_as_detector@v2``.

Every sample is a verbatim model answer captured from a live run of the
corresponding block on the vlm-exam detection benchmark. Feeding the same
answer through the deprecated formatter block and through the shared decoder
must yield the same boxes and classes, so workflows migrating off the
formatter see no change in predictions.
"""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.vlm_decoding import decode_vlm_output
from inference.core.workflows.core_steps.formatters.vlm_as_detector.v2 import (
    VLMAsDetectorBlockV2,
)
from inference.core.workflows.core_steps.models.foundation.anthropic_claude import (
    v5 as claude_v5,
)
from inference.core.workflows.core_steps.models.foundation.openai import v7 as openai_v7
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from tests.workflows.unit_tests.core_steps._vlm_prediction_readers import (
    detection_boxes,
    detection_class_ids,
    detection_count,
    is_detection_prediction,
)

CLAUDE = "claude"
OPENAI = "openai"

# (family, legacy model_type, box_format, image (w, h), classes, raw answer)
SAMPLES = [
    (
        CLAUDE,
        "anthropic-claude",
        "xyxy_absolute",
        (3216, 2136),
        ["cracked insulator"],
        '```json\n[\n  {"box_2d": [953, 838, 1170, 1000], "label": "cracked insulator"},\n'
        '  {"box_2d": [1100, 934, 1367, 1116], "label": "cracked insulator"}\n]\n```',
    ),
    (
        OPENAI,
        "openai",
        "xyxy_absolute",
        (640, 640),
        ["fallen person"],
        '{"detections":[{"label":"fallen person","box_2d":[224,306,343,374]},'
        '{"label":"fallen person","box_2d":[292,333,432,414]}]}',
    ),
    (
        "gemini",
        "google-gemini",
        "yxyx_0_1000",
        (640, 640),
        ["fallen person"],
        '[\n  {"box_2d": [513, 429, 622, 590], "label": "fallen person"},\n'
        '  {"box_2d": [471, 353, 538, 510], "label": "fallen person"}\n]',
    ),
    (
        "qwen",
        "qwen",
        "xyxy_0_1000",
        (640, 640),
        ["fallen person"],
        '```json\n[\n\t{"bbox_2d": [425, 510, 603, 629], "label": "fallen person"},\n'
        '\t{"bbox_2d": [358, 469, 520, 543], "label": "fallen person"}\n]\n```',
    ),
    (
        "zai",
        "zai",
        "xyxy_0_1000",
        (640, 640),
        ["fallen person"],
        '```json\n[\n  {"box_2d": [356, 478, 517, 553], "label": "fallen person"},\n'
        '  {"box_2d": [428, 520, 592, 634], "label": "fallen person"}\n]\n```',
    ),
    (
        "muse",
        "muse",
        "named_0_1000",
        (640, 640),
        ["solar panel hot spot"],
        '[{"label":"solar panel hot spot","x_min":468,"y_min":661,"x_max":490,"y_max":682},'
        '{"label":"solar panel hot spot","x_min":503,"y_min":662,"x_max":523,"y_max":682}]',
    ),
    (
        "kimi",
        "openai",
        "named_normalized",
        (512, 512),
        ["shipping container number", "container ISO type code"],
        '```json\n{\n  "detections": [\n'
        '    {"x_min": 0.331, "y_min": 0.146, "x_max": 0.359, "y_max": 0.664, '
        '"class_name": "shipping container number", "confidence": 0.92},\n'
        '    {"x_min": 0.820, "y_min": 0.208, "x_max": 0.950, "y_max": 0.316, '
        '"class_name": "shipping container number", "confidence": 0.88},\n'
        '    {"x_min": 0.356, "y_min": 0.172, "x_max": 0.380, "y_max": 0.296, '
        '"class_name": "container ISO type code", "confidence": 0.85}\n  ]\n}\n```',
    ),
    (
        "spacexai",
        "spacexai",
        "xyxy_percent",
        (640, 640),
        ["fallen person"],
        '```json\n[{"label":"fallen person","box_2d":[32.9,54.5,54.5,62.2]},'
        '{"label":"fallen person","box_2d":[42.8,58.6,61.7,71.1]}]\n```',
    ),
    (
        "openrouter",
        "openai",
        "named_normalized",
        (1081, 748),
        ["not gate symbol", "nand gate symbol"],
        '```json\n{"detections":[{"x_min":0.1,"y_min":0.2,"x_max":0.2,"y_max":0.3,'
        '"class_name":"not gate symbol","confidence":0.85},'
        '{"x_min":0.4,"y_min":0.5,"x_max":0.5,"y_max":0.6,'
        '"class_name":"nand gate symbol","confidence":0.90}]}\n```',
    ),
]


def _build_image(width: int, height: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image"),
        numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
    )


def _upload_dimensions(
    family: str, image: WorkflowImageData, box_format: str
) -> Tuple[Optional[int], Optional[int]]:
    if family == CLAUDE:
        return claude_v5.detection_upload_dimensions(
            image=image, task_type="object-detection"
        )
    if family == OPENAI:
        return openai_v7.detection_upload_dimensions(
            image=image, task_type="object-detection", box_format=box_format
        )
    return None, None


@pytest.mark.parametrize(
    "family, model_type, box_format, size, classes, raw",
    SAMPLES,
    ids=[sample[0] for sample in SAMPLES],
)
def test_in_block_decoding_matches_legacy_formatter(
    family: str,
    model_type: str,
    box_format: str,
    size: Tuple[int, int],
    classes: List[str],
    raw: str,
) -> None:
    # given
    image = _build_image(*size)
    upload_width, upload_height = _upload_dimensions(family, image, box_format)

    # when
    legacy = VLMAsDetectorBlockV2().run(
        image=image,
        vlm_output=raw,
        classes=classes,
        model_type=model_type,
        task_type="object-detection",
    )
    error_status, predictions = decode_vlm_output(
        task_type="object-detection",
        raw_output=raw,
        image=image,
        classes=classes,
        inference_id="iid",
        box_format=box_format,
        upload_width=upload_width,
        upload_height=upload_height,
    )

    # then
    assert legacy["error_status"] is False
    assert error_status is False
    assert is_detection_prediction(predictions)
    assert detection_count(predictions) == len(legacy["predictions"]) >= 2
    assert np.allclose(
        np.array(detection_boxes(predictions)), legacy["predictions"].xyxy, atol=1.0
    )
    assert detection_class_ids(predictions) == legacy["predictions"].class_id.tolist()
