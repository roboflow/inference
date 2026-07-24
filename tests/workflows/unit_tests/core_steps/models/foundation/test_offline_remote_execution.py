from typing import Any, Dict, Type
from unittest.mock import MagicMock

import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.seg_preview.v1 import (
    SegPreviewBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v1 import (
    SegmentAnything3BlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v2 import (
    SegmentAnything3BlockV2,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    SegmentAnything3BlockV3,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
    SegmentAnything3_3D_ObjectsBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    SegmentAnything3InteractiveBlockV1,
)
from inference.core.workflows import offline


REMOTE_LEAF_CASES = [
    (
        "seg-preview-proxy",
        SegPreviewBlockV1,
        "run_via_request",
        {"class_names": ["cat"], "threshold": 0.5},
    ),
    (
        "sam3-v1-sdk",
        SegmentAnything3BlockV1,
        "run_remotely",
        {"model_id": "sam3/test", "class_names": ["cat"], "threshold": 0.5},
    ),
    (
        "sam3-v1-proxy",
        SegmentAnything3BlockV1,
        "run_via_request",
        {"class_names": ["cat"], "threshold": 0.5},
    ),
    (
        "sam3-v2-sdk",
        SegmentAnything3BlockV2,
        "run_remotely",
        {"model_id": "sam3/test", "class_names": ["cat"], "confidence": 0.5},
    ),
    (
        "sam3-v2-proxy",
        SegmentAnything3BlockV2,
        "run_via_request",
        {"class_names": ["cat"], "confidence": 0.5},
    ),
    (
        "sam3-v3-sdk",
        SegmentAnything3BlockV3,
        "run_remotely",
        {"model_id": "sam3/test", "class_names": ["cat"], "confidence": 0.5},
    ),
    (
        "sam3-v3-proxy",
        SegmentAnything3BlockV3,
        "run_via_request",
        {"class_names": ["cat"], "confidence": 0.5},
    ),
    (
        "sam3-interactive-sdk",
        SegmentAnything3InteractiveBlockV1,
        "run_remotely",
        {
            "points": [{"x": 1, "y": 1, "positive": True}],
            "boxes": None,
            "threshold": 0.5,
            "multimask_output": True,
        },
    ),
    (
        "sam3-interactive-proxy",
        SegmentAnything3InteractiveBlockV1,
        "run_via_request",
        {
            "points": [{"x": 1, "y": 1, "positive": True}],
            "boxes": None,
            "threshold": 0.5,
            "multimask_output": True,
        },
    ),
    (
        "sam3-3d-sdk",
        SegmentAnything3_3D_ObjectsBlockV1,
        "run_remotely",
        {"mask_input": []},
    ),
]


@pytest.mark.parametrize(
    ("block_type", "method_name", "kwargs"),
    [(case[1], case[2], case[3]) for case in REMOTE_LEAF_CASES],
    ids=[case[0] for case in REMOTE_LEAF_CASES],
)
def test_builtin_remote_inference_leaves_fail_closed_offline(
    monkeypatch: pytest.MonkeyPatch,
    block_type: Type[Any],
    method_name: str,
    kwargs: Dict[str, Any],
) -> None:
    monkeypatch.setattr(offline, "OFFLINE_MODE", True)
    block = block_type(
        model_manager=MagicMock(),
        api_key="test-api-key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    with pytest.raises(RuntimeError, match="OFFLINE_MODE"):
        getattr(block, method_name)(images=[], **kwargs)
