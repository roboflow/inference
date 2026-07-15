"""Regression tests: async-stream blocks must not mutate the caller's
``class_names`` list.

When these blocks run as async stream steps, a scalar (non-batch)
``class_names`` list is shared across frames. The blocks default an empty
list to ``[None]`` internally; they must do so on a fresh copy so the
caller's list is never mutated (which would corrupt subsequent frames).
"""

from unittest.mock import MagicMock

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


def test_sam3_v1_run_locally_does_not_mutate_caller_class_names():
    block = SegmentAnything3BlockV1(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    shared_class_names = []

    # With no images the empty-list -> [None] defaulting still runs before the
    # (skipped) per-image loop, which is exactly where the mutation would occur.
    block.run_locally(
        images=[],
        model_id="sam3/sam3_final",
        class_names=shared_class_names,
        threshold=0.5,
    )
    assert shared_class_names == []

    # A second call with the same list must behave identically.
    block.run_locally(
        images=[],
        model_id="sam3/sam3_final",
        class_names=shared_class_names,
        threshold=0.5,
    )
    assert shared_class_names == []


def test_sam3_v2_run_locally_does_not_mutate_caller_class_names():
    block = SegmentAnything3BlockV2(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    shared_class_names = []

    block.run_locally(
        images=[],
        model_id="sam3/sam3_final",
        class_names=shared_class_names,
        confidence=0.5,
    )
    assert shared_class_names == []

    block.run_locally(
        images=[],
        model_id="sam3/sam3_final",
        class_names=shared_class_names,
        confidence=0.5,
    )
    assert shared_class_names == []


def test_seg_preview_v1_run_via_request_does_not_mutate_caller_class_names():
    block = SegPreviewBlockV1(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    shared_class_names = []

    block.run_via_request(
        images=[],
        class_names=shared_class_names,
        threshold=0.5,
    )
    assert shared_class_names == []

    block.run_via_request(
        images=[],
        class_names=shared_class_names,
        threshold=0.5,
    )
    assert shared_class_names == []
