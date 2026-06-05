from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from profiling.memory.metadata import ProfileTier, coerce_profile_tier
from profiling.memory.worker_config import MemoryProfilingWorkerPayload


def test_from_payload_coerces_and_validates() -> None:
    config = MemoryProfilingWorkerPayload.from_payload(
        {
            "module_name": "inference_models.models.yolov8.yolov8_object_detection_onnx",
            "class_name": "YOLOv8ForObjectDetectionOnnx",
            "package_path": "/tmp/pkg",
            "batch_size": 1,
            "height": 640,
            "width": 640,
            "warmup_iterations": 2,
            "measured_iterations": 5,
            "metadata_context": {"profile_tier": "validation"},
        }
    )

    assert config.profiling_shape == (1, 640, 640)
    assert config.package_path == Path("/tmp/pkg")
    assert config.resolved_profile_tier == ProfileTier.VALIDATION


def test_coerce_profile_tier_rejects_missing_and_invalid() -> None:
    with pytest.raises(ValueError, match="profile_tier is required"):
        coerce_profile_tier(None)

    with pytest.raises(ValueError, match="Invalid profile_tier 'bogus'"):
        coerce_profile_tier("bogus")


def test_resolved_profile_tier_requires_valid_value() -> None:
    config = MemoryProfilingWorkerPayload.from_payload(
        {
            "module_name": "m",
            "class_name": "C",
            "package_path": "/tmp/pkg",
            "batch_size": 1,
            "height": 640,
            "width": 640,
            "warmup_iterations": 0,
            "measured_iterations": 1,
        }
    )

    with pytest.raises(ValueError, match="profile_tier is required"):
        _ = config.resolved_profile_tier


def test_from_payload_rejects_invalid_shape() -> None:
    with pytest.raises(ValidationError):
        MemoryProfilingWorkerPayload.from_payload(
            {
                "module_name": "m",
                "class_name": "C",
                "package_path": "/tmp/pkg",
                "batch_size": 0,
                "height": 640,
                "width": 640,
                "warmup_iterations": 0,
                "measured_iterations": 1,
            }
        )
