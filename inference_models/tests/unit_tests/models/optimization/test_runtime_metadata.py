from types import SimpleNamespace

import numpy as np
import pytest
import torch

from inference_models.models.optimization.runtime_metadata import (
    OPTIMIZATION_RUNTIME_METADATA_SCHEMA_VERSION,
    SelectionSnapshot,
    resolve_stage_device,
)


def test_schema_version_advertises_optional_device_field() -> None:
    assert OPTIMIZATION_RUNTIME_METADATA_SCHEMA_VERSION == "1.1"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("not a tensor", None),
        (torch.zeros(2), "cpu"),
        (np.zeros((2, 2), dtype=np.uint8), "cpu"),
        ((torch.zeros(1), torch.zeros(1)), "cpu"),
        ([], None),
        (["detections"], None),
        ([SimpleNamespace(xyxy=torch.zeros((0, 4)))], "cpu"),
        (SimpleNamespace(device=torch.device("cuda:1")), "cuda:1"),
        (SimpleNamespace(device="meta"), "meta"),
    ],
)
def test_resolve_stage_device_reads_without_touching_data(value, expected) -> None:
    assert resolve_stage_device(value) == expected


def test_resolve_stage_device_never_recurses_without_bound() -> None:
    self_referencing = SimpleNamespace()
    self_referencing.xyxy = self_referencing
    first, second = SimpleNamespace(), SimpleNamespace()
    first.xyxy, second.xyxy = second, first

    assert resolve_stage_device(self_referencing) is None
    assert resolve_stage_device([first]) is None
    assert resolve_stage_device([[[torch.zeros(1)]]]) == "cpu"


def test_selection_snapshot_serialization_is_unchanged_by_device_support() -> None:
    snapshot = SelectionSnapshot(requested_id="auto", effective_id="base")

    assert snapshot.to_dict() == {
        "requested_id": "auto",
        "effective_id": "base",
        "fallback_occurred": False,
    }
    assert len(snapshot) == 4
