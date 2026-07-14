#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
from pathlib import Path
from unittest.mock import Mock, patch

import superiorvision
import supervision as sv
import torch
import trackers
import tracktors


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def assert_distribution_contract() -> dict[str, str]:
    expected_versions = {
        "superiorvision": "0.30.0.dev2",
        "tracktors": "2.6.0.dev1",
    }
    versions = {name: metadata.version(name) for name in expected_versions}
    assert (
        versions == expected_versions
    ), f"unexpected package versions: {versions}; expected {expected_versions}"

    package_providers = metadata.packages_distributions()
    expected_providers = {
        "superiorvision": {"superiorvision"},
        "supervision": {"superiorvision"},
        "tracktors": {"tracktors"},
        "trackers": {"tracktors"},
    }
    for package, expected in expected_providers.items():
        actual = {
            canonical_name(provider) for provider in package_providers.get(package, [])
        }
        assert actual == expected, (
            f"{package!r} providers are {sorted(actual)}, "
            f"expected {sorted(expected)}"
        )

    assert superiorvision.Detections is sv.Detections
    assert tracktors.ByteTrackTracker is trackers.ByteTrackTracker
    return versions


def assert_detection_is_cuda(detections: sv.Detections, device: torch.device) -> None:
    for field in ("xyxy", "mask", "confidence", "class_id", "tracker_id"):
        value = getattr(detections, field)
        if value is not None:
            assert isinstance(value, torch.Tensor), f"{field} is {type(value)}"
            assert value.device == device, f"{field} moved to {value.device}"

    for key, value in detections.data.items():
        if isinstance(value, torch.Tensor):
            assert value.device == device, f"data[{key!r}] moved to {value.device}"


def tracker_frame(device: torch.device, dx: float) -> sv.Detections:
    return sv.Detections(
        xyxy=torch.tensor(
            [
                [10.0 + dx, 10.0, 30.0 + dx, 30.0],
                [100.0 + dx, 100.0, 140.0 + dx, 140.0],
            ],
            dtype=torch.float32,
            device=device,
        ),
        confidence=torch.tensor([0.95, 0.90], device=device),
        class_id=torch.tensor([3, 7], dtype=torch.long, device=device),
        data={"marker": torch.tensor([101, 202], dtype=torch.long, device=device)},
    )


def mapped_library_paths(maps: str, soname: str) -> list[str]:
    paths: set[str] = set()
    for line in maps.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and soname in fields[5]:
            paths.add(fields[5])
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-sm", required=True, choices=("87", "110"))
    args = parser.parse_args()

    versions = assert_distribution_contract()

    assert torch.cuda.is_available(), "CUDA is unavailable"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    expected_capability = {
        "87": (8, 7),
        "110": (11, 0),
    }[args.expected_sm]
    actual_capability = torch.cuda.get_device_capability(device)
    assert (
        actual_capability == expected_capability
    ), f"device capability is {actual_capability}, expected {expected_capability}"

    build_arches = set(torch.cuda.get_arch_list())
    required_build_arches = {"sm_87", "sm_110"}
    assert required_build_arches <= build_arches, (
        f"Torch arch list {sorted(build_arches)} lacks "
        f"{sorted(required_build_arches - build_arches)}"
    )

    detections = sv.Detections(
        xyxy=torch.tensor(
            [[0, 0, 10, 10], [0, 0, 4, 4], [20, 20, 30, 30]],
            dtype=torch.float32,
            device=device,
        ),
        confidence=torch.tensor([0.90, 0.80, 0.70], device=device),
        class_id=torch.tensor([1, 1, 1], dtype=torch.long, device=device),
        data={
            "embedding": torch.arange(12, dtype=torch.float32, device=device).reshape(
                3, 4
            )
        },
    )

    cpu_mock = Mock(side_effect=AssertionError("unexpected Tensor.cpu() call"))
    numpy_mock = Mock(side_effect=AssertionError("unexpected Tensor.numpy() call"))

    with patch.object(torch.Tensor, "cpu", cpu_mock), patch.object(
        torch.Tensor, "numpy", numpy_mock
    ):
        selected = detections[
            torch.tensor([True, False, True], dtype=torch.bool, device=device)
        ]
        iou_kept = detections.with_nms(
            threshold=0.5,
            overlap_metric=sv.OverlapMetric.IOU,
        )
        ios_kept = detections.with_nms(
            threshold=0.5,
            overlap_metric=sv.OverlapMetric.IOS,
        )

        assert len(selected) == 2
        assert len(iou_kept) == 3
        assert len(ios_kept) == 2

        for result in (selected, iou_kept, ios_kept):
            assert_detection_is_cuda(result, device)

        tracker = trackers.ByteTrackTracker(
            minimum_consecutive_frames=1,
            track_activation_threshold=0.7,
            high_conf_det_threshold=0.6,
            minimum_iou_threshold=0.1,
        )
        first_result = tracker.update(tracker_frame(device, dx=0.0))
        second_result = tracker.update(tracker_frame(device, dx=1.0))

        assert_detection_is_cuda(first_result, device)
        assert_detection_is_cuda(second_result, device)

    cpu_mock.assert_not_called()
    numpy_mock.assert_not_called()

    assert (
        len(tracker.tracks) == 2
    ), f"expected 2 live tracks, got {len(tracker.tracks)}"
    assert second_result.tracker_id is not None
    assert bool(
        torch.all(second_result.tracker_id >= 0).item()
    ), f"second update did not confirm both tracks: {second_result.tracker_id}"

    for track in tracker.tracks:
        assert track.number_of_successful_consecutive_updates == 2
        assert track.tracker_id >= 0
        assert track.get_state_bbox().device == device
        assert track.state_estimator.kf.state.device == device
        assert track.state_estimator.kf.state_covariance.device == device

    gemm_input = torch.ones((256, 256), dtype=torch.float32, device=device)
    gemm_output = gemm_input @ gemm_input
    assert gemm_output[0, 0].item() == 256.0

    size = 32
    coefficient = 4.0 * torch.eye(
        size, dtype=torch.float32, device=device
    ) + torch.full((size, size), 0.01, dtype=torch.float32, device=device)
    right_hand_side = torch.arange(size, dtype=torch.float32, device=device).reshape(
        size, 1
    )
    solution = torch.linalg.solve(coefficient, right_hand_side)

    torch.testing.assert_close(
        coefficient @ solution,
        right_hand_side,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.cuda.synchronize(device)

    maps = Path("/proc/self/maps").read_text()
    cublas_paths = mapped_library_paths(maps, "libcublas.so")
    cusolver_paths = mapped_library_paths(maps, "libcusolver.so")

    assert cublas_paths, "libcublas.so is not mapped after CUDA GEMM"
    assert cusolver_paths, "libcusolver.so is not mapped after torch.linalg.solve"

    print(
        json.dumps(
            {
                "status": "ok",
                "device": torch.cuda.get_device_name(device),
                "capability": f"{actual_capability[0]}.{actual_capability[1]}",
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_arches": sorted(build_arches),
                "packages": versions,
                "nms_counts": {
                    "iou": len(iou_kept),
                    "ios": len(ios_kept),
                },
                "live_tracks": len(tracker.tracks),
                "libcublas": cublas_paths,
                "libcusolver": cusolver_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
