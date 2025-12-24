import pytest
import torch
from packaging.version import Version

from inference_models.errors import ModelPackageNegotiationError
from inference_models.models.auto_loaders.utils import (
    filter_available_devices_with_selected_device,
)


def test_filter_available_devices_with_selected_device_when_no_selected_device() -> (
    None
):
    # when
    result = filter_available_devices_with_selected_device(
        selected_device=None,
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
    )

    # then
    assert result == (["tesla-t4", "nvidia-l4"], [Version("7.5"), Version("8.7")])


def test_filter_available_devices_with_selected_device_when_cpu_device_selected() -> (
    None
):
    # when
    result = filter_available_devices_with_selected_device(
        selected_device=torch.device(type="cpu"),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
    )

    # then
    assert result == ([], [])


def test_filter_available_devices_with_selected_device_when_device_selected() -> None:
    # when
    result = filter_available_devices_with_selected_device(
        selected_device=torch.device(type="cuda", index=1),
        all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
        all_available_devices_cc=[Version("7.5"), Version("8.7")],
    )

    # then
    assert result == (["nvidia-l4"], [Version("8.7")])


def test_filter_available_devices_with_selected_device_when_invalid_device_selected() -> (
    None
):
    # when
    with pytest.raises(ModelPackageNegotiationError):
        _ = filter_available_devices_with_selected_device(
            selected_device=torch.device(type="cuda", index=2),
            all_available_cuda_devices=["tesla-t4", "nvidia-l4"],
            all_available_devices_cc=[Version("7.5"), Version("8.7")],
        )
