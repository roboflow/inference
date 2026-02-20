from typing import List, Optional, Tuple

import torch
from packaging.version import Version

from inference_models.errors import ModelPackageNegotiationError


def filter_available_devices_with_selected_device(
    selected_device: Optional[torch.device],
    all_available_cuda_devices: List[str],
    all_available_devices_cc: List[Version],
) -> Tuple[List[str], List[Version]]:
    if selected_device is not None and selected_device.type != "cuda":
        return [], []
    if selected_device is not None and selected_device.type == "cuda":
        index = selected_device.index or 0
        if index >= len(all_available_cuda_devices) or index >= len(
            all_available_devices_cc
        ):
            raise ModelPackageNegotiationError(
                message=f"Model Package Negotiation algorithm received selected device: {selected_device} which "
                f"does not match runtime introspection results. If you selected device to run the model "
                f"manually - verify your choice. Otherwise, this error most likely is a bug. Create new "
                f"issue: https://github.com/roboflow/inference/issues",
                help_url="https://inference-models.roboflow.com/errors/package-negotiation/#modelpackagenegotiationerror",
            )
        all_available_cuda_devices = [all_available_cuda_devices[index]]
        all_available_devices_cc = [all_available_devices_cc[index]]
        return all_available_cuda_devices, all_available_devices_cc
    return all_available_cuda_devices, all_available_devices_cc
