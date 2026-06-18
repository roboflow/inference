"""Helpers for tensor-input parity tests.

Under ENABLE_TENSOR_DATA_REPRESENTATION the `_tensor_native` integration tests still feed
the workflow a *numpy* fixture, so the model blocks hit their numpy fallback. To exercise
the OTHER branch (`WorkflowImageData.is_tensor_materialised()` -> True, blocks use the
on-device tensor + "rgb"), feed the image already materialised as a tensor.

`numpy_image_as_tensor` converts a BGR HWC uint8 fixture into exactly the CHW RGB uint8
tensor that `WorkflowImageData.tensor_image` builds, on WORKFLOWS_IMAGE_TENSOR_DEVICE. The
image deserializer wraps a raw torch.Tensor as `WorkflowImageData(tensor_image=...)`, so
passing the result as the `image` runtime parameter makes the input arrive pre-materialised.
"""
import numpy as np
import torch

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE


def numpy_image_as_tensor(bgr_hwc_image: np.ndarray) -> torch.Tensor:
    """BGR HWC uint8 numpy fixture -> CHW RGB uint8 tensor on WORKFLOWS_IMAGE_TENSOR_DEVICE."""
    rgb = bgr_hwc_image[:, :, ::-1].copy()
    return (
        torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
    )
