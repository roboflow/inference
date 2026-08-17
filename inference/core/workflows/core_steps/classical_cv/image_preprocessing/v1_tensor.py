"""Tensor-native sibling of ``image_preprocessing/v1``.

Per-task contract:

* ``flip``: tensor-native, bit-exact - a pure pixel permutation
  (``cv2.flip`` <-> ``torch.flip``), independent of channel order.

* ``rotate`` by right angles (+-90/+-180/+-270/+-360): tensor-native and
  bit-exact, but not a naive ``rot90``: v1 rotates about ``(w // 2, h // 2)``
  (off the true pixel centre for odd dims) onto an ``int()``-truncated
  canvas. In float64 the residual trig terms (cos 90deg ~ 6.1e-17) vanish in
  ``warpAffine``'s 1/1024 fixed-point coordinates, so the inverse map
  degenerates to axis-separable sampling at integer or half-integer
  coordinates:

  - even source axes sample at integers - a permutation, except one border
    index falls outside the source (BORDER_CONSTANT zeros);
  - odd source axes sample halfway between pixels - a 2-tap average with
    warpAffine's fixed-point rounding: ``(a + b + 1) >> 1`` for one half
    axis, a single fused ``(a + b + c + d + 2) >> 2`` when both are half
    (composing two rounded 1-D averages would round twice and diverge on
    ties).

  +-360deg is a genuine warp in v1 (only ``0``/``None`` early-returns a
  copy): identity for even dims, a half-pixel self-average for odd ones.

* ``rotate`` by arbitrary angles: delegates (full fixed-point bilinear warp).

* ``resize``: delegates for every real target - ``cv2.INTER_AREA`` rounding
  is kernel-size- and SIMD-path-dependent, so no torch pooling stays
  bit-exact across platforms. The ``width=height=None`` early-return copy is
  tensor-native (a device-side ``clone``).

Numpy/base64-born images always delegate (no materialised tensor).
"""

from typing import Callable, Optional, Tuple, Type

import torch

from inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1 import (
    ImagePreprocessingManifest,
    apply_flip_image,
    apply_resize_image,
    apply_rotate_image,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlock


class ImagePreprocessingBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[ImagePreprocessingManifest]:
        return ImagePreprocessingManifest

    def run(
        self,
        image: WorkflowImageData,
        task_type: str,
        width: Optional[int],
        height: Optional[int],
        rotation_degrees: Optional[int],
        flip_type: Optional[str],
        *args,
        **kwargs,
    ) -> BlockResult:
        # Validation mirrors v1 and precedes any tensor-vs-delegate decision,
        # so both siblings raise identically.
        if task_type == "resize":
            if width is not None and width <= 0:
                raise ValueError("Width must be greater than 0")
            if height is not None and height <= 0:
                raise ValueError("Height must be greater than 0")
            if image.is_tensor_materialised() and width is None and height is None:
                # v1's early-return copy - replicated as a device-side clone.
                return _emit_tensor_output(image=image, chw=image.tensor_image.clone())
            return _delegate_to_numpy(image, apply_resize_image, width, height)
        elif task_type == "rotate":
            if rotation_degrees is not None and not (-360 <= rotation_degrees <= 360):
                raise ValueError("Rotation degrees must be between -360 and 360")
            if image.is_tensor_materialised():
                if rotation_degrees is None or rotation_degrees == 0:
                    # v1's early-return copy - replicated as a device-side clone.
                    return _emit_tensor_output(
                        image=image, chw=image.tensor_image.clone()
                    )
                if rotation_degrees % 90 == 0:
                    rotated = _rotate_right_angle_tensor(
                        chw=image.tensor_image, rotation_degrees=rotation_degrees
                    )
                    return _emit_tensor_output(image=image, chw=rotated)
            return _delegate_to_numpy(image, apply_rotate_image, rotation_degrees)
        elif task_type == "flip":
            if flip_type is not None and flip_type not in [
                "vertical",
                "horizontal",
                "both",
            ]:
                raise ValueError(
                    "Flip type must be 'vertical', 'horizontal', or 'both'"
                )
            if image.is_tensor_materialised():
                return _emit_tensor_output(
                    image=image,
                    chw=_flip_tensor(chw=image.tensor_image, flip_type=flip_type),
                )
            return _delegate_to_numpy(image, apply_flip_image, flip_type)
        else:
            raise ValueError(f"Invalid task type: {task_type}")


def _delegate_to_numpy(
    image: WorkflowImageData,
    transformation: Callable,
    *transformation_args,
) -> BlockResult:
    response_image = transformation(image.numpy_image, *transformation_args)
    output_image = WorkflowImageData.copy_and_replace(
        origin_image_data=image, numpy_image=response_image
    )
    return {"image": output_image}


def _emit_tensor_output(image: WorkflowImageData, chw: torch.Tensor) -> BlockResult:
    output_image = WorkflowImageData.copy_and_replace(
        origin_image_data=image, tensor_image=chw
    )
    return {"image": output_image}


def _flip_tensor(chw: torch.Tensor, flip_type: Optional[str]) -> torch.Tensor:
    if flip_type == "vertical":
        return torch.flip(chw, dims=(-2,))
    if flip_type == "horizontal":
        return torch.flip(chw, dims=(-1,))
    if flip_type == "both":
        return torch.flip(chw, dims=(-2, -1))
    # v1's apply_flip_image else-branch: pass-through (only flip_type=None
    # reaches it via run()).
    return chw


# Per-axis sampling modes of the right-angle warp (see the module docstring):
# "exact" copies the axis, "shift" moves it by one index with a zero fill
# (the out-of-range border sample of even axes), "half" sums each element
# with its predecessor (the /2 with fixed-point rounding is applied once,
# fused, at the end).
_AXIS_EXACT = "exact"
_AXIS_SHIFT = "shift"
_AXIS_HALF = "half"


def _rotate_right_angle_tensor(
    chw: torch.Tensor, rotation_degrees: int
) -> torch.Tensor:
    """Bit-exact replica of v1's ``apply_rotate_image`` for multiples of 90deg.

    Derived dest->source maps (X = dest column, Y = dest row; ``cx = w // 2``,
    ``cy = h // 2``; half offsets appear exactly when the axis length is odd):

    * q=1 (90deg ccw):  ``y_src = X - 0.5*(h odd)``, ``x_src = w - Y - 0.5*(w odd)``
    * q=2 (180deg):     ``x_src = w - X - 0.5*(w odd)``, ``y_src = h - Y - 0.5*(h odd)``
    * q=3 (270deg):     ``y_src = h - X - 0.5*(h odd)``, ``x_src = Y - 0.5*(w odd)``
    * q=0 (+-360deg):   ``x_src = X - 0.5*(w odd)``, ``y_src = Y - 0.5*(h odd)``

    Each axis is therefore (optionally) reversed, then sampled exactly,
    shifted by one (out-of-range -> 0), or 2-tap averaged; 90/270 transpose
    H<->W at the end.
    """
    height, width = int(chw.shape[-2]), int(chw.shape[-1])
    quarter_turns = int(rotation_degrees // 90) % 4
    y_half = _AXIS_HALF if height % 2 else None
    x_half = _AXIS_HALF if width % 2 else None
    if quarter_turns == 0:
        y_spec = (False, y_half or _AXIS_EXACT)
        x_spec = (False, x_half or _AXIS_EXACT)
    elif quarter_turns == 1:
        y_spec = (False, y_half or _AXIS_EXACT)
        x_spec = (True, x_half or _AXIS_SHIFT)
    elif quarter_turns == 2:
        y_spec = (True, y_half or _AXIS_SHIFT)
        x_spec = (True, x_half or _AXIS_SHIFT)
    else:
        y_spec = (True, y_half or _AXIS_SHIFT)
        x_spec = (False, x_half or _AXIS_EXACT)
    accumulator = chw.to(torch.int32)
    accumulator, y_halved = _sample_axis(accumulator, dim=-2, spec=y_spec)
    accumulator, x_halved = _sample_axis(accumulator, dim=-1, spec=x_spec)
    halvings = y_halved + x_halved
    if halvings:
        # warpAffine's fused fixed-point rounding: (a+b+1)>>1 / (a+b+c+d+2)>>2.
        accumulator = (accumulator + (1 << (halvings - 1))) // (1 << halvings)
    if quarter_turns in (1, 3):
        accumulator = accumulator.transpose(-2, -1)
    return accumulator.to(torch.uint8).contiguous()


def _sample_axis(
    accumulator: torch.Tensor, dim: int, spec: Tuple[bool, str]
) -> Tuple[torch.Tensor, int]:
    reverse, mode = spec
    if reverse:
        accumulator = torch.flip(accumulator, dims=(dim,))
    if mode == _AXIS_EXACT:
        return accumulator, 0
    shifted = _shifted_by_one(accumulator, dim=dim)
    if mode == _AXIS_SHIFT:
        return shifted, 0
    return accumulator + shifted, 1


def _shifted_by_one(accumulator: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = torch.zeros_like(accumulator)
    if dim == -1:
        shifted[..., 1:] = accumulator[..., :-1]
    else:
        shifted[..., 1:, :] = accumulator[..., :-1, :]
    return shifted
