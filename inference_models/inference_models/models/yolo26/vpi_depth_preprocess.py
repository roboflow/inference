"""VPI CUDA resize interoperability for YOLO26 depth preprocessing."""

from __future__ import annotations

import importlib
import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from inference_models.errors import MissingDependencyError, ModelRuntimeError

_VPI_SLOT_COUNT = 2
_MINIMUM_VPI_VERSION = (3, 2)
_MAXIMUM_VPI_MAJOR_VERSION = 3


def _raise_vpi_incompatible(*reasons: str) -> None:
    raise ModelRuntimeError(
        message=(
            "Input is incompatible with the explicit YOLO26 VPI preprocessor: "
            + "; ".join(reasons)
        ),
        help_url=(
            "https://inference-models.roboflow.com/errors/"
            "models-runtime/#modelruntimeerror"
        ),
    )


def _load_vpi_module() -> Any:
    try:
        return importlib.import_module("vpi")
    except Exception as import_error:
        raise MissingDependencyError(
            message=(
                "The explicit YOLO26 VPI preprocessor requires NVIDIA VPI "
                "Python 3.2.x and libnvvpi.so.3 in the runtime container."
            ),
            help_url=(
                "https://inference-models.roboflow.com/errors/"
                "runtime-environment/#missingdependencyerror"
            ),
        ) from import_error


def _parse_vpi_version(vpi_module: Any) -> tuple[int, int]:
    version = str(getattr(vpi_module, "__version__", ""))
    try:
        major, minor = version.split(".", maxsplit=2)[:2]
        return int(major), int(minor)
    except (TypeError, ValueError):
        _raise_vpi_incompatible(
            f"could not parse the installed VPI version {version!r}"
        )


@dataclass
class _VPICUDAImageSlot:
    image: Any
    stream: Any
    reuse_event: torch.cuda.Event
    cuda_lock: Optional[Any] = None
    conversion_pending: bool = False


class _VPICUDAImageSlotPool:
    """Bounded VPI output storage with CUDA-consumer reuse ordering."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        vpi_module: Any,
        slot_count: int = _VPI_SLOT_COUNT,
    ) -> None:
        self.height = height
        self.width = width
        self._slots = queue.LifoQueue(maxsize=slot_count)
        for _ in range(slot_count):
            self._slots.put(
                _VPICUDAImageSlot(
                    image=vpi_module.Image(
                        (width, height),
                        vpi_module.Format.BGR8,
                    ),
                    stream=vpi_module.Stream(),
                    reuse_event=torch.cuda.Event(),
                )
            )

    def acquire(self) -> _VPICUDAImageSlot:
        slot = self._slots.get()
        if slot.conversion_pending:
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.vpi-slot-reuse-wait"):
                slot.reuse_event.synchronize()
            slot.conversion_pending = False
            self._release_cuda_lock(slot)
        elif slot.cuda_lock is not None:
            self._release_cuda_lock(slot)

        return slot

    def release(self, slot: _VPICUDAImageSlot) -> None:
        self._slots.put(slot)

    @staticmethod
    def _release_cuda_lock(slot: _VPICUDAImageSlot) -> None:
        if slot.cuda_lock is None:
            return
        slot.cuda_lock.__exit__(None, None, None)
        slot.cuda_lock = None


class BorrowedVPICUDAImage:
    """Keep a VPI CUDA lock alive until its Torch consumer is complete."""

    def __init__(
        self,
        *,
        tensor: torch.Tensor,
        slot: _VPICUDAImageSlot,
        pool: _VPICUDAImageSlotPool,
    ) -> None:
        self.tensor = tensor
        self._slot = slot
        self._pool = pool
        self._returned = False

    def mark_consumed_and_release(self, *, stream: torch.cuda.Stream) -> None:
        """Order slot reuse after all prior work on the consumer stream."""
        if self._returned:
            raise RuntimeError("VPI CUDA image borrow was already returned")
        try:
            self._slot.reuse_event.record(stream)
            self._slot.conversion_pending = True
        except Exception:
            stream.synchronize()
            self._pool._release_cuda_lock(self._slot)
            raise
        finally:
            self._pool.release(self._slot)
            self._returned = True

    def abort_and_release(self, *, stream: torch.cuda.Stream) -> None:
        """Synchronize possible consumer work before releasing a failed borrow."""
        if self._returned:
            return
        try:
            stream.synchronize()
            self._pool._release_cuda_lock(self._slot)
        finally:
            self._pool.release(self._slot)
            self._returned = True


class VPICUDALetterboxResizer:
    """Resize host uint8 HWC images into reusable VPI CUDA storage."""

    def __init__(
        self,
        *,
        device: torch.device,
        vpi_module: Optional[Any] = None,
    ) -> None:
        if device.type != "cuda":
            _raise_vpi_incompatible(f"requires a CUDA device, received {device}")
        self._device = device
        self._vpi = vpi_module if vpi_module is not None else _load_vpi_module()
        installed_version = _parse_vpi_version(self._vpi)
        if (
            installed_version < _MINIMUM_VPI_VERSION
            or installed_version[0] > _MAXIMUM_VPI_MAJOR_VERSION
        ):
            _raise_vpi_incompatible(
                "requires NVIDIA VPI >=3.2,<4, received "
                f"{getattr(self._vpi, '__version__', None)!r}"
            )
        self._pool_lock = threading.Lock()
        self._pool: Optional[_VPICUDAImageSlotPool] = None

    def resize(
        self,
        *,
        image: np.ndarray,
        output_height: int,
        output_width: int,
        target_device: torch.device,
    ) -> BorrowedVPICUDAImage:
        """Submit VPI CUDA resize and expose its output as a zero-copy tensor."""
        self._validate_request(
            image=image,
            output_height=output_height,
            output_width=output_width,
            target_device=target_device,
        )
        pool = self._get_pool(height=output_height, width=output_width)
        slot = pool.acquire()
        lock_entered = False
        try:
            source = self._vpi.asimage(image, self._vpi.Format.BGR8)
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.vpi-resize-submit"):
                source.rescale(
                    slot.image,
                    interp=self._vpi.Interp.LINEAR,
                    backend=self._vpi.Backend.CUDA,
                    stream=slot.stream,
                )
            with torch.cuda.nvtx.range(
                "yolo26-depth.preprocess.vpi-resize-synchronize"
            ):
                slot.stream.sync()
            with torch.cuda.nvtx.range("yolo26-depth.preprocess.vpi-output-lock"):
                slot.cuda_lock = slot.image.rlock_cuda()
                cuda_buffer = slot.cuda_lock.__enter__()
                lock_entered = True
                tensor = torch.as_tensor(cuda_buffer, device=target_device)
            self._validate_zero_copy_output(
                tensor=tensor,
                cuda_buffer=cuda_buffer,
                output_height=output_height,
                output_width=output_width,
            )
        except Exception:
            if lock_entered:
                pool._release_cuda_lock(slot)
            pool.release(slot)
            raise

        return BorrowedVPICUDAImage(tensor=tensor, slot=slot, pool=pool)

    def _get_pool(
        self,
        *,
        height: int,
        width: int,
    ) -> _VPICUDAImageSlotPool:
        with self._pool_lock:
            if self._pool is None:
                self._pool = _VPICUDAImageSlotPool(
                    height=height,
                    width=width,
                    vpi_module=self._vpi,
                )
            elif self._pool.height != height or self._pool.width != width:
                _raise_vpi_incompatible(
                    "VPI output shape is fixed per model instance; initialized "
                    f"{(self._pool.height, self._pool.width)} but received "
                    f"{(height, width)}"
                )

            return self._pool

    def _validate_request(
        self,
        *,
        image: np.ndarray,
        output_height: int,
        output_width: int,
        target_device: torch.device,
    ) -> None:
        reasons = []
        if image.dtype != np.uint8:
            reasons.append(f"image dtype must be uint8, received {image.dtype}")
        if image.ndim != 3 or image.shape[-1] != 3:
            reasons.append(
                "image must have HWC shape with three channels, received "
                f"{image.shape}"
            )
        if image.ndim == 3 and image.shape[-1] == 3 and not image.flags.c_contiguous:
            reasons.append("image must be contiguous HWC; implicit copies are disabled")
        if output_height <= 0 or output_width <= 0:
            reasons.append(
                "output dimensions must be positive, received "
                f"{(output_height, output_width)}"
            )
        if target_device.type != self._device.type or (
            self._device.index is not None and target_device.index != self._device.index
        ):
            reasons.append(
                f"target device must be {self._device}, received {target_device}"
            )
        if reasons:
            _raise_vpi_incompatible(*reasons)

    @staticmethod
    def _validate_zero_copy_output(
        *,
        tensor: torch.Tensor,
        cuda_buffer: Any,
        output_height: int,
        output_width: int,
    ) -> None:
        reasons = []
        if tensor.device.type != "cuda":
            reasons.append(f"VPI output must remain on CUDA, received {tensor.device}")
        if tensor.dtype != torch.uint8:
            reasons.append(f"VPI output dtype must be uint8, received {tensor.dtype}")
        if tuple(tensor.shape) != (output_height, output_width, 3):
            reasons.append(
                "VPI output shape must be "
                f"{(output_height, output_width, 3)}, received {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            reasons.append(
                "VPI output must be contiguous HWC; implicit copies are disabled"
            )
        cuda_interface = getattr(cuda_buffer, "__cuda_array_interface__", None)
        if not isinstance(cuda_interface, dict) or "data" not in cuda_interface:
            reasons.append("VPI output does not expose the CUDA Array Interface")
        elif tensor.data_ptr() != cuda_interface["data"][0]:
            reasons.append("VPI-to-Torch conversion unexpectedly copied the output")
        if reasons:
            _raise_vpi_incompatible(*reasons)
