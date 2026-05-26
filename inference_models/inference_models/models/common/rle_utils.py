from typing import Optional

import numpy as np
import torch
from pycocotools import mask as mask_utils

from inference_models.models.base.types import InstancesRLEMasks


def torch_mask_to_coco_rle(mask: torch.Tensor) -> dict:
    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    mask_flat = mask.permute(1, 0).reshape(-1)
    values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
    counts = lengths.cpu().tolist()

    if values[0] == 1:
        counts.insert(0, 0)

    h, w = mask.shape
    return mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)


def numpy_mask_to_coco_rle(mask: np.ndarray) -> dict:
    mask_bool = np.asarray(mask, dtype=bool)
    mask_flat = np.ravel(mask_bool, order="F")
    if mask_flat.size == 0:
        h, w = mask_bool.shape
        return mask_utils.frPyObjects({"counts": [], "size": [h, w]}, h, w)
    transitions = np.flatnonzero(mask_flat[1:] != mask_flat[:-1]) + 1
    counts = np.diff(
        np.concatenate(
            (
                np.array([0], dtype=np.int64),
                transitions.astype(np.int64, copy=False),
                np.array([mask_flat.size], dtype=np.int64),
            )
        )
    ).tolist()
    if mask_flat[0]:
        counts.insert(0, 0)
    h, w = mask_bool.shape
    return mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)


def unpack_bitpacked_masks_numpy(bitpacked_masks: np.ndarray, width: int) -> np.ndarray:
    packed = np.asarray(bitpacked_masks, dtype=np.uint8)
    if packed.ndim != 3:
        raise ValueError(
            f"Expected bitpacked masks with shape (N, H, Wbytes), got {packed.shape}."
        )
    unpacked = np.unpackbits(np.ascontiguousarray(packed), axis=-1, bitorder="little")
    return np.ascontiguousarray(unpacked[..., :width])


class LazyInstancesRLEMasks(InstancesRLEMasks):
    """Materializes COCO RLE counts only when a caller actually needs them."""

    def __init__(
        self,
        image_size: tuple,
        mask_gpu: Optional[torch.Tensor] = None,
        mask_packed_gpu: Optional[torch.Tensor] = None,
        mask_packed_width: Optional[int] = None,
        mask_cpu: Optional[np.ndarray] = None,
        rle_counts_gpu: Optional[torch.Tensor] = None,
        rle_lengths_gpu: Optional[torch.Tensor] = None,
        rle_counts_cpu: Optional[np.ndarray] = None,
        rle_lengths_cpu: Optional[np.ndarray] = None,
        done_event: Optional["torch.cuda.Event"] = None,
    ):
        self.image_size = image_size
        self._masks: list = []
        self._materialized = False
        self._mask_gpu = mask_gpu
        self._mask_packed_gpu = mask_packed_gpu
        self._mask_packed_width = mask_packed_width
        self._mask_cpu = mask_cpu
        self._rle_counts_gpu = rle_counts_gpu
        self._rle_lengths_gpu = rle_lengths_gpu
        self._rle_counts_cpu = rle_counts_cpu
        self._rle_lengths_cpu = rle_lengths_cpu
        self._done_event = done_event

    @property
    def masks(self) -> list:
        self._ensure_materialized()
        return self._masks

    @masks.setter
    def masks(self, value: list) -> None:
        self._masks = value
        self._materialized = True

    def _ensure_mask_cpu(self) -> np.ndarray:
        if self._mask_cpu is not None:
            return self._mask_cpu
        if self._mask_packed_gpu is not None:
            device = self._mask_packed_gpu.device
            stream = torch.cuda.current_stream(device)
            if self._done_event is not None:
                self._done_event.wait(stream)
            packed_cpu = self._mask_packed_gpu.cpu().numpy()
            width = (
                self._mask_packed_width
                if self._mask_packed_width is not None
                else self.image_size[1]
            )
            self._mask_cpu = unpack_bitpacked_masks_numpy(packed_cpu, width=width).view(
                np.bool_
            )
            return self._mask_cpu
        if self._mask_gpu is None:
            self._mask_cpu = np.empty(
                (0, self.image_size[0], self.image_size[1]), dtype=bool
            )
            return self._mask_cpu
        device = self._mask_gpu.device
        stream = torch.cuda.current_stream(device)
        if self._done_event is not None:
            self._done_event.wait(stream)
        mask_cpu = self._mask_gpu.cpu().numpy()
        if mask_cpu.dtype == np.uint8:
            mask_cpu = mask_cpu.view(np.bool_)
        else:
            mask_cpu = mask_cpu.astype(bool, copy=False)
        self._mask_cpu = mask_cpu
        return self._mask_cpu

    def _ensure_rle_cpu(self) -> None:
        if self._rle_counts_cpu is not None and self._rle_lengths_cpu is not None:
            return
        if self._rle_counts_gpu is None or self._rle_lengths_gpu is None:
            return
        device = self._rle_lengths_gpu.device
        stream = torch.cuda.current_stream(device)
        if self._done_event is not None:
            self._done_event.wait(stream)
        lengths_cpu = self._rle_lengths_gpu.cpu().numpy().astype(np.int32, copy=False)
        if lengths_cpu.size == 0:
            counts_cpu = np.empty((0, 0), dtype=np.int32)
        else:
            max_len = int(lengths_cpu.max())
            counts_slice = self._rle_counts_gpu[:, :max_len]
            counts_cpu = counts_slice.cpu().numpy().astype(np.int32, copy=False)
        self._rle_lengths_cpu = lengths_cpu
        self._rle_counts_cpu = counts_cpu

    def _ensure_materialized(self) -> None:
        if self._materialized:
            return
        self._ensure_rle_cpu()
        if self._rle_counts_cpu is not None and self._rle_lengths_cpu is not None:
            h, w = self.image_size
            self._masks = [
                mask_utils.frPyObjects(
                    {
                        "counts": self._rle_counts_cpu[
                            i, : int(self._rle_lengths_cpu[i])
                        ]
                        .astype(np.int64, copy=False)
                        .tolist(),
                        "size": [h, w],
                    },
                    h,
                    w,
                )["counts"]
                for i in range(self._rle_lengths_cpu.shape[0])
            ]
        else:
            mask_cpu = self._ensure_mask_cpu()
            self._masks = [
                numpy_mask_to_coco_rle(mask=mask_cpu[i])["counts"]
                for i in range(mask_cpu.shape[0])
            ]
        self._materialized = True

    def to_coco_rle_masks(self) -> list:
        self._ensure_materialized()
        return super().to_coco_rle_masks()


def coco_rle_masks_to_numpy_mask(instances_masks: InstancesRLEMasks) -> np.ndarray:
    if len(instances_masks.masks) == 0:
        return np.empty(
            (0, instances_masks.image_size[0], instances_masks.image_size[1]),
            dtype=bool,
        )
    return np.ascontiguousarray(
        mask_utils.decode(instances_masks.to_coco_rle_masks())
        .transpose(2, 0, 1)
        .astype(bool)
    )


def coco_rle_masks_to_torch_mask(
    instances_masks: InstancesRLEMasks, device: Optional[torch.device] = None
) -> torch.Tensor:
    if len(instances_masks.masks) == 0:
        return torch.empty(
            size=(0, instances_masks.image_size[0], instances_masks.image_size[1]),
            dtype=torch.bool,
            device=device,
        )
    return torch.from_numpy(
        np.ascontiguousarray(
            mask_utils.decode(instances_masks.to_coco_rle_masks())
            .transpose(2, 0, 1)
            .astype(bool)
        )
    ).to(device=device, dtype=torch.bool)
