"""Triton kernels behind ``segmented_nms``: IoU bitmask and greedy keep, every image.

Imported only when the segmented stage initializes its kernels: the base stage,
selection and CPU tests need no Triton. Two launches per call, both on the
current torch stream, no host wait::

    _iou_mask_kernel      grid (B, W, W), W = ceil(max_count / 64)
        program (b, r, c): sorted rows r*64.., sorted columns c*64.. of image b
        bit j of mask[b, i, c] = IoU(box_i, box_j) > threshold, for j > i
        skips c < r (never read), as nms_kernel_impl skips row_start > col_start
    _greedy_keep_kernel   grid (B,), one program per image
        removed = 0
        for i in sorted order:
            if bit i of removed is 0: keep i; removed |= mask[b, i]
        stops after cap kept boxes: the first cap kept boxes never depend on
        later ones, so this is torchvision's keep list cut to max_detections

IoU follows torchvision 0.28 ``devIoU`` (row box first, as there)::

    w = fmaxf(fminf(a2, b2) - fmaxf(a0, b0), 0)    h likewise
    inter = w * h; Sa = (a2 - a0) * (a3 - a1); Sb = (b2 - b0) * (b3 - b1)
    inter / (Sa + Sb - inter) > float(threshold)

``tl.maximum``/``tl.minimum`` lower to maxnum/minnum (a NaN operand yields the
other, as CUDA ``fmaxf``); division is IEEE ``tl.div_rn``. The tagged source
does not fix whether nvcc contracted a multiply-add into an FMA, so the kernel
offers six explicit variants (``ARITHMETIC``), launched with
``enable_fp_fusion=False`` so that only the written FMAs happen. ``calibration.calibrate``
checks every variant against the installed ``torchvision.ops.nms`` on sampled
near-threshold pairs. A unique match is empirical evidence, not a proof that
the bits agree on every input.

Concurrency: scratch is allocated per call on the current stream; the only
process-wide state is ``_LAUNCH_LOCK``, held while a launch call runs, so a
first-call JIT compilation is never raced by a second thread.
"""

import hashlib
import struct
import threading
from pathlib import Path
from typing import Any, Dict

import torch
import torchvision
import triton
import triton.language as tl

BLOCK = 64
# Name -> (SUM_MODE, DENOMINATOR_MODE):
#   SUM_MODE 0: Sa + Sb      1: fma(a_w, a_h, Sb)      2: fma(b_w, b_h, Sa)
#   DENOMINATOR_MODE 0: sum - inter                    1: fma(-w, h, sum)
ARITHMETIC = {
    "add/sub": (0, 0),
    "fma_a/sub": (1, 0),
    "fma_b/sub": (2, 0),
    "add/fma": (0, 1),
    "fma_a/fma": (1, 1),
    "fma_b/fma": (2, 1),
}
IOU_LAUNCH = {"num_warps": 4, "enable_fp_fusion": False}
GREEDY_LAUNCH = {"num_warps": 1}
# One warp's width: no 1-element arange, and few compiled variants (32..256 words).
MIN_GREEDY_WORDS = 32
_LAUNCH_LOCK = threading.Lock()
KERNEL_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@triton.jit(do_not_specialize=["words_per_row", "max_count"])
def _iou_mask_kernel(
    boxes_ptr,
    order_ptr,
    starts_ptr,
    counts_ptr,
    mask_ptr,
    words_per_row,
    max_count,
    threshold,
    SUM_MODE: tl.constexpr,
    DENOMINATOR_MODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    image = tl.program_id(0).to(tl.int64)
    row_word = tl.program_id(1)
    col_word = tl.program_id(2)
    count = tl.load(counts_ptr + image)
    if (col_word >= row_word) & (col_word * BLOCK < count):
        start = tl.load(starts_ptr + image)
        lanes = tl.arange(0, BLOCK)
        rows = row_word * BLOCK + lanes
        cols = col_word * BLOCK + lanes
        row_ok = rows < count
        col_ok = cols < count
        row_box = start + tl.load(order_ptr + start + rows, mask=row_ok, other=0)
        col_box = start + tl.load(order_ptr + start + cols, mask=col_ok, other=0)

        a0 = tl.load(boxes_ptr + row_box * 4 + 0, mask=row_ok, other=0.0)[:, None]
        a1 = tl.load(boxes_ptr + row_box * 4 + 1, mask=row_ok, other=0.0)[:, None]
        a2 = tl.load(boxes_ptr + row_box * 4 + 2, mask=row_ok, other=0.0)[:, None]
        a3 = tl.load(boxes_ptr + row_box * 4 + 3, mask=row_ok, other=0.0)[:, None]
        b0 = tl.load(boxes_ptr + col_box * 4 + 0, mask=col_ok, other=0.0)[None, :]
        b1 = tl.load(boxes_ptr + col_box * 4 + 1, mask=col_ok, other=0.0)[None, :]
        b2 = tl.load(boxes_ptr + col_box * 4 + 2, mask=col_ok, other=0.0)[None, :]
        b3 = tl.load(boxes_ptr + col_box * 4 + 3, mask=col_ok, other=0.0)[None, :]

        width = tl.maximum(tl.minimum(a2, b2) - tl.maximum(a0, b0), 0.0)
        height = tl.maximum(tl.minimum(a3, b3) - tl.maximum(a1, b1), 0.0)
        inter = width * height
        a_w = tl.broadcast_to(a2 - a0, (BLOCK, BLOCK))
        a_h = tl.broadcast_to(a3 - a1, (BLOCK, BLOCK))
        b_w = tl.broadcast_to(b2 - b0, (BLOCK, BLOCK))
        b_h = tl.broadcast_to(b3 - b1, (BLOCK, BLOCK))
        if SUM_MODE == 0:
            area_sum = a_w * a_h + b_w * b_h
        elif SUM_MODE == 1:
            area_sum = tl.fma(a_w, a_h, b_w * b_h)
        else:
            area_sum = tl.fma(b_w, b_h, a_w * a_h)
        if DENOMINATOR_MODE == 0:
            denominator = area_sum - inter
        else:
            denominator = tl.fma(-width, height, area_sum)
        iou = tl.div_rn(inter, denominator)

        suppress = (iou > threshold) & (cols[None, :] > rows[:, None]) & col_ok[None, :]
        bit = tl.full((1, BLOCK), 1, tl.int64) << lanes[None, :].to(tl.int64)
        # Distinct powers of two: their sum is their OR (bit 63 wraps as intended).
        word = tl.sum(tl.where(suppress, bit, 0), axis=1)
        row_offsets = (image * max_count + rows) * words_per_row + col_word
        tl.store(mask_ptr + row_offsets, word, mask=row_ok)


@triton.jit(do_not_specialize=["words_per_row", "max_count", "cap"])
def _greedy_keep_kernel(
    mask_ptr,
    order_ptr,
    starts_ptr,
    counts_ptr,
    rows_ptr,
    kept_ptr,
    words_per_row,
    max_count,
    cap,
    WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    image = tl.program_id(0).to(tl.int64)
    count = tl.load(counts_ptr + image)
    start = tl.load(starts_ptr + image)
    word_ids = tl.arange(0, WORDS)
    row_words = (count + BLOCK - 1) // BLOCK
    removed = tl.zeros((WORDS,), dtype=tl.int64)
    kept = count * 0
    word = count * 0
    while (word < row_words) & (kept < cap):
        current = tl.sum(tl.where(word_ids == word, removed, 0))
        bit = count * 0
        while (bit < BLOCK) & (word * BLOCK + bit < count) & (kept < cap):
            if ((current >> bit) & 1) == 0:
                position = word * BLOCK + bit
                local_row = tl.load(order_ptr + start + position)
                tl.store(rows_ptr + image * cap + kept, start + local_row)
                kept += 1
                # Words before ``word`` were never written for this row (and
                # hold only earlier columns); torchvision skips them likewise.
                suppressed = tl.load(
                    mask_ptr
                    + (image * max_count + position) * words_per_row
                    + word_ids,
                    mask=(word_ids >= word) & (word_ids < row_words),
                    other=0,
                )
                removed = removed | suppressed
                current = tl.sum(tl.where(word_ids == word, removed, 0))
            bit += 1
        word += 1
    tl.store(kept_ptr + image, kept)


class TritonNmsKernels:
    """``segmented_nms.NmsKernels`` with the Triton kernels of this module.

    Args:
        arithmetic: One of ``ARITHMETIC``; ``build_kernels`` picks it by
            calibration, which constructs every variant.

    Raises:
        ValueError: On an unknown arithmetic name.
    """

    def __init__(self, *, arithmetic: str):
        if arithmetic not in ARITHMETIC:
            raise ValueError(
                f"arithmetic must be one of {sorted(ARITHMETIC)}, got {arithmetic!r}"
            )

        self.arithmetic = arithmetic

    def keep(
        self,
        boxes_for_nms: torch.Tensor,
        order: torch.Tensor,
        *,
        starts: torch.Tensor,
        counts: torch.Tensor,
        max_count: int,
        iou_threshold: float,
        rows: torch.Tensor,
        kept: torch.Tensor,
    ) -> None:
        """Launch both kernels on the current stream (see ``segmented_nms.NmsKernels``).

        Args:
            boxes_for_nms: (K, 4) float32, class offsets applied.
            order: (K,) int64 image-local sorted positions.
            starts: (B,) int64 first row per image.
            counts: (B,) int64 rows per image (0 skips the image).
            max_count: Host bound on ``counts``, at least 1.
            iou_threshold: NMS IoU threshold, rounded to float32 as torchvision.
            rows: (B, cap) int64 output.
            kept: (B,) int64 output.
        """
        batch_size, cap = rows.shape
        words = triton.cdiv(max_count, BLOCK)
        sum_mode, denominator_mode = ARITHMETIC[self.arithmetic]
        mask = torch.empty(
            (batch_size, max_count, words), dtype=torch.int64, device=rows.device
        )
        with _LAUNCH_LOCK:
            _iou_mask_kernel[(batch_size, words, words)](
                boxes_for_nms,
                order,
                starts,
                counts,
                mask,
                words,
                max_count,
                float32_threshold(iou_threshold),
                SUM_MODE=sum_mode,
                DENOMINATOR_MODE=denominator_mode,
                BLOCK=BLOCK,
                **IOU_LAUNCH,
            )
            _greedy_keep_kernel[(batch_size,)](
                mask,
                order,
                starts,
                counts,
                rows,
                kept,
                words,
                max_count,
                cap,
                WORDS=max(MIN_GREEDY_WORDS, triton.next_power_of_2(words)),
                BLOCK=BLOCK,
                **GREEDY_LAUNCH,
            )

    def identity(self) -> Dict[str, Any]:
        """Kernel source hash, Triton version, arithmetic and launch options.

        Returns:
            JSON-friendly facts.
        """
        facts = {
            "backend": "triton",
            "triton": triton.__version__,
            "torchvision": torchvision.__version__,
            "kernel_source": Path(__file__).name,
            "kernel_source_sha256": KERNEL_SOURCE_SHA256,
            "arithmetic": self.arithmetic,
            "iou_launch": dict(IOU_LAUNCH),
            "greedy_launch": dict(GREEDY_LAUNCH),
            "division": "tl.div_rn (IEEE round to nearest)",
            "max_min": "tl.maximum/tl.minimum: maxnum/minnum, as CUDA fmaxf/fminf",
        }

        return facts


def float32_threshold(value: float) -> float:
    """``value`` rounded to float32, as C++ ``const float threshold = double``.

    Args:
        value: Threshold from Python.

    Returns:
        The float32 value as a Python float (exactly representable).
    """
    rounded = struct.unpack("f", struct.pack("f", value))[0]

    return rounded
