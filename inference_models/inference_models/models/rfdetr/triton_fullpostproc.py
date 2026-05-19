"""Fused RF-DETR instance-segmentation post-processing in Triton.

For the common rfdetr-seg-nano path (batch=1, no static crop, STRETCH_TO
resize, class remapping active):

  rfdetr_postproc_triton_kernel  (grid = num_queries * num_classes_total)
    One program per (q, c) pair: sigmoid + class remap + conf threshold +
    cxcywh->xyxy + letterbox-denormalize + clip + banker's rounding;
    atomic_add into a counter to reserve a compact output slot.

  Mask upsample uses ``F.interpolate(bilinear, antialias=True)`` followed by
  a ``> 0`` threshold — bit-for-bit identical to the reference path's
  ``align_instance_segmentation_results`` mask handling. We reuse the
  reference here (rather than a custom kernel) because cuDNN's antialiased
  bilinear is hard to match in fp32 and the kernel-level win is in the
  filter step, not the upsample.
"""
from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def rfdetr_postproc_triton_kernel(
        logits_ptr,
        bboxes_ptr,
        threshold_ptr,
        class_map_ptr,
        combined_out_ptr,
        survivor_idx_out_ptr,
        counter_ptr,
        num_queries,
        num_classes_total,
        inference_w,
        inference_h,
        pad_left,
        pad_top,
        inv_scale_w,
        inv_scale_h,
        orig_w,
        orig_h,
        logits_stride_q,
        bboxes_stride_q,
        PER_CLASS: tl.constexpr,
        HAS_REMAPPING: tl.constexpr,
    ):
        # One program per (query, class). The reference path does top-k-flat
        # over the (Q*C) sigmoid grid (`num_select == num_queries`), so a
        # single query can contribute multiple detections — once per class
        # that survives remap + threshold. Per-query argmax would silently
        # drop the others.
        pid_q = tl.program_id(0)
        pid_c = tl.program_id(1)
        if pid_q >= num_queries or pid_c >= num_classes_total:
            return

        logit = tl.load(logits_ptr + pid_q * logits_stride_q + pid_c)

        if HAS_REMAPPING:
            top_c = tl.load(class_map_ptr + pid_c)
            valid = top_c >= 0
        else:
            top_c = pid_c
            valid = pid_c < num_classes_total

        abs_l = tl.abs(logit)
        z = tl.exp(-abs_l)
        sig_pos = 1.0 / (1.0 + z)
        sig_neg = z / (1.0 + z)
        conf = tl.where(logit >= 0.0, sig_pos, sig_neg)

        if PER_CLASS:
            safe_c = tl.where(valid, top_c, 0)
            thr = tl.load(threshold_ptr + safe_c)
        else:
            thr = tl.load(threshold_ptr)
        keep = valid & (conf > thr)

        if not keep:
            return

        # Match the non-Triton path's FP32 evaluation order for bit-parity.
        cx_pct = tl.load(bboxes_ptr + pid_q * bboxes_stride_q + 0)
        cy_pct = tl.load(bboxes_ptr + pid_q * bboxes_stride_q + 1)
        w_pct = tl.load(bboxes_ptr + pid_q * bboxes_stride_q + 2)
        h_pct = tl.load(bboxes_ptr + pid_q * bboxes_stride_q + 3)

        x1_pct = cx_pct - 0.5 * w_pct
        y1_pct = cy_pct - 0.5 * h_pct
        x2_pct = cx_pct + 0.5 * w_pct
        y2_pct = cy_pct + 0.5 * h_pct

        x1 = x1_pct * inference_w
        y1 = y1_pct * inference_h
        x2 = x2_pct * inference_w
        y2 = y2_pct * inference_h

        x1 = x1 - pad_left
        y1 = y1 - pad_top
        x2 = x2 - pad_left
        y2 = y2 - pad_top

        x1 = x1 * inv_scale_w
        y1 = y1 * inv_scale_h
        x2 = x2 * inv_scale_w
        y2 = y2 * inv_scale_h

        x1 = tl.maximum(tl.minimum(x1, orig_w), 0.0)
        y1 = tl.maximum(tl.minimum(y1, orig_h), 0.0)
        x2 = tl.maximum(tl.minimum(x2, orig_w), 0.0)
        y2 = tl.maximum(tl.minimum(y2, orig_h), 0.0)

        # Banker's rounding (half-to-even) matches torch.round().int().
        x1_r = tl.floor(x1 + 0.5)
        y1_r = tl.floor(y1 + 0.5)
        x2_r = tl.floor(x2 + 0.5)
        y2_r = tl.floor(y2 + 0.5)
        x1_i = x1_r.to(tl.int32)
        y1_i = y1_r.to(tl.int32)
        x2_i = x2_r.to(tl.int32)
        y2_i = y2_r.to(tl.int32)
        x1_i = tl.where(((x1_r - x1) == 0.5) & ((x1_i & 1) != 0), x1_i - 1, x1_i)
        y1_i = tl.where(((y1_r - y1) == 0.5) & ((y1_i & 1) != 0), y1_i - 1, y1_i)
        x2_i = tl.where(((x2_r - x2) == 0.5) & ((x2_i & 1) != 0), x2_i - 1, x2_i)
        y2_i = tl.where(((y2_r - y2) == 0.5) & ((y2_i & 1) != 0), y2_i - 1, y2_i)

        slot = tl.atomic_add(counter_ptr, 1)

        # Cap output at num_queries — mirrors the reference's flat top-K
        # cap. Host slices to min(counter, num_queries) to ignore overflow.
        if slot >= num_queries:
            return

        # Bitcast conf (fp32) as int32 so the whole record writes with int32
        # stores. Host views the same memory as int32 and extracts via
        # numpy.view(np.float32).
        conf_bits = conf.to(tl.float32, bitcast=False)
        conf_i32 = conf_bits.to(tl.int32, bitcast=True)
        base = slot * 6
        tl.store(combined_out_ptr + base + 0, x1_i)
        tl.store(combined_out_ptr + base + 1, y1_i)
        tl.store(combined_out_ptr + base + 2, x2_i)
        tl.store(combined_out_ptr + base + 3, y2_i)
        tl.store(combined_out_ptr + base + 4, conf_i32)
        tl.store(combined_out_ptr + base + 5, top_c)
        tl.store(survivor_idx_out_ptr + slot, pid_q.to(tl.int32))


_THRESHOLD_CACHE: dict = {}
_EMPTY_INT32 = torch.empty((1,), dtype=torch.int32)
_SCRATCH_CACHE: dict = {}
_CLASS_MAPPING_INT32_CACHE: dict = {}


def _get_scratch_buffers(num_queries: int, device: torch.device):
    key = (num_queries, device)
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        combined = torch.empty((num_queries, 6), dtype=torch.int32, device=device)
        survivor_idx = torch.empty((num_queries,), dtype=torch.int32, device=device)
        counter = torch.zeros((1,), dtype=torch.int32, device=device)
        cached = (combined, survivor_idx, counter)
        _SCRATCH_CACHE[key] = cached
    return cached


def _get_class_mapping_int32(class_mapping: torch.Tensor, device: torch.device) -> torch.Tensor:
    if class_mapping.dtype == torch.int32 and class_mapping.device == device and class_mapping.is_contiguous():
        return class_mapping
    key = (id(class_mapping), device)
    cached = _CLASS_MAPPING_INT32_CACHE.get(key)
    if cached is not None:
        return cached
    cached = class_mapping.to(dtype=torch.int32, device=device).contiguous()
    _CLASS_MAPPING_INT32_CACHE[key] = cached
    return cached


def _prepare_threshold(threshold, device: torch.device, num_classes: int):
    if isinstance(threshold, torch.Tensor):
        t = threshold
        if t.dtype != torch.float32 or t.device != device or not t.is_contiguous():
            t = t.to(dtype=torch.float32, device=device).contiguous()
        return t, True
    key = (float(threshold), device)
    cached = _THRESHOLD_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([float(threshold)], dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = cached
    return cached, False


def rfdetr_triton_postproc(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    threshold: "torch.Tensor | float",
    num_classes: int,
    class_mapping: Optional[torch.Tensor],
    inference_size_wh: Tuple[int, int],
    pad_ltrb: Tuple[int, int, int, int],
    scale_wh: Tuple[float, float],
    orig_size_wh: Tuple[int, int],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    "torch.cuda.Event",
]:
    """Filter step only — returns (combined, survivor_idx, counter, done_event).

    Buffers are unsliced — the caller waits on ``done_event``, reads
    ``counter`` to learn n_survivors, and slices to ``[:n_survivors]``.
    ``combined[:, 4]`` holds fp32 conf as int32 bits; reinterpret on the
    host with ``.view(torch.float32)``. The mask upsample is intentionally
    left to the caller (torch ``F.interpolate``) so the result is bit-exact
    with the non-postproc reference path.
    """
    device = bboxes.device
    num_queries, num_classes_total = logits.shape[1], logits.shape[2]

    logits_2d = logits[0] if logits[0].is_contiguous() else logits[0].contiguous()
    bboxes_2d = bboxes[0] if bboxes[0].is_contiguous() else bboxes[0].contiguous()

    combined, survivor_idx, counter = _get_scratch_buffers(num_queries, device)
    counter.zero_()

    thr_tensor, per_class = _prepare_threshold(threshold, device, num_classes)

    if class_mapping is not None:
        has_remap = True
        cmap = _get_class_mapping_int32(class_mapping, device)
    else:
        has_remap = False
        cmap = _EMPTY_INT32.to(device, non_blocking=True)

    inf_w, inf_h = inference_size_wh
    pad_l, pad_t, _, _ = pad_ltrb
    sw, sh = scale_wh
    orig_w, orig_h = orig_size_wh
    per_class_constexpr = 1 if per_class else 0
    has_remap_constexpr = 1 if has_remap else 0

    rfdetr_postproc_triton_kernel[(num_queries, num_classes_total)](
        logits_2d,
        bboxes_2d,
        thr_tensor,
        cmap,
        combined,
        survivor_idx,
        counter,
        num_queries,
        num_classes_total,
        int(inf_w),
        int(inf_h),
        int(pad_l),
        int(pad_t),
        float(1.0 / sw),
        float(1.0 / sh),
        int(orig_w),
        int(orig_h),
        logits_2d.stride(0),
        bboxes_2d.stride(0),
        PER_CLASS=tl.constexpr(per_class_constexpr),
        HAS_REMAPPING=tl.constexpr(has_remap_constexpr),
    )

    done_event = torch.cuda.Event()
    done_event.record(torch.cuda.current_stream(device))

    return combined, survivor_idx, counter, done_event
