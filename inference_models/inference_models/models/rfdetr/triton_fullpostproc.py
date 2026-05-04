"""Fused RF-DETR instance-segmentation post-processing in Triton.

Two kernels replace the post-TRT chain for the common rfdetr-seg-nano path
(batch=1, no static crop, STRETCH_TO resize, class remapping active):

  _rfdetr_fullpost_filter_kernel  (grid = num_queries)
    sigmoid argmax + class remap + conf threshold + cxcywh->xyxy +
    letterbox-denormalize + clip + banker's rounding; atomic_add into a
    counter to reserve a compact output slot.

  _rfdetr_fullpost_mask_kernel_compact  (grid = num_queries * tile_y * tile_x)
    Bilinear upsample masks (e.g. 78x78 -> orig_h x orig_w) + threshold > 0 +
    uint8 emit. Early-exits on s >= counter[0] without an intermediate sync.
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
    def _rfdetr_fullpost_filter_kernel(
        logits_ptr,
        bboxes_ptr,
        threshold_ptr,
        class_map_ptr,
        combined_out_ptr,
        survivor_idx_out_ptr,
        mask_any_out_ptr,
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
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_queries:
            return
        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < num_classes_total

        logits_row = tl.load(
            logits_ptr + pid * logits_stride_q + offs_c,
            mask=mask_c,
            other=-float("inf"),
        )
        max_val = tl.max(logits_row, axis=0)
        BIG = 1 << 30
        is_max = logits_row == max_val
        idx_or_big = tl.where(is_max & mask_c, offs_c, BIG)
        raw_c = tl.min(idx_or_big, axis=0)

        if HAS_REMAPPING:
            top_c = tl.load(class_map_ptr + raw_c)
            valid = top_c >= 0
        else:
            top_c = raw_c
            valid = raw_c < num_classes_total

        abs_max = tl.abs(max_val)
        z = tl.exp(-abs_max)
        sig_pos = 1.0 / (1.0 + z)
        sig_neg = z / (1.0 + z)
        conf = tl.where(max_val >= 0.0, sig_pos, sig_neg)

        if PER_CLASS:
            safe_c = tl.where(valid, top_c, 0)
            thr = tl.load(threshold_ptr + safe_c)
        else:
            thr = tl.load(threshold_ptr)
        keep = valid & (conf > thr)

        if not keep:
            return

        # Match the non-Triton path's FP32 evaluation order for bit-parity.
        cx_pct = tl.load(bboxes_ptr + pid * bboxes_stride_q + 0)
        cy_pct = tl.load(bboxes_ptr + pid * bboxes_stride_q + 1)
        w_pct = tl.load(bboxes_ptr + pid * bboxes_stride_q + 2)
        h_pct = tl.load(bboxes_ptr + pid * bboxes_stride_q + 3)

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
        tl.store(survivor_idx_out_ptr + slot, pid.to(tl.int32))
        tl.store(mask_any_out_ptr + slot, 0)


    @triton.jit
    def _rfdetr_fullpost_mask_kernel_compact(
        masks_ptr,
        survivor_idx_ptr,
        counter_ptr,
        out_ptr,
        mask_any_ptr,
        mask_h,
        mask_w,
        orig_h,
        orig_w,
        mask_scale_y,
        mask_scale_x,
        masks_stride_q,
        masks_stride_h,
        out_stride_s,
        out_stride_h,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        s = tl.program_id(0)
        tile_y = tl.program_id(1)
        tile_x = tl.program_id(2)

        # GPU-side early exit — skip programs past the live survivor count.
        n_survivors = tl.load(counter_ptr)
        if s >= n_survivors:
            return

        q = tl.load(survivor_idx_ptr + s)

        offs_y = tile_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = tile_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_yy = offs_y < orig_h
        mask_xx = offs_x < orig_w
        m_outbox = mask_yy[:, None] & mask_xx[None, :]

        src_y_f = (offs_y.to(tl.float32) + 0.5) * mask_scale_y - 0.5
        src_x_f = (offs_x.to(tl.float32) + 0.5) * mask_scale_x - 0.5
        src_y_2d = src_y_f[:, None]
        src_x_2d = src_x_f[None, :]

        y0 = tl.floor(src_y_2d).to(tl.int32)
        x0 = tl.floor(src_x_2d).to(tl.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        dy = src_y_2d - y0.to(tl.float32)
        dx = src_x_2d - x0.to(tl.float32)

        y0c = tl.maximum(tl.minimum(y0, mask_h - 1), 0)
        y1c = tl.maximum(tl.minimum(y1, mask_h - 1), 0)
        x0c = tl.maximum(tl.minimum(x0, mask_w - 1), 0)
        x1c = tl.maximum(tl.minimum(x1, mask_w - 1), 0)

        base = q * masks_stride_q

        p00 = tl.load(masks_ptr + base + y0c * masks_stride_h + x0c, mask=m_outbox, other=0.0)
        p01 = tl.load(masks_ptr + base + y0c * masks_stride_h + x1c, mask=m_outbox, other=0.0)
        p10 = tl.load(masks_ptr + base + y1c * masks_stride_h + x0c, mask=m_outbox, other=0.0)
        p11 = tl.load(masks_ptr + base + y1c * masks_stride_h + x1c, mask=m_outbox, other=0.0)

        w_tl = (1.0 - dy) * (1.0 - dx)
        w_tr = (1.0 - dy) * dx
        w_bl = dy * (1.0 - dx)
        w_br = dy * dx
        val = p00 * w_tl + p01 * w_tr + p10 * w_bl + p11 * w_br
        bin_val = (val > 0.0).to(tl.int8)

        out_offsets = offs_y[:, None] * out_stride_h + offs_x[None, :]
        tl.store(out_ptr + s * out_stride_s + out_offsets, bin_val, mask=m_outbox)

        tile_any = tl.max(bin_val.to(tl.int32), axis=0)
        tile_any2 = tl.max(tile_any, axis=0)
        tl.atomic_max(mask_any_ptr + s, tile_any2)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


_THRESHOLD_CACHE: dict = {}
_EMPTY_INT32 = torch.empty((1,), dtype=torch.int32)
_MASK_BIN_BUFFER_CACHE: dict = {}
_SCRATCH_CACHE: dict = {}
_CLASS_MAPPING_INT32_CACHE: dict = {}


def _get_scratch_buffers(num_queries: int, device: torch.device):
    key = (num_queries, device)
    cached = _SCRATCH_CACHE.get(key)
    if cached is None:
        combined = torch.empty((num_queries, 6), dtype=torch.int32, device=device)
        survivor_idx = torch.empty((num_queries,), dtype=torch.int32, device=device)
        mask_any = torch.empty((num_queries,), dtype=torch.int32, device=device)
        counter = torch.zeros((1,), dtype=torch.int32, device=device)
        cached = (combined, survivor_idx, mask_any, counter)
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


def _get_mask_bin_buffer(
    capacity: int, orig_h: int, orig_w: int, device: torch.device
) -> torch.Tensor:
    # Rows beyond n_survivors may hold stale data from prior frames; callers
    # must size their read by the atomic counter.
    key = (capacity, orig_h, orig_w, device)
    buf = _MASK_BIN_BUFFER_CACHE.get(key)
    if buf is None:
        buf = torch.empty(
            (capacity, orig_h, orig_w), dtype=torch.uint8, device=device
        )
        _MASK_BIN_BUFFER_CACHE[key] = buf
    return buf


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


def triton_rfdetr_fullpost(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
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
    torch.Tensor,
    "torch.cuda.Event",
]:
    """Returns (combined, mask_bin, mask_any, counter, done_event). Buffers
    are unsliced — the caller DtoH's ``counter`` to learn n_survivors and
    slices to ``[:n_survivors]``. ``combined[:, 4]`` holds fp32 conf as
    int32 bits; use ``numpy.view(np.float32)`` on the host."""
    assert TRITON_AVAILABLE, "triton not available"
    assert bboxes.is_cuda and logits.is_cuda and masks.is_cuda
    assert bboxes.shape[0] == 1 and logits.shape[0] == 1 and masks.shape[0] == 1, "batch=1 only"

    device = bboxes.device
    num_queries, num_classes_total = logits.shape[1], logits.shape[2]
    _, _, mask_h, mask_w = masks.shape

    logits_2d = logits[0] if logits[0].is_contiguous() else logits[0].contiguous()
    bboxes_2d = bboxes[0] if bboxes[0].is_contiguous() else bboxes[0].contiguous()
    masks_3d = masks[0] if masks[0].is_contiguous() else masks[0].contiguous()

    combined, survivor_idx, mask_any, counter = _get_scratch_buffers(
        num_queries, device
    )
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

    BLOCK_C = max(32, _next_pow2(num_classes_total))
    _rfdetr_fullpost_filter_kernel[(num_queries,)](
        logits_2d,
        bboxes_2d,
        thr_tensor,
        cmap,
        combined,
        survivor_idx,
        mask_any,
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
        PER_CLASS=1 if per_class else 0,
        HAS_REMAPPING=1 if has_remap else 0,
        BLOCK_C=BLOCK_C,
    )

    mask_bin_full = _get_mask_bin_buffer(num_queries, orig_h, orig_w, device)

    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        num_queries,
        (orig_h + BLOCK_H - 1) // BLOCK_H,
        (orig_w + BLOCK_W - 1) // BLOCK_W,
    )
    _rfdetr_fullpost_mask_kernel_compact[grid](
        masks_3d,
        survivor_idx,
        counter,
        mask_bin_full,
        mask_any,
        int(mask_h),
        int(mask_w),
        int(orig_h),
        int(orig_w),
        float(mask_h / orig_h),
        float(mask_w / orig_w),
        masks_3d.stride(0),
        masks_3d.stride(1),
        mask_bin_full.stride(0),
        mask_bin_full.stride(1),
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )

    done_event = torch.cuda.Event()
    done_event.record(torch.cuda.current_stream(device))

    return combined, mask_bin_full, mask_any, counter, done_event
