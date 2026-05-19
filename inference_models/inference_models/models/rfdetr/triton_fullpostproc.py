"""Single-kernel RF-DETR instance-segmentation post-processing in Triton.

Fast path scope:
- batch size == 1
- RF-DETR seg TRT tensor shapes: Q=100, C=91, Mh=Mw=78
- no static crop, no letterbox padding, no nonsquare intermediate size
- output mask resize is an upsample (the common benchmark / parity case)

Within one Triton launch, each program owns one output detection rank and:
- performs flat top-k over the (Q, C) sigmoid grid
- applies class remap + confidence filtering
- denormalizes / rescales / rounds the selected box
- resizes the selected 78x78 mask to the original image size and thresholds it
- optionally bit-packs the dense mask sidecar for lower DtoH transfer volume

The resize uses cached 2-tap closed-form bilinear tables, so the per-inference
hot path remains a single Triton launch without any CUDA bootstrap probes.

Host-side work is limited to slicing the preallocated buffers once the kernel
completes and wrapping them in ``InstanceDetections`` / RLE containers.
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


FASTPATH_NUM_QUERIES = 100
FASTPATH_NUM_CLASSES_TOTAL = 91
FASTPATH_MASK_H = 78
FASTPATH_MASK_W = 78
_TOPK_PAD = 128
_CLASS_BLOCK = 128
_MASK_TILE_H = 8
_MASK_TILE_W = 64
_RLE_TILE_H = 32
_RLE_TILE_W = 16
_RLE_MERGE_TILE = 32
_MAX_U32 = 0xFFFFFFFF


if TRITON_AVAILABLE:

    @triton.jit
    def rfdetr_fullpostproc_triton_kernel(
        logits_ptr,
        bboxes_ptr,
        masks_ptr,
        threshold_ptr,
        class_map_ptr,
        y_indices_ptr,
        y_weights_ptr,
        x_indices_ptr,
        x_weights_ptr,
        rle_counts_ptr,
        rle_lengths_ptr,
        combined_out_ptr,
        mask_out_ptr,
        counter_ptr,
        orig_w,
        orig_h,
        logits_stride_q,
        bboxes_stride_q,
        masks_stride_q,
        masks_stride_h,
        masks_stride_w,
        rle_counts_stride_q,
        rle_lengths_stride_q,
        mask_out_stride_q,
        mask_out_stride_h,
        mask_out_stride_w,
        PER_CLASS: tl.constexpr,
        HAS_REMAPPING: tl.constexpr,
        EMIT_RLE: tl.constexpr,
        PACK_DENSE_MASKS: tl.constexpr,
        NUM_QUERIES: tl.constexpr,
        NUM_CLASSES_TOTAL: tl.constexpr,
        MASK_H: tl.constexpr,
        MASK_W: tl.constexpr,
        TOPK_PAD: tl.constexpr,
        CLASS_BLOCK: tl.constexpr,
        MASK_TILE_H: tl.constexpr,
        MASK_TILE_W: tl.constexpr,
        RLE_TILE_H: tl.constexpr,
        RLE_TILE_W: tl.constexpr,
        RLE_MERGE_TILE: tl.constexpr,
    ):
        pid_det = tl.program_id(0)

        # Maintain the reference flat top-k exactly: top 100 scores over the
        # 100x91 sigmoid grid, before class remap / thresholding.
        top_packed = tl.zeros((TOPK_PAD,), dtype=tl.int64)
        class_offsets = tl.arange(0, CLASS_BLOCK)
        rank_offsets = tl.arange(0, TOPK_PAD)
        top_limit = tl.full((), NUM_QUERIES, tl.int32)
        num_classes_total = tl.full((), NUM_CLASSES_TOTAL, tl.int32)

        for q in tl.range(0, NUM_QUERIES, num_stages=1):
            valid_class = class_offsets < NUM_CLASSES_TOTAL
            logit = tl.load(
                logits_ptr + q * logits_stride_q + class_offsets,
                mask=valid_class,
                other=-float("inf"),
            )
            abs_l = tl.abs(logit)
            z = tl.exp(-abs_l)
            sig_pos = 1.0 / (1.0 + z)
            sig_neg = z / (1.0 + z)
            conf = tl.where(logit >= 0.0, sig_pos, sig_neg)
            conf_bits = conf.to(tl.float32, bitcast=False).to(tl.int32, bitcast=True)
            flat_idx = q * NUM_CLASSES_TOTAL + class_offsets
            packed = tl.where(
                valid_class,
                (conf_bits.to(tl.int64) << 32) | flat_idx.to(tl.int64),
                tl.zeros((CLASS_BLOCK,), dtype=tl.int64),
            )
            merged = tl.reshape(tl.join(top_packed, packed), (TOPK_PAD + CLASS_BLOCK,))
            top_packed = tl.topk(merged, k=TOPK_PAD)

        selected_q = tl.full((), 0, tl.int32)
        selected_c = tl.full((), 0, tl.int32)
        selected_conf = tl.full((), 0.0, tl.float32)
        keep_count = tl.full((), 0, tl.int32)

        for rank in tl.range(0, NUM_QUERIES, num_stages=1):
            packed = tl.sum(
                tl.where(
                    rank_offsets == rank,
                    top_packed,
                    tl.zeros((TOPK_PAD,), dtype=tl.int64),
                ),
                axis=0,
            )
            conf_bits = (packed >> 32).to(tl.int32)
            conf = conf_bits.to(tl.float32, bitcast=True)
            flat_idx = (packed & tl.full((), 0xFFFFFFFF, tl.int64)).to(tl.int32)
            query_idx = flat_idx // num_classes_total
            raw_class = flat_idx - query_idx * num_classes_total

            if HAS_REMAPPING:
                mapped_class = tl.load(class_map_ptr + raw_class)
                valid = mapped_class >= 0
            else:
                mapped_class = raw_class
                valid = raw_class < top_limit

            if PER_CLASS:
                safe_class = tl.where(valid, mapped_class, 0)
                threshold = tl.load(threshold_ptr + safe_class)
            else:
                threshold = tl.load(threshold_ptr)

            keep = valid & (conf > threshold)
            select_now = keep & (keep_count == pid_det)
            selected_q = tl.where(select_now, query_idx, selected_q)
            selected_c = tl.where(select_now, mapped_class, selected_c)
            selected_conf = tl.where(select_now, conf, selected_conf)
            keep_count += keep.to(tl.int32)

        if pid_det == 0:
            tl.store(counter_ptr, keep_count)

        base = pid_det * 6
        active = pid_det < keep_count
        if not active:
            tl.store(combined_out_ptr + base + 0, 0)
            tl.store(combined_out_ptr + base + 1, 0)
            tl.store(combined_out_ptr + base + 2, 0)
            tl.store(combined_out_ptr + base + 3, 0)
            tl.store(combined_out_ptr + base + 4, 0)
            tl.store(combined_out_ptr + base + 5, -1)
            return

        cx_pct = tl.load(bboxes_ptr + selected_q * bboxes_stride_q + 0)
        cy_pct = tl.load(bboxes_ptr + selected_q * bboxes_stride_q + 1)
        w_pct = tl.load(bboxes_ptr + selected_q * bboxes_stride_q + 2)
        h_pct = tl.load(bboxes_ptr + selected_q * bboxes_stride_q + 3)

        x1_pct = cx_pct - 0.5 * w_pct
        y1_pct = cy_pct - 0.5 * h_pct
        x2_pct = cx_pct + 0.5 * w_pct
        y2_pct = cy_pct + 0.5 * h_pct

        orig_w_f = orig_w.to(tl.float32)
        orig_h_f = orig_h.to(tl.float32)
        x1 = x1_pct * orig_w_f
        y1 = y1_pct * orig_h_f
        x2 = x2_pct * orig_w_f
        y2 = y2_pct * orig_h_f

        # Match torch.round(...).int() with half-to-even tie handling.
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

        conf_bits_out = selected_conf.to(tl.float32, bitcast=False).to(
            tl.int32, bitcast=True
        )
        tl.store(combined_out_ptr + base + 0, x1_i)
        tl.store(combined_out_ptr + base + 1, y1_i)
        tl.store(combined_out_ptr + base + 2, x2_i)
        tl.store(combined_out_ptr + base + 3, y2_i)
        tl.store(combined_out_ptr + base + 4, conf_bits_out)
        tl.store(combined_out_ptr + base + 5, selected_c)

        mask_base = masks_ptr + selected_q * masks_stride_q
        if EMIT_RLE:
            row_offsets = tl.arange(0, RLE_TILE_H)
            col_offsets = tl.arange(0, RLE_TILE_W)
            merge_offsets = tl.arange(0, RLE_MERGE_TILE)
            lengths_row_ptr = rle_lengths_ptr + pid_det * rle_lengths_stride_q
            counts_stride_col = orig_h + 1

            for out_x in tl.range(0, orig_w, RLE_TILE_W, num_stages=1):
                x = out_x + col_offsets
                x_mask = x < orig_w
                x_table_offset = x * 2
                x_idx_a = tl.load(
                    x_indices_ptr + x_table_offset + 0, mask=x_mask, other=0
                )
                x_idx_b = tl.load(
                    x_indices_ptr + x_table_offset + 1, mask=x_mask, other=0
                )
                wx_a = tl.load(
                    x_weights_ptr + x_table_offset + 0, mask=x_mask, other=0.0
                )
                wx_b = tl.load(
                    x_weights_ptr + x_table_offset + 1, mask=x_mask, other=0.0
                )
                col_base = (
                    rle_counts_ptr
                    + pid_det * rle_counts_stride_q
                    + x * counts_stride_col
                )
                run_length_vec = tl.zeros((RLE_TILE_W,), dtype=tl.int32)
                prev_value_vec = tl.zeros((RLE_TILE_W,), dtype=tl.int32)
                counts_idx_vec = tl.zeros((RLE_TILE_W,), dtype=tl.int32)

                for out_y in tl.range(0, orig_h, RLE_TILE_H, num_stages=1):
                    y = out_y + row_offsets
                    y_mask = y < orig_h
                    tile_mask = y_mask[:, None] & x_mask[None, :]
                    y_table_offset = y * 2
                    y_idx_a = tl.load(
                        y_indices_ptr + y_table_offset + 0, mask=y_mask, other=0
                    )
                    y_idx_b = tl.load(
                        y_indices_ptr + y_table_offset + 1, mask=y_mask, other=0
                    )
                    wy_a = tl.load(
                        y_weights_ptr + y_table_offset + 0, mask=y_mask, other=0.0
                    )
                    wy_b = tl.load(
                        y_weights_ptr + y_table_offset + 1, mask=y_mask, other=0.0
                    )

                    m00 = tl.load(
                        mask_base
                        + y_idx_a[:, None] * masks_stride_h
                        + x_idx_a[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m01 = tl.load(
                        mask_base
                        + y_idx_a[:, None] * masks_stride_h
                        + x_idx_b[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m10 = tl.load(
                        mask_base
                        + y_idx_b[:, None] * masks_stride_h
                        + x_idx_a[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m11 = tl.load(
                        mask_base
                        + y_idx_b[:, None] * masks_stride_h
                        + x_idx_b[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    interp_top = wx_a[None, :] * m00 + wx_b[None, :] * m01
                    interp_bottom = wx_a[None, :] * m10 + wx_b[None, :] * m11
                    bits = (
                        (wy_a[:, None] * interp_top + wy_b[:, None] * interp_bottom)
                        > 0.0
                    ).to(tl.int32)

                    for local_y in tl.static_range(0, RLE_TILE_H):
                        valid_row = out_y + local_y < orig_h
                        row_mask = (row_offsets[:, None] == local_y).to(tl.int32)
                        bit_row = tl.sum(bits * row_mask, axis=0)
                        update_mask = x_mask & valid_row
                        change_row = update_mask & (bit_row != prev_value_vec)
                        tl.store(
                            col_base + counts_idx_vec, run_length_vec, mask=change_row
                        )
                        counts_idx_vec += change_row.to(tl.int32)
                        prev_value_vec = tl.where(update_mask, bit_row, prev_value_vec)
                        run_length_vec = tl.where(
                            update_mask,
                            tl.where(change_row, 1, run_length_vec + 1),
                            run_length_vec,
                        )

                tl.store(col_base + counts_idx_vec, run_length_vec, mask=x_mask)
                tl.store(
                    lengths_row_ptr + x,
                    counts_idx_vec + 1,
                    mask=x_mask,
                )

            final_len = tl.full((), 0, tl.int32)
            prev_end_value = tl.full((), 0, tl.int32)
            final_counts_ptr = rle_counts_ptr + pid_det * rle_counts_stride_q

            for out_x in tl.range(0, orig_w, RLE_TILE_W, num_stages=1):
                x = out_x + col_offsets
                x_mask = x < orig_w
                col_lengths = tl.load(lengths_row_ptr + x, mask=x_mask, other=0)

                for local_x in tl.static_range(0, RLE_TILE_W):
                    col_x = out_x + local_x
                    valid_col = col_x < orig_w
                    col_mask = (col_offsets == local_x).to(tl.int32)
                    col_len = tl.sum(col_lengths * col_mask, axis=0)
                    col_counts_ptr = (
                        rle_counts_ptr
                        + pid_det * rle_counts_stride_q
                        + col_x * counts_stride_col
                    )

                    is_first_col = final_len == 0
                    merge_with_zero = (~is_first_col) & (prev_end_value == 0)
                    do_merge = valid_col & merge_with_zero
                    first_count = tl.load(col_counts_ptr + 0, mask=valid_col, other=0)
                    prev_last = tl.load(
                        final_counts_ptr + final_len - 1,
                        mask=do_merge,
                        other=0,
                    )
                    tl.store(
                        final_counts_ptr + final_len - 1,
                        prev_last + first_count,
                        mask=do_merge,
                    )
                    src_start = tl.where(merge_with_zero, 1, 0)
                    dst_start = tl.where(is_first_col, 0, final_len)
                    copy_len = tl.where(merge_with_zero, col_len - 1, col_len)

                    for merge_off in tl.range(
                        0, counts_stride_col, RLE_MERGE_TILE, num_stages=1
                    ):
                        idx = merge_off + merge_offsets
                        copy_mask = valid_col & (idx < copy_len)
                        vals = tl.load(
                            col_counts_ptr + src_start + idx,
                            mask=copy_mask,
                            other=0,
                        )
                        tl.store(
                            final_counts_ptr + dst_start + idx,
                            vals,
                            mask=copy_mask,
                        )

                    updated_final_len = tl.where(
                        is_first_col,
                        copy_len,
                        final_len + copy_len,
                    )
                    final_len = tl.where(valid_col, updated_final_len, final_len)
                    prev_end_value = tl.where(
                        valid_col, (col_len - 1) & 1, prev_end_value
                    )

            tl.store(lengths_row_ptr + 0, final_len)
        else:
            row_offsets = tl.arange(0, MASK_TILE_H)
            col_offsets = tl.arange(0, MASK_TILE_W)
            if PACK_DENSE_MASKS:
                packed_col_offsets = tl.arange(0, MASK_TILE_W // 8)
                bit_weights = (1 << tl.arange(0, 8)).to(tl.int32)

            for out_y in tl.range(0, orig_h, MASK_TILE_H, num_stages=1):
                y = out_y + row_offsets
                y_mask = y < orig_h
                y_table_offset = y * 2
                y_idx_a = tl.load(
                    y_indices_ptr + y_table_offset + 0, mask=y_mask, other=0
                )
                y_idx_b = tl.load(
                    y_indices_ptr + y_table_offset + 1, mask=y_mask, other=0
                )
                wy_a = tl.load(
                    y_weights_ptr + y_table_offset + 0, mask=y_mask, other=0.0
                )
                wy_b = tl.load(
                    y_weights_ptr + y_table_offset + 1, mask=y_mask, other=0.0
                )

                for out_x in tl.range(0, orig_w, MASK_TILE_W, num_stages=1):
                    x = out_x + col_offsets
                    x_mask = x < orig_w
                    tile_mask = y_mask[:, None] & x_mask[None, :]
                    x_table_offset = x * 2
                    x_idx_a = tl.load(
                        x_indices_ptr + x_table_offset + 0, mask=x_mask, other=0
                    )
                    x_idx_b = tl.load(
                        x_indices_ptr + x_table_offset + 1, mask=x_mask, other=0
                    )
                    wx_a = tl.load(
                        x_weights_ptr + x_table_offset + 0, mask=x_mask, other=0.0
                    )
                    wx_b = tl.load(
                        x_weights_ptr + x_table_offset + 1, mask=x_mask, other=0.0
                    )

                    m00 = tl.load(
                        mask_base
                        + y_idx_a[:, None] * masks_stride_h
                        + x_idx_a[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m01 = tl.load(
                        mask_base
                        + y_idx_a[:, None] * masks_stride_h
                        + x_idx_b[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m10 = tl.load(
                        mask_base
                        + y_idx_b[:, None] * masks_stride_h
                        + x_idx_a[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    m11 = tl.load(
                        mask_base
                        + y_idx_b[:, None] * masks_stride_h
                        + x_idx_b[None, :] * masks_stride_w,
                        mask=tile_mask,
                        other=0.0,
                    )
                    interp_top = wx_a[None, :] * m00 + wx_b[None, :] * m01
                    interp_bottom = wx_a[None, :] * m10 + wx_b[None, :] * m11
                    interp = wy_a[:, None] * interp_top + wy_b[:, None] * interp_bottom
                    bits = (interp > 0.0).to(tl.int32)
                    if PACK_DENSE_MASKS:
                        packed = tl.sum(
                            tl.reshape(bits, (MASK_TILE_H, MASK_TILE_W // 8, 8))
                            * bit_weights[None, None, :],
                            axis=2,
                        ).to(tl.uint8)
                        byte_mask = (
                            tl.sum(
                                tl.reshape(x_mask.to(tl.int32), (MASK_TILE_W // 8, 8)),
                                axis=1,
                            )
                            > 0
                        )
                        out_ptr = (
                            mask_out_ptr
                            + pid_det * mask_out_stride_q
                            + y[:, None] * mask_out_stride_h
                            + (out_x // 8 + packed_col_offsets)[None, :]
                            * mask_out_stride_w
                        )
                        tl.store(
                            out_ptr,
                            packed,
                            mask=y_mask[:, None] & byte_mask[None, :],
                        )
                    else:
                        out_ptr = (
                            mask_out_ptr
                            + pid_det * mask_out_stride_q
                            + y[:, None] * mask_out_stride_h
                            + x[None, :] * mask_out_stride_w
                        )
                        tl.store(out_ptr, bits.to(tl.uint8), mask=tile_mask)


_THRESHOLD_CACHE: dict = {}
_EMPTY_INT32 = torch.empty((1,), dtype=torch.int32)
_EMPTY_INT32_DEVICE_CACHE: dict = {}
_SCRATCH_CACHE: dict = {}
_CLASS_MAPPING_INT32_CACHE: dict = {}
_AA_RESIZE_CACHE: dict = {}
_RLE_SCRATCH_CACHE: dict = {}


def _build_resize_axis_tables(
    in_size: int,
    out_size: int,
    device: torch.device,
    horizontal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns exact 2-tap tables extracted from the reference CUDA resize op."""

    basis = torch.eye(in_size, dtype=torch.float32, device=device)
    if horizontal:
        resized = torch.nn.functional.interpolate(
            basis[:, None, None, :],
            size=(1, out_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0, 0, :]
    else:
        resized = torch.nn.functional.interpolate(
            basis[:, None, :, None],
            size=(out_size, 1),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[:, 0, :, 0]

    resized_cpu = resized.cpu()
    indices = torch.empty((out_size, 2), dtype=torch.int32)
    weights = torch.zeros((out_size, 2), dtype=torch.float32)
    for out_idx in range(out_size):
        support = torch.nonzero(
            resized_cpu[:, out_idx].abs() > 0, as_tuple=False
        ).flatten()
        if support.numel() == 0:
            raise ValueError(
                f"Reference bilinear AA resize produced no support for axis "
                f"{out_idx} of shape {in_size}->{out_size}."
            )
        if support.numel() > 2:
            raise ValueError(
                "RF-DETR Triton fullpost fast path only supports 2-tap "
                "upsample resize tables, but the reference resize produced "
                f"{support.numel()} taps for shape {in_size}->{out_size}."
            )

        idx_a = int(support[0])
        indices[out_idx, 0] = idx_a
        weights[out_idx, 0] = resized_cpu[idx_a, out_idx]
        if support.numel() == 1:
            indices[out_idx, 1] = idx_a
            weights[out_idx, 1] = 0.0
        else:
            idx_b = int(support[1])
            indices[out_idx, 1] = idx_b
            weights[out_idx, 1] = resized_cpu[idx_b, out_idx]
    return indices.to(device=device, non_blocking=True), weights.to(
        device=device, non_blocking=True
    )


def _get_resize_tables(
    orig_h: int,
    orig_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (device, orig_h, orig_w)
    cached = _AA_RESIZE_CACHE.get(key)
    if cached is None:
        y_indices, y_weights = _build_resize_axis_tables(
            in_size=FASTPATH_MASK_H,
            out_size=orig_h,
            device=device,
            horizontal=False,
        )
        x_indices, x_weights = _build_resize_axis_tables(
            in_size=FASTPATH_MASK_W,
            out_size=orig_w,
            device=device,
            horizontal=True,
        )
        cached = (y_indices, y_weights, x_indices, x_weights)
        _AA_RESIZE_CACHE[key] = cached
    y_indices, y_weights, x_indices, x_weights = cached
    return y_indices, y_weights, x_indices, x_weights


def _get_scratch_buffers(
    num_queries: int,
    orig_h: int,
    orig_w: int,
    device: torch.device,
    pack_dense_masks: bool,
):
    key = (device, pack_dense_masks)
    cached = _SCRATCH_CACHE.get(key)
    mask_w = (orig_w + 7) // 8 if pack_dense_masks else orig_w
    shape = (num_queries, orig_h, mask_w)
    if cached is None or cached[0] != shape:
        combined = torch.empty((num_queries, 6), dtype=torch.int32, device=device)
        mask_bin = torch.empty(
            (num_queries, orig_h, mask_w), dtype=torch.uint8, device=device
        )
        counter = torch.empty((1,), dtype=torch.int32, device=device)
        cached = (shape, combined, mask_bin, counter)
        _SCRATCH_CACHE[key] = cached
    _, combined, mask_bin, counter = cached
    return combined, mask_bin, counter


def _get_rle_buffers(
    num_queries: int,
    orig_h: int,
    orig_w: int,
    device: torch.device,
):
    cached = _RLE_SCRATCH_CACHE.get(device)
    max_counts = orig_w * (orig_h + 1)
    shape = (num_queries, max_counts, orig_w)
    if cached is None or cached[0] != shape:
        counts = torch.empty(
            (num_queries, max_counts), dtype=torch.int32, device=device
        )
        lengths = torch.empty((num_queries, orig_w), dtype=torch.int32, device=device)
        cached = (shape, counts, lengths)
        _RLE_SCRATCH_CACHE[device] = cached
    _, counts, lengths = cached
    return counts, lengths


def _get_class_mapping_int32(
    class_mapping: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if (
        class_mapping.dtype == torch.int32
        and class_mapping.device == device
        and class_mapping.is_contiguous()
    ):
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
        tensor = threshold
        if (
            tensor.dtype != torch.float32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            tensor = tensor.to(dtype=torch.float32, device=device).contiguous()
        return tensor, True
    key = (float(threshold), device)
    cached = _THRESHOLD_CACHE.get(key)
    if cached is None:
        cached = torch.tensor([float(threshold)], dtype=torch.float32, device=device)
        _THRESHOLD_CACHE[key] = cached
    return cached, False


def _get_empty_int32_on_device(device: torch.device) -> torch.Tensor:
    cached = _EMPTY_INT32_DEVICE_CACHE.get(device)
    if cached is None:
        cached = torch.empty((1,), dtype=torch.int32, device=device)
        _EMPTY_INT32_DEVICE_CACHE[device] = cached
    return cached


def rfdetr_triton_postproc(
    bboxes: torch.Tensor,
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: "torch.Tensor | float",
    num_classes: int,
    class_mapping: Optional[torch.Tensor],
    inference_size_wh: Tuple[int, int],
    scale_wh: Tuple[float, float],
    orig_size_wh: Tuple[int, int],
    emit_rle: bool = False,
    pack_dense_masks: bool = False,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    "torch.cuda.Event",
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Returns fast-path scratch buffers and completion event.

    ``combined`` is ``(Q, 6)`` int32 where column 4 is fp32 confidence bits.
    ``mask_bin`` is uint8 scratch: either ``(Q, H, W)`` dense bytes or
    ``(Q, H, ceil(W / 8))`` bit-packed bytes when ``pack_dense_masks`` is true.
    ``counter`` stores the number of kept detections from the reference flat
    top-k output. When ``emit_rle`` is true, ``rle_counts`` and ``rle_lengths``
    hold COCO-style uncompressed run-length counts for each surviving detection.
    """

    device = bboxes.device
    num_queries, num_classes_total = logits.shape[1], logits.shape[2]
    mask_h, mask_w = masks.shape[2], masks.shape[3]
    if (
        num_queries != FASTPATH_NUM_QUERIES
        or num_classes_total != FASTPATH_NUM_CLASSES_TOTAL
        or mask_h != FASTPATH_MASK_H
        or mask_w != FASTPATH_MASK_W
    ):
        raise ValueError(
            "RF-DETR Triton fullpost fast path only supports the fixed TRT "
            f"shape (Q={FASTPATH_NUM_QUERIES}, C={FASTPATH_NUM_CLASSES_TOTAL}, "
            f"Mh={FASTPATH_MASK_H}, Mw={FASTPATH_MASK_W}), got "
            f"{(num_queries, num_classes_total, mask_h, mask_w)}."
        )

    logits_2d = logits[0] if logits[0].is_contiguous() else logits[0].contiguous()
    bboxes_2d = bboxes[0] if bboxes[0].is_contiguous() else bboxes[0].contiguous()
    masks_3d = masks[0] if masks[0].is_contiguous() else masks[0].contiguous()

    orig_w, orig_h = orig_size_wh
    combined, mask_bin, counter = _get_scratch_buffers(
        num_queries=num_queries,
        orig_h=orig_h,
        orig_w=orig_w,
        device=device,
        pack_dense_masks=pack_dense_masks and not emit_rle,
    )
    y_indices, y_weights, x_indices, x_weights = _get_resize_tables(
        orig_h=orig_h,
        orig_w=orig_w,
        device=device,
    )
    if emit_rle:
        rle_counts, rle_lengths_scratch = _get_rle_buffers(
            num_queries=num_queries,
            orig_h=orig_h,
            orig_w=orig_w,
            device=device,
        )
    else:
        rle_counts = None
        rle_lengths_scratch = None

    thr_tensor, per_class = _prepare_threshold(threshold, device, num_classes)
    if class_mapping is not None:
        has_remap = True
        cmap = _get_class_mapping_int32(class_mapping, device)
    else:
        has_remap = False
        cmap = _get_empty_int32_on_device(device)

    _ = inference_size_wh
    _ = scale_wh
    dummy_int32 = _get_empty_int32_on_device(device)

    rfdetr_fullpostproc_triton_kernel[(num_queries,)](
        logits_2d,
        bboxes_2d,
        masks_3d,
        thr_tensor,
        cmap,
        y_indices,
        y_weights,
        x_indices,
        x_weights,
        rle_counts if rle_counts is not None else dummy_int32,
        rle_lengths_scratch if rle_lengths_scratch is not None else dummy_int32,
        combined,
        mask_bin,
        counter,
        int(orig_w),
        int(orig_h),
        logits_2d.stride(0),
        bboxes_2d.stride(0),
        masks_3d.stride(0),
        masks_3d.stride(1),
        masks_3d.stride(2),
        (rle_counts.stride(0) if rle_counts is not None else 1),
        (rle_lengths_scratch.stride(0) if rle_lengths_scratch is not None else 1),
        mask_bin.stride(0),
        mask_bin.stride(1),
        mask_bin.stride(2),
        PER_CLASS=1 if per_class else 0,
        HAS_REMAPPING=1 if has_remap else 0,
        EMIT_RLE=1 if emit_rle else 0,
        PACK_DENSE_MASKS=1 if (pack_dense_masks and not emit_rle) else 0,
        NUM_QUERIES=FASTPATH_NUM_QUERIES,
        NUM_CLASSES_TOTAL=FASTPATH_NUM_CLASSES_TOTAL,
        MASK_H=FASTPATH_MASK_H,
        MASK_W=FASTPATH_MASK_W,
        TOPK_PAD=_TOPK_PAD,
        CLASS_BLOCK=_CLASS_BLOCK,
        MASK_TILE_H=_MASK_TILE_H,
        MASK_TILE_W=_MASK_TILE_W,
        RLE_TILE_H=_RLE_TILE_H,
        RLE_TILE_W=_RLE_TILE_W,
        RLE_MERGE_TILE=_RLE_MERGE_TILE,
        num_warps=4,
        num_stages=1,
    )

    done_event = torch.cuda.Event()
    done_event.record(torch.cuda.current_stream(device))
    rle_lengths = rle_lengths_scratch[:, 0] if rle_lengths_scratch is not None else None
    return combined, mask_bin, counter, done_event, rle_counts, rle_lengths
