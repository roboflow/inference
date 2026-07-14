"""Optional exact Triton kernels for persistent tracker instance caches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton  # type: ignore[import-not-found]
    import triton.language as tl  # type: ignore[import-not-found]
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


@dataclass(frozen=True)
class InstanceCacheKernelResult:
    """Device outputs preserving stable new and seen input positions."""

    seen: torch.Tensor
    new_indices: torch.Tensor
    seen_indices: torch.Tensor
    partition_counts: torch.Tensor

    @property
    def new_counts(self) -> torch.Tensor:
        """Return the per-stream new-item counts without launching a kernel."""
        return self.partition_counts[:, 0]

    @property
    def seen_counts(self) -> torch.Tensor:
        """Return the per-stream seen-item counts without launching a kernel."""
        return self.partition_counts[:, 1]


def instance_cache_hash_capacity(cache_size: int) -> int:
    """Return a power-of-two table with load factor no greater than one half."""
    minimum = max(4, cache_size * 2)
    return 1 << (minimum - 1).bit_length()


if triton is not None and tl is not None:

    @triton.jit
    def _hash_slot(key, HASH_MASK: tl.constexpr):
        """Mix one signed 64-bit tracker ID into a power-of-two table."""
        mixed = key * -7046029254386353131
        mixed = mixed ^ (mixed >> 32)
        return mixed & HASH_MASK

    @triton.jit
    def _exact_fifo_hash_kernel(
        flat_ids,
        stream_metadata,
        cache_rows,
        ring_ids,
        ring_valid,
        ring_hash_slots,
        write_indices,
        cache_counts,
        hash_keys,
        hash_values,
        seen,
        new_indices,
        seen_indices,
        partition_counts,
        CACHE_SIZE: tl.constexpr,
        RING_STRIDE: tl.constexpr,
        HASH_CAPACITY: tl.constexpr,
        HASH_MASK: tl.constexpr,
        MAX_INPUTS: tl.constexpr,
    ):
        """Classify one stream sequentially while streams run in parallel."""
        stream = tl.program_id(0)
        input_offset = tl.load(stream_metadata + stream * 2)
        input_count = tl.load(stream_metadata + stream * 2 + 1)
        cache_row = tl.load(cache_rows + stream)
        ring_base = cache_row * RING_STRIDE
        hash_base = cache_row * HASH_CAPACITY
        output_base = stream * MAX_INPUTS
        write_index = tl.load(write_indices + cache_row)
        cache_count = tl.load(cache_counts + cache_row)
        new_count = 0
        seen_count = 0

        for input_position in range(MAX_INPUTS):
            if input_position < input_count:
                key = tl.load(flat_ids + input_offset + input_position)
                slot = _hash_slot(key, HASH_MASK)
                probe_count = 0
                found = False
                searching = True
                while searching:
                    candidate_ring_slot = tl.load(hash_values + hash_base + slot)
                    occupied = candidate_ring_slot >= 0
                    candidate_key = tl.load(
                        hash_keys + hash_base + slot,
                        mask=occupied,
                        other=0,
                    )
                    matches = occupied & (candidate_key == key)
                    found = found | matches
                    probe_count += 1
                    searching = occupied & (~matches) & (probe_count < HASH_CAPACITY)
                    if searching:
                        slot = (slot + 1) & HASH_MASK

                tl.store(seen + input_offset + input_position, found)
                if found:
                    tl.store(
                        seen_indices + output_base + seen_count,
                        input_position,
                    )
                    seen_count += 1
                else:
                    if cache_count == CACHE_SIZE:
                        hole = tl.load(ring_hash_slots + ring_base + write_index)
                        scan = (hole + 1) & HASH_MASK
                        shifting = True
                        while shifting:
                            scan_ring_slot = tl.load(hash_values + hash_base + scan)
                            scan_occupied = scan_ring_slot >= 0
                            if scan_occupied:
                                scan_key = tl.load(hash_keys + hash_base + scan)
                                home = _hash_slot(scan_key, HASH_MASK)
                                scan_distance = (scan - home) & HASH_MASK
                                hole_distance = (hole - home) & HASH_MASK
                                move = hole_distance < scan_distance
                                if move:
                                    tl.store(
                                        hash_keys + hash_base + hole,
                                        scan_key,
                                    )
                                    tl.store(
                                        hash_values + hash_base + hole,
                                        scan_ring_slot,
                                    )
                                    tl.store(
                                        ring_hash_slots + ring_base + scan_ring_slot,
                                        hole,
                                    )
                                    hole = scan
                                scan = (scan + 1) & HASH_MASK
                            else:
                                tl.store(hash_values + hash_base + hole, -1)
                                shifting = False
                    else:
                        cache_count += 1

                    insert_slot = _hash_slot(key, HASH_MASK)
                    insert_occupied = (
                        tl.load(hash_values + hash_base + insert_slot) >= 0
                    )
                    while insert_occupied:
                        insert_slot = (insert_slot + 1) & HASH_MASK
                        insert_occupied = (
                            tl.load(hash_values + hash_base + insert_slot) >= 0
                        )
                    tl.store(hash_keys + hash_base + insert_slot, key)
                    tl.store(
                        hash_values + hash_base + insert_slot,
                        write_index,
                    )
                    tl.store(ring_ids + ring_base + write_index, key)
                    tl.store(ring_valid + ring_base + write_index, True)
                    tl.store(
                        ring_hash_slots + ring_base + write_index,
                        insert_slot,
                    )
                    tl.store(
                        new_indices + output_base + new_count,
                        input_position,
                    )
                    new_count += 1
                    write_index += 1
                    if write_index == CACHE_SIZE:
                        write_index = 0

        tl.store(write_indices + cache_row, write_index)
        tl.store(cache_counts + cache_row, cache_count)
        tl.store(partition_counts + stream * 2, new_count)
        tl.store(partition_counts + stream * 2 + 1, seen_count)

else:
    _exact_fifo_hash_kernel = None


def has_triton_instance_cache() -> bool:
    """Return whether the optional exact FIFO/hash kernel is importable."""
    return _exact_fifo_hash_kernel is not None


def run_triton_instance_cache(
    flat_ids: torch.Tensor,
    stream_metadata: torch.Tensor,
    cache_rows: torch.Tensor,
    ring_ids: torch.Tensor,
    ring_valid: torch.Tensor,
    ring_hash_slots: torch.Tensor,
    write_indices: torch.Tensor,
    cache_counts: torch.Tensor,
    hash_keys: torch.Tensor,
    hash_values: torch.Tensor,
    *,
    cache_size: int,
    max_inputs: int,
) -> Optional[InstanceCacheKernelResult]:
    """Run exact stream-parallel FIFO classification or return fallback."""
    if _exact_fifo_hash_kernel is None or flat_ids.device.type != "cuda":
        return None
    stream_count = stream_metadata.shape[0]
    hash_capacity = instance_cache_hash_capacity(cache_size)
    supported = (
        flat_ids.dtype == torch.long
        and flat_ids.ndim == 1
        and flat_ids.is_contiguous()
        and stream_metadata.dtype == torch.long
        and stream_metadata.shape == (stream_count, 2)
        and stream_metadata.is_contiguous()
        and cache_rows.dtype == torch.long
        and cache_rows.shape == (stream_count,)
        and cache_rows.is_contiguous()
        and ring_ids.dtype == torch.long
        and ring_ids.ndim == 2
        and ring_ids.shape[1] == cache_size + 1
        and ring_ids.is_contiguous()
        and ring_valid.dtype == torch.bool
        and ring_valid.shape == ring_ids.shape
        and ring_valid.is_contiguous()
        and ring_hash_slots.dtype == torch.int32
        and ring_hash_slots.shape == ring_ids.shape
        and ring_hash_slots.is_contiguous()
        and write_indices.dtype == torch.long
        and write_indices.shape == (ring_ids.shape[0],)
        and write_indices.is_contiguous()
        and cache_counts.dtype == torch.int32
        and cache_counts.shape == write_indices.shape
        and cache_counts.is_contiguous()
        and hash_keys.dtype == torch.long
        and hash_keys.shape == (ring_ids.shape[0], hash_capacity)
        and hash_keys.is_contiguous()
        and hash_values.dtype == torch.int32
        and hash_values.shape == hash_keys.shape
        and hash_values.is_contiguous()
        and all(
            tensor.device == flat_ids.device
            for tensor in (
                stream_metadata,
                cache_rows,
                ring_ids,
                ring_valid,
                ring_hash_slots,
                write_indices,
                cache_counts,
                hash_keys,
                hash_values,
            )
        )
    )
    if not supported:
        return None
    kernel_inputs = 1 << (max(1, max_inputs) - 1).bit_length()
    seen = torch.empty(flat_ids.shape[0], dtype=torch.bool, device=flat_ids.device)
    new_indices = torch.empty(
        (stream_count, kernel_inputs),
        dtype=torch.long,
        device=flat_ids.device,
    )
    seen_indices = torch.empty_like(new_indices)
    partition_counts = torch.empty(
        (stream_count, 2),
        dtype=torch.int32,
        device=flat_ids.device,
    )
    if stream_count:
        with torch.cuda.device(flat_ids.device):
            _exact_fifo_hash_kernel[(stream_count,)](
                flat_ids,
                stream_metadata,
                cache_rows,
                ring_ids,
                ring_valid,
                ring_hash_slots,
                write_indices,
                cache_counts,
                hash_keys,
                hash_values,
                seen,
                new_indices,
                seen_indices,
                partition_counts,
                CACHE_SIZE=cache_size,
                RING_STRIDE=cache_size + 1,
                HASH_CAPACITY=hash_capacity,
                HASH_MASK=hash_capacity - 1,
                MAX_INPUTS=kernel_inputs,
                num_warps=1,
            )
    return InstanceCacheKernelResult(
        seen=seen,
        new_indices=new_indices,
        seen_indices=seen_indices,
        partition_counts=partition_counts,
    )
