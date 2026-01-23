"""
GCP Serverless Logging Memory Measurement Utilities.

This module provides tiered memory measurement:
- Basic mode: Just model footprint (cheap, always available)
- Detailed mode: Full system snapshot (expensive, opt-in)
"""

from typing import Optional, Tuple

from inference.core.gcp_logging.events import MemorySnapshot


def get_gpu_allocated() -> int:
    """
    Get current GPU memory allocated by tensors.

    This is cheap - just returns an internal counter, no GPU sync required.
    Returns 0 if CUDA is not available.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
    except ImportError:
        pass
    return 0


def measure_memory_before_load(detailed: bool = False) -> Tuple[int, Optional[dict]]:
    """
    Measure memory state before loading a model.

    Args:
        detailed: If True, capture full memory snapshot (expensive).
                  If False, just capture allocated memory (cheap).

    Returns:
        Tuple of (gpu_allocated, detailed_state_dict or None)
    """
    allocated = get_gpu_allocated()

    if not detailed:
        return allocated, None

    return allocated, _measure_detailed_state()


def measure_memory_after_load(
    allocated_before: int,
    detailed_before: Optional[dict],
    detailed: bool = False,
) -> Tuple[int, Optional[MemorySnapshot]]:
    """
    Measure memory state after loading a model and compute footprint.

    Args:
        allocated_before: GPU allocated memory before loading
        detailed_before: Detailed state dict from before (or None)
        detailed: If True, capture full memory snapshot

    Returns:
        Tuple of (model_footprint_bytes, MemorySnapshot or None)
    """
    allocated_after = get_gpu_allocated()
    footprint = allocated_after - allocated_before

    if not detailed:
        return footprint, None

    detailed_after = _measure_detailed_state()

    snapshot = MemorySnapshot(
        gpu_allocated_before=allocated_before,
        gpu_allocated_after=allocated_after,
        gpu_reserved_after=detailed_after.get("gpu_reserved"),
        gpu_free=detailed_after.get("gpu_free"),
        gpu_total=detailed_after.get("gpu_total"),
        process_rss_bytes=detailed_after.get("process_rss_bytes"),
        system_available_bytes=detailed_after.get("system_available_bytes"),
    )

    return footprint, snapshot


def measure_memory_for_eviction(detailed: bool = False) -> Optional[MemorySnapshot]:
    """
    Measure memory state for eviction event.

    Only captures gpu_free before eviction. The after state is measured
    separately after the model is removed.

    Args:
        detailed: If True, capture memory state

    Returns:
        MemorySnapshot with gpu_free_before, or None if not detailed
    """
    if not detailed:
        return None

    state = _measure_detailed_state()
    return MemorySnapshot(
        gpu_free=state.get("gpu_free"),
    )


def _measure_detailed_state() -> dict:
    """
    Capture full memory state (expensive operation).

    This includes GPU memory (requires sync) and system memory (syscalls).
    Only call when GCP_LOGGING_DETAILED_MEMORY is enabled.
    """
    state = {}

    # GPU memory (expensive - requires GPU synchronization)
    try:
        import torch

        if torch.cuda.is_available():
            gpu_free, gpu_total = torch.cuda.mem_get_info()
            state.update(
                {
                    "gpu_allocated": torch.cuda.memory_allocated(),
                    "gpu_reserved": torch.cuda.memory_reserved(),
                    "gpu_free": gpu_free,
                    "gpu_total": gpu_total,
                }
            )
    except ImportError:
        pass

    # System memory (moderate cost - syscalls)
    try:
        import psutil

        state["process_rss_bytes"] = psutil.Process().memory_info().rss
        state["system_available_bytes"] = psutil.virtual_memory().available
    except ImportError:
        pass

    return state
