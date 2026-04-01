"""Detect CUDA memory leaks in the inference server model lifecycle.

Exercises the exact production code path (ModelManager + WithFixedSizeCache +
RoboflowModelRegistry + InferenceModels*Adapter classes) and measures device-level
VRAM at each step to identify whether GPU memory is properly reclaimed after
model eviction.

Five phases:
  1. Load/Evict       — load N models, evict all, check VRAM returns to baseline
  2. Inference         — run many inferences on one model, check for per-inference growth
  3. Load/Infer/Evict  — repeated load→infer→evict cycles, check cumulative growth
  4. Production Sim    — WithFixedSizeCache round-robin, check VRAM with constant model count
  5. Embedding Cache   — SAM3 embed_image with unique images, check VRAM growth per embedding

Example:
    python profile_cuda_memory_leaks.py --api-key YOUR_KEY
    python profile_cuda_memory_leaks.py --api-key YOUR_KEY --model-set small --phases 1
    python profile_cuda_memory_leaks.py --api-key YOUR_KEY --output leak_report.png --verbose
"""

import argparse
import gc
import os
import sys
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, List

# Environment setup — must come before inference imports to avoid Redis/metrics side effects
os.environ.setdefault("USE_INFERENCE_MODELS", "True")
os.environ.setdefault("METRICS_ENABLED", "False")
os.environ.setdefault("DISABLE_INFERENCE_CACHE", "True")
os.environ.setdefault("MODELS_CACHE_AUTH_ENABLED", "False")

import numpy as np
import torch

from inference.core.entities.requests.inference import (
    InferenceRequestImage,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.sam2 import Sam2EmbeddingRequest
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.models.utils import ROBOFLOW_MODEL_TYPES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MB = 1024**2

MODEL_SETS = {
    "small": [
        "yolov8n-640",
        "rfdetr-nano",
    ],
    "standard": [
        # Object Detection
        "yolov8n-640",
        "yolov8s-640",
        "yolov11n-640",
        "rfdetr-base",
        "rfdetr-nano",
        # Instance Segmentation
        "yolov8n-seg-640",
        # Keypoint Detection
        "yolov8n-pose-640",
        # Foundation models with embedding caches
        "sam3/sam3_interactive",
    ],
    "full": [
        # Object Detection
        "yolov8n-640",
        "yolov8s-640",
        "yolov11n-640",
        "rfdetr-base",
        "rfdetr-nano",
        # Instance Segmentation
        "yolov8n-seg-640",
        "rfdetr-seg-nano",
        # Keypoint Detection
        "yolov8n-pose-640",
        # Core / Foundation models
        "florence-2-base",
        "sam2/hiera_tiny",
        "sam3/sam3_interactive",
    ],
}

# Models that support standard ObjectDetectionInferenceRequest
# Models that have embedding caches storing GPU tensors
EMBEDDING_CACHE_MODELS = {
    "sam3/sam3_interactive",
    "sam2/hiera_tiny",
}

# SAM3 interactive model ID for Phase 5
SAM3_MODEL_ID = "sam3/sam3_interactive"

DETECTION_MODELS = {
    "yolov8n-640",
    "yolov8s-640",
    "yolov11n-640",
    "rfdetr-base",
    "rfdetr-nano",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VRAMMeasurement:
    label: str
    timestamp: float
    vram_bytes: int


@dataclass
class PhaseResult:
    name: str
    passed: bool
    baseline_mb: float
    final_mb: float
    leaked_mb: float
    details: str
    measurements: List[VRAMMeasurement] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VRAM Tracker
# ---------------------------------------------------------------------------
class VRAMTracker:
    def __init__(self, device: torch.device):
        self.device = device
        self.log: List[VRAMMeasurement] = []

    def snapshot(self, label: str = "") -> int:
        """Full VRAM snapshot: GC + sync + empty cache + measure."""
        gc.collect()
        gc.collect()
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)
        used = self._measure()
        m = VRAMMeasurement(label=label, timestamp=time.monotonic(), vram_bytes=used)
        self.log.append(m)
        return used

    def snapshot_light(self, label: str = "") -> int:
        """Light VRAM snapshot: sync + measure only (no GC or cache flush)."""
        torch.cuda.synchronize(self.device)
        used = self._measure()
        m = VRAMMeasurement(label=label, timestamp=time.monotonic(), vram_bytes=used)
        self.log.append(m)
        return used

    def _measure(self) -> int:
        free, total = torch.cuda.mem_get_info(self.device)
        return total - free

    @staticmethod
    def fmt(vram_bytes: int) -> str:
        return f"{vram_bytes / MB:.1f} MB"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_base_model_manager() -> ModelManager:
    """Create a raw ModelManager (no cache) matching the production registry."""
    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    return ModelManager(model_registry=registry)


def create_cached_model_manager(max_size: int) -> WithFixedSizeCache:
    """Create ModelManager + WithFixedSizeCache matching production setup."""
    registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    base = ModelManager(model_registry=registry)
    return WithFixedSizeCache(base, max_size=max_size)


def create_test_image() -> np.ndarray:
    """640x640 random BGR uint8 image."""
    return (np.random.rand(640, 640, 3) * 255).astype(np.uint8)


def make_detection_request(
    image: np.ndarray, api_key: str, model_id: str
) -> ObjectDetectionInferenceRequest:
    return ObjectDetectionInferenceRequest(
        image=InferenceRequestImage(type="numpy_object", value=image),
        model_id=model_id,
        api_key=api_key,
        confidence=0.5,
    )


def try_infer(manager, model_id: str, request, verbose: bool = False) -> bool:
    """Attempt inference; return True on success, False on failure."""
    try:
        manager.infer_from_request_sync(model_id, request)
        return True
    except Exception as e:
        if verbose:
            print(f"    [skip inference for {model_id}: {type(e).__name__}: {e}]")
        return False


def count_adapter_instances() -> Dict[str, int]:
    """Count live adapter instances via gc. Expensive — use sparingly."""
    gc.collect()
    adapter_names = {
        "InferenceModelsObjectDetectionAdapter",
        "InferenceModelsInstanceSegmentationAdapter",
        "InferenceModelsKeyPointsDetectionAdapter",
        "InferenceModelsClassificationAdapter",
        "InferenceModelsSemanticSegmentationAdapter",
        "InferenceModelsSAMAdapter",
        "InferenceModelsSAM2Adapter",
        "InferenceModelsClipAdapter",
        "InferenceModelsFlorence2Adapter",
        "InferenceModelsPaligemmaAdapter",
    }
    counts: Dict[str, int] = {}
    for obj in gc.get_objects():
        name = type(obj).__name__
        if name in adapter_names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def print_header(text: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {text}")
    print("=" * width)


def print_step(text: str) -> None:
    print(f"  {text}")


# ---------------------------------------------------------------------------
# Phase 1: Load / Evict
# ---------------------------------------------------------------------------
def run_phase1(
    models: List[str],
    api_key: str,
    tracker: VRAMTracker,
    threshold_mb: float,
    verbose: bool,
) -> PhaseResult:
    print_header("Phase 1: Load / Evict (basic leak detection)")
    measurements: List[VRAMMeasurement] = []

    manager = create_base_model_manager()

    baseline = tracker.snapshot("p1_baseline")
    measurements.append(tracker.log[-1])
    print_step(f"Baseline VRAM: {tracker.fmt(baseline)}")

    # Load all models
    model_refs = {}
    for model_id in models:
        print_step(f"Loading {model_id}...")
        try:
            manager.add_model(model_id, api_key)
        except Exception as e:
            print_step(f"  FAILED to load {model_id}: {e}")
            continue
        # Grab a weakref to the underlying model object
        try:
            model_obj = manager[model_id]
            model_refs[model_id] = weakref.ref(model_obj)
            del model_obj
        except Exception:
            pass
        vram = tracker.snapshot(f"p1_loaded_{model_id}")
        measurements.append(tracker.log[-1])
        if verbose:
            print_step(f"  VRAM after load: {tracker.fmt(vram)}")

    peak = tracker.snapshot("p1_peak")
    measurements.append(tracker.log[-1])
    print_step(f"Peak VRAM (all models loaded): {tracker.fmt(peak)}")
    print_step(f"Models in manager: {len(manager)}")

    # Evict all models
    loaded_ids = list(manager.keys())
    for model_id in loaded_ids:
        print_step(f"Removing {model_id}...")
        manager.remove(model_id)
        vram = tracker.snapshot(f"p1_removed_{model_id}")
        measurements.append(tracker.log[-1])

        # Check weakref
        ref = model_refs.get(model_id)
        alive = ref is not None and ref() is not None
        if verbose:
            status = "ALIVE (leaked!)" if alive else "collected"
            print_step(f"  VRAM: {tracker.fmt(vram)} | Python object: {status}")

    # Final cleanup
    del manager
    final = tracker.snapshot("p1_final")
    measurements.append(tracker.log[-1])

    # Check for surviving objects
    survivors = {
        mid: ref
        for mid, ref in model_refs.items()
        if ref is not None and ref() is not None
    }
    adapter_counts = count_adapter_instances()

    leaked_mb = (final - baseline) / MB
    passed = leaked_mb <= threshold_mb

    details_parts = [
        f"baseline={tracker.fmt(baseline)}, peak={tracker.fmt(peak)}, final={tracker.fmt(final)}, delta={leaked_mb:.1f} MB",
    ]
    if survivors:
        details_parts.append(
            f"WARNING: {len(survivors)} model(s) NOT garbage collected: {list(survivors.keys())}"
        )
    if adapter_counts:
        details_parts.append(f"Surviving adapter instances: {adapter_counts}")

    print_step(f"Final VRAM: {tracker.fmt(final)} (delta from baseline: {leaked_mb:.1f} MB)")
    if survivors:
        print_step(
            f"  WARNING: {len(survivors)} model object(s) still alive after removal!"
        )
    if adapter_counts:
        print_step(f"  Surviving adapters: {adapter_counts}")

    return PhaseResult(
        name="Phase 1 (Load/Evict)",
        passed=passed,
        baseline_mb=baseline / MB,
        final_mb=final / MB,
        leaked_mb=leaked_mb,
        details="; ".join(details_parts),
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Phase 2: Inference leak
# ---------------------------------------------------------------------------
def run_phase2(
    models: List[str],
    api_key: str,
    tracker: VRAMTracker,
    threshold_mb: float,
    num_inferences: int,
    verbose: bool,
) -> PhaseResult:
    print_header("Phase 2: Inference leak (per-inference VRAM growth)")
    measurements: List[VRAMMeasurement] = []

    # Pick the first detection model available
    det_models = [m for m in models if m in DETECTION_MODELS]
    if not det_models:
        print_step("No detection models in list — skipping Phase 2")
        baseline = tracker.snapshot("p2_skip")
        return PhaseResult(
            name="Phase 2 (Inference)",
            passed=True,
            baseline_mb=baseline / MB,
            final_mb=baseline / MB,
            leaked_mb=0.0,
            details="Skipped: no detection models in model list",
        )

    model_id = det_models[0]
    image = create_test_image()
    request = make_detection_request(image, api_key, model_id)

    manager = create_base_model_manager()
    print_step(f"Loading {model_id} for inference test...")
    manager.add_model(model_id, api_key)

    # Warmup
    print_step("Running warmup inference...")
    try_infer(manager, model_id, request, verbose=verbose)

    baseline = tracker.snapshot("p2_baseline")
    measurements.append(tracker.log[-1])
    print_step(f"Post-warmup baseline: {tracker.fmt(baseline)}")

    # Run inferences
    checkpoint_interval = max(1, num_inferences // 10)
    for i in range(1, num_inferences + 1):
        try_infer(manager, model_id, request)
        if i % checkpoint_interval == 0 or i == num_inferences:
            vram = tracker.snapshot_light(f"p2_infer_{i}")
            measurements.append(tracker.log[-1])
            if verbose:
                delta = (vram - baseline) / MB
                print_step(f"  After {i}/{num_inferences} inferences: {tracker.fmt(vram)} (delta: {delta:.1f} MB)")

    final = tracker.snapshot("p2_final")
    measurements.append(tracker.log[-1])

    # Cleanup
    manager.remove(model_id)
    del manager
    tracker.snapshot("p2_cleanup")

    leaked_mb = (final - baseline) / MB
    passed = leaked_mb <= threshold_mb

    print_step(
        f"After {num_inferences} inferences: {tracker.fmt(final)} (delta: {leaked_mb:.1f} MB)"
    )

    return PhaseResult(
        name="Phase 2 (Inference)",
        passed=passed,
        baseline_mb=baseline / MB,
        final_mb=final / MB,
        leaked_mb=leaked_mb,
        details=f"{num_inferences} inferences on {model_id}, delta={leaked_mb:.1f} MB",
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Phase 3: Load / Infer / Evict cycles
# ---------------------------------------------------------------------------
def run_phase3(
    models: List[str],
    api_key: str,
    tracker: VRAMTracker,
    threshold_mb: float,
    num_cycles: int,
    num_inferences_per_cycle: int,
    verbose: bool,
) -> PhaseResult:
    print_header("Phase 3: Load / Infer / Evict cycles (cumulative leak)")
    measurements: List[VRAMMeasurement] = []

    image = create_test_image()

    # Pick a model — prefer detection for inference, but load-only works too
    test_model = models[0]
    can_infer = test_model in DETECTION_MODELS
    request = make_detection_request(image, api_key, test_model) if can_infer else None

    baseline = tracker.snapshot("p3_baseline")
    measurements.append(tracker.log[-1])
    print_step(f"Baseline VRAM: {tracker.fmt(baseline)}")
    print_step(
        f"Running {num_cycles} load/evict cycles with {test_model}"
        + (f" ({num_inferences_per_cycle} inferences each)" if can_infer else " (load-only)")
    )

    cycle_vrams = []
    for cycle in range(num_cycles):
        manager = create_base_model_manager()
        try:
            manager.add_model(test_model, api_key)
        except Exception as e:
            print_step(f"  Cycle {cycle + 1}: FAILED to load: {e}")
            del manager
            continue

        if can_infer:
            for _ in range(num_inferences_per_cycle):
                try_infer(manager, test_model, request)

        manager.remove(test_model)
        del manager

        vram = tracker.snapshot(f"p3_cycle_{cycle + 1}")
        measurements.append(tracker.log[-1])
        cycle_vrams.append(vram)

        delta = (vram - baseline) / MB
        if verbose:
            print_step(f"  Cycle {cycle + 1}/{num_cycles}: {tracker.fmt(vram)} (delta: {delta:.1f} MB)")

    if len(cycle_vrams) < 2:
        return PhaseResult(
            name="Phase 3 (Cycle)",
            passed=True,
            baseline_mb=baseline / MB,
            final_mb=baseline / MB,
            leaked_mb=0.0,
            details="Insufficient cycles completed",
            measurements=measurements,
        )

    # Compute growth: simple linear regression slope
    x = np.arange(len(cycle_vrams), dtype=np.float64)
    y = np.array(cycle_vrams, dtype=np.float64) / MB
    slope = float(np.polyfit(x, y, 1)[0])  # MB per cycle

    final_delta = (cycle_vrams[-1] - baseline) / MB
    passed = final_delta <= threshold_mb

    print_step(f"VRAM trend: {slope:.1f} MB/cycle over {len(cycle_vrams)} cycles")
    print_step(f"Total delta from baseline: {final_delta:.1f} MB")

    return PhaseResult(
        name="Phase 3 (Cycle)",
        passed=passed,
        baseline_mb=baseline / MB,
        final_mb=cycle_vrams[-1] / MB,
        leaked_mb=final_delta,
        details=f"slope={slope:.1f} MB/cycle over {len(cycle_vrams)} cycles, total_delta={final_delta:.1f} MB",
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Phase 4: Production simulation
# ---------------------------------------------------------------------------
def run_phase4(
    models: List[str],
    api_key: str,
    tracker: VRAMTracker,
    threshold_mb: float,
    cache_size: int,
    num_rounds: int,
    verbose: bool,
) -> PhaseResult:
    print_header("Phase 4: Production simulation (WithFixedSizeCache)")
    measurements: List[VRAMMeasurement] = []

    if len(models) <= cache_size:
        print_step(
            f"Need more models ({len(models)}) than cache size ({cache_size}) to trigger eviction. "
            f"Increase model list or decrease --cache-size."
        )
        baseline = tracker.snapshot("p4_skip")
        return PhaseResult(
            name="Phase 4 (Production Sim)",
            passed=True,
            baseline_mb=baseline / MB,
            final_mb=baseline / MB,
            leaked_mb=0.0,
            details=f"Skipped: need more models ({len(models)}) than cache_size ({cache_size})",
        )

    image = create_test_image()

    manager = create_cached_model_manager(max_size=cache_size)

    baseline = tracker.snapshot("p4_baseline")
    measurements.append(tracker.log[-1])
    print_step(f"Baseline VRAM: {tracker.fmt(baseline)}")
    print_step(
        f"Cache size: {cache_size}, models: {len(models)}, rounds: {num_rounds}"
    )

    # Track VRAM at the same model position across rounds to detect drift.
    # Key insight: different models have different VRAM footprints, so we must
    # compare the same model at the same cache position across rounds.
    # vram_by_round[round_num][model_position] = vram_bytes
    vram_by_round: Dict[int, Dict[int, int]] = {}
    step = 0

    for round_num in range(num_rounds):
        vram_by_round[round_num] = {}
        for pos, model_id in enumerate(models):
            step += 1
            try:
                manager.add_model(model_id, api_key)
            except Exception as e:
                print_step(f"  Step {step}: FAILED to load {model_id}: {e}")
                continue

            # Run a few inferences if possible
            if model_id in DETECTION_MODELS:
                req = make_detection_request(image, api_key, model_id)
                for _ in range(5):
                    try_infer(manager, model_id, req)

            vram = tracker.snapshot(f"p4_r{round_num + 1}_{model_id}")
            measurements.append(tracker.log[-1])
            vram_by_round[round_num][pos] = vram

            loaded = len(manager)
            if verbose:
                print_step(
                    f"  R{round_num + 1} step {step}: loaded {model_id} "
                    f"({loaded}/{cache_size} models) | VRAM: {tracker.fmt(vram)}"
                )

    # Cleanup
    manager.clear()
    del manager
    post_cleanup = tracker.snapshot("p4_post_cleanup")
    measurements.append(tracker.log[-1])
    cleanup_delta = (post_cleanup - baseline) / MB

    # Compare VRAM at equivalent positions across rounds.
    # For each model position that appears in all rounds, compute
    # the drift from round 1 to the last round.
    drifts_mb = []
    embedding_cache_drifts_mb = []
    other_drifts_mb = []
    if num_rounds >= 2:
        positions_in_all = set(vram_by_round[0].keys())
        for r in range(1, num_rounds):
            positions_in_all &= set(vram_by_round[r].keys())
        for pos in sorted(positions_in_all):
            r1_vram = vram_by_round[0][pos]
            last_vram = vram_by_round[num_rounds - 1][pos]
            drift = (last_vram - r1_vram) / MB
            drifts_mb.append(drift)
            model_name = models[pos] if pos < len(models) else f"pos{pos}"
            if model_name in EMBEDDING_CACHE_MODELS:
                embedding_cache_drifts_mb.append(drift)
            else:
                other_drifts_mb.append(drift)
            if verbose:
                tag = " [has embedding cache]" if model_name in EMBEDDING_CACHE_MODELS else ""
                print_step(
                    f"  Drift for {model_name}{tag}: R1={tracker.fmt(r1_vram)} → "
                    f"R{num_rounds}={tracker.fmt(last_vram)} ({drift:+.1f} MB)"
                )

    max_other_drift = max(other_drifts_mb) if other_drifts_mb else 0.0
    max_embed_drift = max(embedding_cache_drifts_mb) if embedding_cache_drifts_mb else 0.0

    # Use non-embedding-cache model drift for pass/fail (Phase 5 tests embedding cache separately)
    passed = max_other_drift <= threshold_mb and cleanup_delta <= threshold_mb

    print_step(f"Per-position VRAM drift (R1 → R{num_rounds}):")
    if other_drifts_mb:
        print_step(f"  Non-embedding-cache models: avg={sum(other_drifts_mb)/len(other_drifts_mb):.1f} MB, max={max_other_drift:.1f} MB")
    if embedding_cache_drifts_mb:
        print_step(f"  Embedding-cache models (sam3/sam2): avg={sum(embedding_cache_drifts_mb)/len(embedding_cache_drifts_mb):.1f} MB, max={max_embed_drift:.1f} MB (tested in Phase 5)")
    print_step(f"VRAM after full cleanup: {tracker.fmt(post_cleanup)} (delta from baseline: {cleanup_delta:.1f} MB)")

    return PhaseResult(
        name="Phase 4 (Production Sim)",
        passed=passed,
        baseline_mb=baseline / MB,
        final_mb=post_cleanup / MB,
        leaked_mb=max(max_other_drift, cleanup_delta),
        details=(
            f"other_max_drift={max_other_drift:.1f} MB, "
            f"embed_cache_max_drift={max_embed_drift:.1f} MB, "
            f"post_cleanup_delta={cleanup_delta:.1f} MB"
        ),
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Phase 5: Embedding cache VRAM growth (SAM3)
# ---------------------------------------------------------------------------
def run_phase5(
    api_key: str,
    tracker: VRAMTracker,
    threshold_mb: float,
    num_embeddings: int,
    embedding_cache_size: int,
    verbose: bool,
) -> PhaseResult:
    print_header(
        f"Phase 5: SAM3 embedding cache (cache_size={embedding_cache_size}, "
        f"num_images={num_embeddings})"
    )
    measurements: List[VRAMMeasurement] = []

    # Override the SAM3 embedding cache size for this test
    os.environ["SAM3_MAX_EMBEDDING_CACHE_SIZE"] = str(embedding_cache_size)

    manager = create_base_model_manager()

    print_step(f"Loading {SAM3_MODEL_ID}...")
    try:
        manager.add_model(SAM3_MODEL_ID, api_key)
    except Exception as e:
        print_step(f"FAILED to load {SAM3_MODEL_ID}: {e}")
        return PhaseResult(
            name="Phase 5 (Embedding Cache)",
            passed=True,
            baseline_mb=0,
            final_mb=0,
            leaked_mb=0,
            details=f"Skipped: failed to load {SAM3_MODEL_ID}: {e}",
        )

    # Get the underlying model to call embed_image directly
    model = manager[SAM3_MODEL_ID]

    # Warmup with one embedding
    warmup_image = InferenceRequestImage(
        type="numpy_object", value=create_test_image()
    )
    warmup_request = Sam2EmbeddingRequest(
        image=warmup_image,
        image_id="warmup_0",
        api_key=api_key,
    )
    model.infer_from_request(warmup_request)

    baseline = tracker.snapshot("p5_baseline")
    measurements.append(tracker.log[-1])
    print_step(f"Post-warmup baseline (1 cached embedding): {tracker.fmt(baseline)}")

    # Send N unique images through embed_image, each with a unique image_id.
    # With cache_size=1, each new image should evict the previous embedding.
    # If VRAM grows, the evicted GPU tensors are not being freed.
    for i in range(1, num_embeddings + 1):
        unique_image = InferenceRequestImage(
            type="numpy_object",
            value=(np.random.rand(640, 640, 3) * 255).astype(np.uint8),
        )
        embed_request = Sam2EmbeddingRequest(
            image=unique_image,
            image_id=f"leak_test_{i}",
            api_key=api_key,
        )
        model.infer_from_request(embed_request)

        if i % max(1, num_embeddings // 10) == 0 or i == num_embeddings:
            vram = tracker.snapshot_light(f"p5_embed_{i}")
            measurements.append(tracker.log[-1])
            delta = (vram - baseline) / MB
            if verbose:
                print_step(
                    f"  After {i}/{num_embeddings} embeddings: "
                    f"{tracker.fmt(vram)} (delta: {delta:+.1f} MB)"
                )

    # Final measurement with full GC
    final = tracker.snapshot("p5_final")
    measurements.append(tracker.log[-1])

    # Cleanup
    manager.remove(SAM3_MODEL_ID)
    del manager
    post_cleanup = tracker.snapshot("p5_post_cleanup")
    measurements.append(tracker.log[-1])

    leaked_mb = (final - baseline) / MB
    cleanup_delta = (post_cleanup - tracker.log[0].vram_bytes) / MB
    passed = leaked_mb <= threshold_mb

    print_step(
        f"After {num_embeddings} unique embeddings (cache_size={embedding_cache_size}): "
        f"{tracker.fmt(final)} (delta: {leaked_mb:.1f} MB)"
    )
    print_step(f"Post-cleanup VRAM delta from initial: {cleanup_delta:.1f} MB")

    return PhaseResult(
        name="Phase 5 (Embedding Cache)",
        passed=passed,
        baseline_mb=baseline / MB,
        final_mb=final / MB,
        leaked_mb=leaked_mb,
        details=(
            f"{num_embeddings} unique images, cache_size={embedding_cache_size}, "
            f"delta={leaked_mb:.1f} MB"
        ),
        measurements=measurements,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary(results: List[PhaseResult]) -> None:
    print_header("CUDA Memory Leak Detection Results")
    any_failed = False
    for r in results:
        status = "PASS" if r.passed else "LEAK DETECTED"
        icon = " " if r.passed else "!"
        print(f"  {icon} {r.name:<30s} {status:<15s} [{r.details}]")
        if not r.passed:
            any_failed = True
    print()
    if any_failed:
        print("  RESULT: Memory leak(s) detected. See details above.")
    else:
        print("  RESULT: No memory leaks detected.")
    print()


def generate_chart(results: List[PhaseResult], output_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart generation")
        return

    num_phases = len(results)
    fig, axes = plt.subplots(1, num_phases, figsize=(6 * num_phases, 5), squeeze=False)
    fig.suptitle("CUDA Memory Leak Detection", fontsize=14)

    for idx, result in enumerate(results):
        ax = axes[0][idx]
        if not result.measurements:
            ax.set_title(f"{result.name}\n(no data)")
            continue

        vrams = [m.vram_bytes / MB for m in result.measurements]
        labels = [m.label for m in result.measurements]
        steps = range(len(vrams))

        ax.plot(steps, vrams, marker=".", color="steelblue", linewidth=1.5)
        ax.axhline(
            y=result.baseline_mb,
            color="green",
            linestyle="--",
            alpha=0.7,
            label="baseline",
        )

        if not result.passed:
            ax.axhline(
                y=result.baseline_mb + result.leaked_mb,
                color="red",
                linestyle="--",
                alpha=0.7,
                label="final",
            )

        ax.set_title(
            f"{result.name}\n{'PASS' if result.passed else 'LEAK: ' + f'{result.leaked_mb:.0f} MB'}",
            color="green" if result.passed else "red",
        )
        ax.set_ylabel("VRAM (MB)")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)

        # Rotate and show select labels
        if len(labels) <= 20:
            ax.set_xticks(list(steps))
            ax.set_xticklabels(
                [l.split("_", 2)[-1] if "_" in l else l for l in labels],
                rotation=45,
                ha="right",
                fontsize=6,
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect CUDA memory leaks in the inference server model lifecycle"
    )
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model aliases to test (overrides --model-set)",
    )
    parser.add_argument(
        "--model-set",
        type=str,
        choices=["small", "standard", "full"],
        default="standard",
        help="Preset model group (default: standard)",
    )
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=100,
        help="Number of inferences for Phase 2 (default: 100)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=5,
        help="Load/evict cycles for Phase 3 (default: 5)",
    )
    parser.add_argument(
        "--num-inferences-per-cycle",
        type=int,
        default=10,
        help="Inferences per cycle in Phase 3 (default: 10)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=3,
        help="Full model rotations for Phase 4 (default: 3)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=3,
        help="WithFixedSizeCache max_size for Phase 4 (default: 3)",
    )
    parser.add_argument(
        "--threshold-mb",
        type=float,
        default=50.0,
        help="MB delta to flag as leak (default: 50)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="CUDA device (default: cuda:0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for chart PNG",
    )
    parser.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Which phases to run (default: 1 2 3 4 5)",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=50,
        help="Number of unique images for Phase 5 SAM3 embedding cache test (default: 50)",
    )
    parser.add_argument(
        "--embedding-cache-size",
        type=int,
        default=1,
        help="SAM3 embedding cache size for Phase 5 — mirrors production SAM3_MAX_EMBEDDING_CACHE_SIZE (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step measurements",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device)
    models = args.models if args.models else MODEL_SETS[args.model_set]

    print(f"Device: {device}")
    print(f"Models: {models}")
    print(f"Phases: {args.phases}")
    print(f"Leak threshold: {args.threshold_mb} MB")

    # Initialize CUDA context
    torch.cuda.init()
    torch.cuda.set_device(device)

    tracker = VRAMTracker(device)
    initial_vram = tracker.snapshot("initial")
    print(f"Initial VRAM (after CUDA init): {tracker.fmt(initial_vram)}")

    results: List[PhaseResult] = []

    if "1" in args.phases:
        results.append(
            run_phase1(models, args.api_key, tracker, args.threshold_mb, args.verbose)
        )

    if "2" in args.phases:
        results.append(
            run_phase2(
                models,
                args.api_key,
                tracker,
                args.threshold_mb,
                args.num_inferences,
                args.verbose,
            )
        )

    if "3" in args.phases:
        results.append(
            run_phase3(
                models,
                args.api_key,
                tracker,
                args.threshold_mb,
                args.num_cycles,
                args.num_inferences_per_cycle,
                args.verbose,
            )
        )

    if "4" in args.phases:
        results.append(
            run_phase4(
                models,
                args.api_key,
                tracker,
                args.threshold_mb,
                args.cache_size,
                args.num_rounds,
                args.verbose,
            )
        )

    if "5" in args.phases:
        results.append(
            run_phase5(
                args.api_key,
                tracker,
                args.threshold_mb,
                args.num_embeddings,
                args.embedding_cache_size,
                args.verbose,
            )
        )

    print_summary(results)

    if args.output:
        generate_chart(results, args.output)

    any_leaked = any(not r.passed for r in results)
    sys.exit(1 if any_leaked else 0)


if __name__ == "__main__":
    main()
