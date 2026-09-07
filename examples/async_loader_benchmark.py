"""
Benchmark demonstrating dedicated model loader benefits (Issue #2448).

This script simulates the production scenario where multiple workflow executions
need to load models, demonstrating how a dedicated loader executor helps.

BEFORE fix:
- Model loads (3s each) run in the shared 16-worker workflow pool
- Each load blocks a workflow worker
- Load coalescing already works (same manager)

AFTER fix with dedicated loader:
- Model loads run in separate 4-worker loader pool
- Workflow workers still block on future.result() but load I/O is elsewhere
- Load coalescing works across same manager
- Different managers can load same model in parallel (no cross-manager coalescing)

NOTE: This is Phase 1. Workers are NOT fully freed - they still block on
future.result(). True non-blocking requires returning Futures to workflow engine.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List
from unittest.mock import MagicMock


class MockModel:
    """Mock model that simulates slow loading."""

    def __init__(self, model_id: str, api_key: str, **kwargs):
        time.sleep(3.0)  # Simulate 3s model load
        self.model_id = model_id
        self.api_key = api_key
        self.task_type = "object-detection"


class MockModelManager:
    """Mock manager for benchmark."""

    def __init__(self, model_registry):
        self.model_registry = model_registry
        self._models = {}
        self._lock = Lock()

    def add_model(self, model_id: str, api_key: str, **kwargs):
        with self._lock:
            if model_id in self._models:
                return
            model = self.model_registry.get_model()(model_id, api_key)
            self._models[model_id] = model

    def __contains__(self, model_id: str):
        return model_id in self._models


def simulate_workflow_execution_sync(manager, model_id: str, workflow_id: int):
    """Simulate a workflow execution using synchronous model loading (BEFORE fix)."""
    start = time.time()

    # This blocks the thread pool worker for 3 seconds!
    manager.add_model(model_id=model_id, api_key="key")

    # Simulate quick inference
    time.sleep(0.1)

    elapsed = time.time() - start
    return {"workflow_id": workflow_id, "model": model_id, "time": elapsed}


def simulate_workflow_execution_async(loader, manager, model_id: str, workflow_id: int):
    """Simulate a workflow execution using async model loading (AFTER fix)."""
    start = time.time()

    # Submit to dedicated loader pool (NOTE: still blocks this thread via future.result())
    future = loader.add_model_async(
        model_manager=manager,
        model_id=model_id,
        api_key="key",
    )

    if future is not None:
        # Wait for load to complete
        future.result(timeout=10.0)

    # Simulate quick inference
    time.sleep(0.1)

    elapsed = time.time() - start
    return {"workflow_id": workflow_id, "model": model_id, "time": elapsed}


def run_benchmark_sync():
    """Benchmark synchronous (blocking) model loading - BEFORE fix."""
    print("=" * 80)
    print("BEFORE FIX: Synchronous model loading (blocks thread pool)")
    print("=" * 80)

    model_registry = MagicMock()
    model_registry.get_model.return_value = MockModel
    manager = MockModelManager(model_registry)

    # Simulate 8 concurrent workflows, 4 need model A, 4 need model B
    workflows = [
        ("model-a/1", i) for i in range(4)
    ] + [
        ("model-b/1", i) for i in range(4, 8)
    ]

    # Use shared thread pool (simulating workflow executor)
    shared_pool = ThreadPoolExecutor(max_workers=16)

    print(f"Starting {len(workflows)} workflows on 16-worker pool...")
    print("Models need to cold load (3s each)...")
    start = time.time()

    futures = [
        shared_pool.submit(simulate_workflow_execution_sync, manager, model_id, wf_id)
        for model_id, wf_id in workflows
    ]

    results = [f.result() for f in futures]
    total_time = time.time() - start

    shared_pool.shutdown(wait=True)

    # Analysis
    times = [r["time"] for r in results]
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  p50 workflow time: {sorted(times)[len(times)//2]:.2f}s")
    print(f"  p90 workflow time: {sorted(times)[int(len(times)*0.9)]:.2f}s")
    print(f"  Max workflow time: {max(times):.2f}s")
    print(f"\nNote: MockModelManager already deduplicates same-model loads.")
    print(f"The synchronous approach blocks {len(workflows)} workflow workers during loads.")

    return {
        "total_time": total_time,
        "p50": sorted(times)[len(times)//2],
        "p90": sorted(times)[int(len(times)*0.9)],
        "max": max(times),
    }


def run_benchmark_async():
    """Benchmark asynchronous (non-blocking) model loading - AFTER fix."""
    print("\n" + "=" * 80)
    print("AFTER FIX: Asynchronous model loading (dedicated loader pool)")
    print("=" * 80)

    from inference.core.managers.async_loader import AsyncModelLoader
    from inference.core.managers.base import ModelManager

    model_registry = MagicMock()
    model_registry.get_model.return_value = MockModel
    manager = ModelManager(model_registry)
    loader = AsyncModelLoader(max_workers=4)

    # Simulate 8 concurrent workflows, 4 need model A, 4 need model B
    workflows = [
        ("model-a/1", i) for i in range(4)
    ] + [
        ("model-b/1", i) for i in range(4, 8)
    ]

    # Use shared thread pool (simulating workflow executor)
    shared_pool = ThreadPoolExecutor(max_workers=16)

    print(f"Starting {len(workflows)} workflows on 16-worker pool...")
    print("Models need to cold load (3s each), but load in dedicated pool...")
    start = time.time()

    futures = [
        shared_pool.submit(
            simulate_workflow_execution_async, loader, manager, model_id, wf_id
        )
        for model_id, wf_id in workflows
    ]

    results = [f.result() for f in futures]
    total_time = time.time() - start

    shared_pool.shutdown(wait=True)
    loader.shutdown(wait=True)

    # Analysis
    times = [r["time"] for r in results]
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  p50 workflow time: {sorted(times)[len(times)//2]:.2f}s")
    print(f"  p90 workflow time: {sorted(times)[int(len(times)*0.9)]:.2f}s")
    print(f"  Max workflow time: {max(times):.2f}s")
    print(f"\nNote: Dedicated loader pool isolates model loading from workflow execution.")
    print(f"Load coalescing: 4 requests for model-a → 1 actual load (same manager)")
    print(f"Load coalescing: 4 requests for model-b → 1 actual load (same manager)")
    print(f"Workers still block on future.result() - Phase 2 needed for true async.")

    return {
        "total_time": total_time,
        "p50": sorted(times)[len(times)//2],
        "p90": sorted(times)[int(len(times)*0.9)],
        "max": max(times),
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEDICATED MODEL LOADER BENCHMARK (Issue #2448 Phase 1)")
    print("=" * 80)
    print("\nScenario: 8 concurrent workflows, 2 distinct models (3s load each)")
    print("Demonstrates: Isolation of model loading to dedicated executor\n")

    before = run_benchmark_sync()
    after = run_benchmark_async()

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Total time: {before['total_time']:.2f}s → {after['total_time']:.2f}s")
    print(f"p50 latency: {before['p50']:.2f}s → {after['p50']:.2f}s")
    print(f"p90 latency: {before['p90']:.2f}s → {after['p90']:.2f}s")
    print("\nKey improvements:")
    print("✅ Model loading isolated to dedicated executor (4 workers)")
    print("✅ Load coalescing prevents duplicate loads per manager")
    print("✅ Foundation for Phase 2: true async workflow re-scheduling")
    print("\nLimitations (addressed in Phase 2):")
    print("⚠️  Workflow workers still block on future.result()")
    print("⚠️  No cross-manager load coalescing (by design)")
    print("⚠️  No admission control or queue limiting yet")
    print("=" * 80)
