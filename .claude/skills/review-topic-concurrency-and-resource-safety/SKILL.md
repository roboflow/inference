---
name: review-topic-concurrency-and-resource-safety
description: Load when a diff adds threads/async/background work (ThreadPoolExecutor, threading.Thread, asyncio.Lock, concurrent.futures.Future) or touches torch.jit.load/script, torchscript_state_global_lock, acquire_with_timeout, inference/core/managers/** (add_model/remove_model), on-disk caches/artifacts (inference/core/cache/**, model_artifacts.py, MODEL_CACHE_DIR, FileLock, AtomicPath), or stream resources (inference_pipeline.py terminate, webrtc/modal worker, VideoSource).
---

# Review topic: Concurrency, resource & artifact/cache safety

## When this applies
Load when the diff shows ANY of these content signals (paths are hints, not the trigger):
- New/resized background execution: `ThreadPoolExecutor`, `threading.Thread`, `asyncio.Lock/Task`, `concurrent.futures.Future`, `weakref.finalize`, daemon threads, watchdogs, pingback loops.
- Shared, process-global or non-thread-safe runtime state: `torch.jit.load/script`, CUDA/TensorRT context, autocast contexts, ONNX sessions, shared model caches, per-model locks.
- Model lifecycle / managers: `inference/core/managers/**`, `WithFixedSizeCache`, model eviction/pinning, `add_model`/`remove_model`.
- On-disk caches or artifacts: `inference/core/cache/**`, `model_artifacts.py`, `MODEL_CACHE_DIR`, `get_cache_dir`, cache-key/slug/hash logic, `FileLock`, `AtomicPath`, `tempfile`, `os.replace/os.rename`, `shutil.rmtree`.
- Long-running stream/server resources: `inference/core/interfaces/stream/inference_pipeline.py`, webrtc/modal worker, `VideoSource` lifecycle, connection/session pools.

## Review checklist

### BLOCK
- **Global runtime mutation is serialized.** Any new call into `torch.jit.load/script`, TensorRT engine build, or ORT session creation on a shared object holds the right lock. Model loaders thread `torchscript_state_global_lock` through; prefer `torchscript_guard` over an ad-hoc lock. (Rule 1) (concurrent load corrupting the process-global TorchScript registry — #2373)
- **Cache/artifact writes are atomic.** New writes into `MODEL_CACHE_DIR` go through `save_*_in_cache` → `dump_*_atomic`, never a bare `open(...,'w')` a concurrent reader can see half-written. (Rule 5) (truncated-artifact read by another worker)
- **Cache keys are valid, unique, and path-safe.** New key/slug logic stays within root, fits OS path limits, and hashes to avoid collisions — no user/model-id string used raw as a directory name. (Rule 6)
- **Every `future.result()` on a hot/dispatch path has a timeout.** Unbounded `.result()` hangs the pipeline forever if GPU work stalls. (Rule 4) (#2489, #2486)

### FLAG
- **Locks acquire with a timeout and always release.** (Rule 2) (`asyncio.Lock` bound to a since-closed event loop — #1750)
- **Every executor/thread has a shutdown path AND a GC fallback.** (Rule 3) (leaked `ThreadPoolExecutor` on dropped owner — #2491)
- **Cross-worker deletion/replacement is locked and idempotent.** (Rule 7)
- **No leaked contexts/resources.** (Rule 9) (autocast context entered at model init, never exited — #2363)
- **Cleanup degrades gracefully.** (Rule 10) (Redis outage crashing the request path — #2387)

### NIT
- **No unbounded growth.** New caches/queues/deques have a max size + eviction. (Rule 8)

### Not blocking
- Do NOT demand a lock/timeout/finalizer on code that is genuinely single-owner and single-threaded (e.g. per-request local state that never escapes the request, a temp dir created and `rmtree`'d in the same function).
- Do NOT block on a missing atomic-write when the target is a throwaway path outside `MODEL_CACHE_DIR` that no other worker reads.
- Do NOT require `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT` on a `.result()` that is not on a hot/dispatch path (e.g. a one-shot startup call with an explicit local timeout).
- Prefer reusing the existing safe helper over hand-rolling; a bespoke reimplementation is a FLAG (drift risk), not a BLOCK, unless it is demonstrably wrong.

## What to check (canonical rules)
1. **Serialize global runtime mutation.** Model loaders receive and use `torchscript_state_global_lock` (threaded in via `ModelManager.__init__` in `managers/base.py`). CUDA/autocast contexts must not leak across models.
2. **Locks: timeout + guaranteed release.** New `Lock()`/`RLock()` acquisitions use `acquire_with_timeout` (`managers/base.py`) or `with lock:`; no lock held across blocking I/O or a `future.result()`; no `asyncio.Lock` bound to a transient/closed event loop.
3. **Executors/threads: explicit shutdown + GC fallback.** A new `ThreadPoolExecutor` is shut down on both success and exception, plus a `weakref.finalize(self, executor.shutdown, wait=False)` so a dropped owner still reaps it (see `_get_response_executor` / `shutdown_pipeline` in `inference_models_adapters.py`). A new `Thread` is `.join()`ed in `terminate()` and gated by a stop flag (`inference_pipeline.py`).
4. **Bounded `future.result()`.** On hot/dispatch paths use `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT` and the shared `resolve_futures`/`contains_future` helpers (`execution_engine/v1/executor/utils.py`), not a bespoke recursive resolver.
5. **Atomic cache/artifact writes.** Writers use `save_*_in_cache` → `dump_bytes_atomic`/`dump_json_atomic`/`dump_text_lines_atomic`, which build on `AtomicPath` (temp file in same dir + `os.replace`, cleanup on error) in `utils/file_system.py`, gated by `ATOMIC_CACHE_WRITES_ENABLED`.
6. **Valid, unique, path-safe cache keys.** New key logic stays within root (`cache_path_is_within_root`), fits OS limits (`path_fits_os_limits`), and slug+hashes (`slugify_model_id_to_cache_key`, `get_model_id_cache_path` in `cache/model_artifacts.py`). No raw model-id as a directory name.
7. **Locked, idempotent cross-worker delete/replace.** Cache clears use `FileLock` with a timeout, re-check existence after acquiring, tolerate `FileNotFoundError` from a racing worker, and never crash on lock-acquire failure (`clear_cache`, `cache/model_artifacts.py`).
8. **No unbounded growth.** New caches/queues/deques have a max size + eviction respecting pinned models; collectors don't accumulate unresolved futures/contexts.
9. **No leaked contexts/resources.** Context managers entered in `__init__`/`from_pretrained` (autocast, ORT sessions, temp dirs) are exited on every path; per-model temp dirs live only as long as needed.
10. **Graceful cleanup.** `rmtree`/Redis/lock failures are caught and logged, not propagated to the request (`CacheUnavailableError` fallback in `roboflow_api.py`).

## Key files & Reference PRs
- `inference/core/utils/torchscript_guard.py` — `torchscript_guard` RLock-guarded `torch.jit.script` override (`_torch_jit_script_lock`); canonical "serialize a process-global runtime mutation". (#2373)
- `inference/core/managers/base.py` — `torchscript_state_global_lock`, per-model `_models_state_locks`, `acquire_with_timeout`; how locks are threaded into model loaders and always time out.
- `inference/core/cache/model_artifacts.py` — `save_*_in_cache` atomic writes, `get_model_id_cache_path`/`slugify_model_id_to_cache_key`/`cache_path_is_within_root`/`path_fits_os_limits` path-safe keys, `FileLock`-guarded race-tolerant `clear_cache`.
- `inference/core/utils/file_system.py` — `AtomicPath` (temp-in-same-dir + `os.replace`, cleanup on error) and the `dump_*_atomic` writers all cache writers should use.
- `inference/core/models/inference_models_adapters.py` — `_get_response_executor` with `weakref.finalize` GC-reap + explicit `shutdown_pipeline`; the correct background-executor lifecycle. (#2491)
- `inference/core/interfaces/stream/inference_pipeline.py` — stop-flag (`_stop`) + `.join()` thread lifecycle in `terminate()`, and bounded resolution via `resolve_futures` with `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT`. (#2489)
- `inference/core/workflows/execution_engine/v1/compiler/cache.py` — `BasicWorkflowsCache`: bounded, lock-guarded in-memory cache with `deque(maxlen=...)` eviction and a stable md5 key.
- Other regressions: leaked autocast context at init (#2363); Redis-outage crash vs `CacheUnavailableError` fallback (#2387); watchdog/timeout misfires on long-running modal/webrtc workers (#1875, #1769, #1750).
