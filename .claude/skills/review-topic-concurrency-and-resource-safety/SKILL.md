---
name: review-topic-concurrency-and-resource-safety
description: Load when a PR adds threads/async/background work (ThreadPoolExecutor, Thread, asyncio, Future), touches model caches/locks/model lifecycle, cache-dir/temp-dir/artifact writes, or long-running stream/server resources (InferencePipeline, webrtc worker, model managers). Purpose — review that shared runtime state is synchronized, background work is bounded and cleaned up, and cache/artifact writes are atomic, correctly keyed, and cross-worker safe.
---

# Review topic: Concurrency, resource & artifact/cache safety

## When this applies
Load this skill when the diff exhibits ANY of these content signals (paths are hints, not the trigger):
- Introduces or resizes background execution: `ThreadPoolExecutor`, `threading.Thread`, `asyncio.Lock/Task`, `concurrent.futures.Future`, daemon threads, watchdogs, pingback loops.
- Touches shared, process-global or non-thread-safe runtime state: TorchScript (`torch.jit.load/script`), CUDA/TensorRT context, autocast contexts, ONNX sessions, shared model caches, per-model locks.
- Touches model lifecycle / managers: `inference/core/managers/**`, `WithFixedSizeCache`, model eviction/pinning, `add_model`/`remove_model`.
- Touches caches or artifacts on disk: `inference/core/cache/**`, `model_artifacts.py`, `MODEL_CACHE_DIR`, `get_cache_dir`, cache-key/slug/hash logic, `FileLock`, `tempfile`, `os.replace/os.rename`, `shutil.rmtree`, atomic-write helpers.
- Touches long-running stream/server resources: `inference/core/interfaces/stream/inference_pipeline.py`, webrtc/modal worker, VideoSource lifecycle, connection/session pools.

## What to protect
- **Shared non-thread-safe runtime state stays consistent.** `torch.jit.load/script` mutate a process-global TorchScript registry; concurrent loads corrupt it → wrong/failed model loads under multi-worker or concurrent `add_model`. Same for CUDA/autocast contexts leaking across models.
- **Background work is bounded and reaped.** Every executor/thread/future has an owner that shuts it down (including on GC and on error paths) and a timeout on `.result()`. Otherwise: thread/executor leaks, hung requests, and unresolved futures pinning GPU memory.
- **Cache/artifact writes are atomic and correctly keyed.** A reader in another worker must never observe a partially-written file; a cache key must uniquely and validly identify the artifact. Otherwise: torn/corrupt weights, stale or wrong-device artifacts, path-length/collision failures, cross-worker partial visibility.
- **Cleanup and cache misses degrade gracefully.** rmtree/lock-acquire/ephemeral-cache (Redis) failures must not crash the request path or leave locks held.

## What to check
1. **Global runtime mutation is serialized.** Any new call into `torch.jit.load/script`, autocast/`bf16_context`, TensorRT engine build, or ORT session creation on a shared object acquires the right lock. Model loaders must thread `torchscript_state_global_lock` through (see `ModelManager.__init__`, base.py:66). Prefer the existing `torchscript_guard` / manager lock over a new ad-hoc lock.
2. **Locks are acquired with a timeout and always released.** New `Lock()`/`RLock()` acquisitions use the `acquire_with_timeout` pattern (base.py:670) or `with lock:`; no lock held across blocking I/O or a `future.result()`; no `asyncio.Lock` bound to a since-closed event loop (regression class of #1750).
3. **Every executor/thread has a shutdown path AND a GC fallback.** New `ThreadPoolExecutor` is closed in `shutdown_pipeline`/`terminate` on both success and exception, and has a `weakref.finalize(self, executor.shutdown, wait=False)` so a dropped owner still reaps it (pattern from #2491). New `Thread` is `.join()`ed in `terminate()` and gated by a stop flag (inference_pipeline.py:890-918).
4. **Every `future.result()` on a hot/dispatch path has a timeout.** Unbounded `.result()` blocks the pipeline forever if GPU work hangs — use `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT` and the shared `resolve_futures`/`contains_future` helpers, not a bespoke recursive resolver (#2489, #2486).
5. **Cache/artifact writes are atomic.** New writes into `MODEL_CACHE_DIR` go through `save_*_in_cache` / `dump_*_atomic` (temp file in same dir + `os.replace`), never a bare `open(...,'w')` that a concurrent reader can see half-written (`_AtomicWriter` in file_system.py:34-70; gated by `ATOMIC_CACHE_WRITES_ENABLED`).
6. **Cache keys are valid, unique, and path-safe.** New cache-key/slug logic stays within root (`cache_path_is_within_root`), fits OS path limits (`path_fits_os_limits`), and hashes to avoid collisions (`slugify_model_id_to_cache_key`, model_artifacts.py:271). No user/model-id string used raw as a directory name.
7. **Cross-worker deletion/replacement is locked and idempotent.** Cache clears use `FileLock` with a timeout, re-check existence after acquiring, tolerate `FileNotFoundError` from a racing worker, and never crash on lock-acquire failure (`clear_cache`, model_artifacts.py:179-238).
8. **No unbounded growth.** New caches/queues/deques have a max size + eviction (respecting pinned models); background collectors don't accumulate unresolved futures/contexts.
9. **No leaked contexts/resources.** Context managers entered in `__init__`/`from_pretrained` (autocast, sessions, temp dirs) are exited on every path (the SAM3 `bf16_context` leak, #2363); temp dirs created for a model outlive only as long as needed.
10. **Cleanup degrades gracefully.** rmtree/redis/lock failures are caught and logged, not propagated to the request (`CacheUnavailableError` fallback, #2387).

## Common failure modes
- **Concurrent TorchScript load corrupts a process-global** — no lock around `torch.jit.load/script` (fixed by the global lock in #2373).
- **Leaked `ThreadPoolExecutor`** when the owning block/adapter is dropped without `shutdown` — fixed with `weakref.finalize` in #2491.
- **Unbounded `future.result()`** hanging the stream pipeline — bounded timeout + shared resolver added in #2489; duplicated bespoke future-resolvers consolidated in #2486.
- **Leaked CUDA bf16 autocast context** entered at model init and never exited, polluting later inference — #2363.
- **Ephemeral (Redis) cache outage crashes the request path** instead of falling back — #2387.
- **Non-atomic cache write** → another worker reads a truncated artifact (guard: `dump_*_atomic` + `ATOMIC_CACHE_WRITES_ENABLED`).
- **Raw model-id as directory name** → path-length/collision/traversal failures (guard: slug+hash cache keys).
- **`asyncio.Lock` bound to a dead event loop** in the webrtc worker — #1750.
- **Watchdog/timeout misfires** on long-running modal/webrtc workers — #1875, #1769.

## Example implementations (point here)
- `inference/core/utils/torchscript_guard.py` — RLock-guarded `torch.jit.script` override; the canonical "serialize a process-global runtime mutation" pattern. (contract established by #2373)
- `inference/core/managers/base.py` — `torchscript_state_global_lock`, per-model `_models_state_locks`, and `acquire_with_timeout` (base.py:670); how locks are threaded into model loaders and always time-out.
- `inference/core/cache/model_artifacts.py` — atomic cache writes (`save_*_in_cache` → `dump_*_atomic`), path-safe/collision-resistant cache keys (`get_model_id_cache_path`, `slugify_model_id_to_cache_key`), and `FileLock`-guarded, race-tolerant `clear_cache`.
- `inference/core/utils/file_system.py` — `_AtomicWriter` (temp-in-same-dir + `os.replace`, cleanup on error) — the atomic-write primitive all cache writers should use.
- `inference/core/models/inference_models_adapters.py` — executor with `weakref.finalize` GC-reap + explicit `shutdown_pipeline`; the correct background-executor lifecycle. (#2491)
- `inference/core/interfaces/stream/inference_pipeline.py` — stop-flag + `.join()` thread lifecycle in `terminate()`, and bounded future resolution via `resolve_futures` with `WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT`. (#2489)
- `inference/core/workflows/execution_engine/v1/compiler/cache.py` — bounded, lock-guarded in-memory cache with deque eviction and a stable md5 hash key.

## Severity guidance
- **Critical** — data-corrupting or hang-inducing concurrency: missing lock around a shared non-thread-safe runtime (TorchScript/CUDA/ORT), non-atomic write to a shared cache/artifact read by other workers, or an unbounded `.result()` on the pipeline dispatch path. Also: a cache key that can collide or escape root.
- **High** — leaked executor/thread/context with no shutdown or GC fallback, lock held across blocking I/O, `asyncio.Lock` bound to a transient loop, or cleanup failure that can crash the request path.
- **Medium** — missing acquire timeout, unbounded-but-slow-growing cache/queue, non-graceful degradation on cache miss, or a bespoke reimplementation of an existing safe helper (future resolver, atomic writer) that risks drift.
