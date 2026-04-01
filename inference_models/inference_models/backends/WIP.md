# Backends — Analysis & WIP Notes

## base.py — Clean, minimal ABC

The interface is tight: `class_names`, `unload`, `infer_async`, `infer_sync`. No `load` method — loading is implicit in `__init__`. This is intentional ("Loading happens in `__init__`, blocks until ready"), and it's the right call for the current design where one Backend = one model.

**Missing from the interface** (per requirements doc):
- No `stats()` / health / status introspection. The MODEL_MANAGER_CONTEXT doc lists "Real-time status: what is loading, what is loaded", "Metrics: loading times, usage frequency", "Workers signal health state". Currently the Backend ABC has no way to report health or metrics — this would need to live here or in ModelManager, but having an `is_healthy` property on Backend itself would let SubprocessBackend check if its child process is alive.
- No `max_batch_size` property. Batching is the single highest-impact optimization (8x throughput at batch=16 for YOLOv8n). When adaptive batching lands, the batcher needs to know each backend's limit.

## direct.py (DirectBackend) — Stub only

Currently all `...` (pass). Per the architecture doc, this is the right backend for InferencePipeline / CPU-only / constrained hardware where IPC round-trip per frame is unacceptable.

- The context doc says "a single `ThreadPoolExecutor` is injected at construction and shared across all in-process backends." The current stub's `__init__` takes `(model_id, api_key, **kwargs)` — no executor parameter. When implemented, `__init__` will need an `executor` parameter, or the ModelManager will need to inject one. See "Shared Executor" section below.

## subproc.py (SubprocessBackend)

This is the production-path backend.

### 1. `torch.cuda.empty_cache()` on every request (worker lines 136-139)

BASELINE_BENCH_CONTEXT.md documents that removing this was worth **+20 RPS** — the only micro-optimization in subproc.py that moved the needle. Yet it's still in the code. The worker calls `empty_cache()` in the `finally` block of every single inference call. This forces CUDA's caching allocator to return memory to the driver, which is expensive and unnecessary for steady-state operation where the same model processes similarly-sized inputs repeatedly.

**Recommendation**: Remove it entirely, or make it periodic (every N requests) / on OOM-retry only.

### 2. SHM transport forces GPU→CPU→SHM→CPU→GPU round-trip

In `shm_serializer.py` (lines 93-95), CUDA tensors are moved to CPU (`.cpu()`) before writing to shared memory. On the worker side, they're reconstructed as CPU tensors via `torch.from_numpy`. So the actual path for a CUDA tensor through SHM is:

```
GPU → CPU (cudaMemcpy D2H) → SHM (memcpy) → CPU (memcpy) → [model.infer does .cuda()] → GPU
```

This is correct behavior for the SHM path — SHM can't hold GPU memory. The `cuda_ipc` transport exists precisely to avoid this. The `auto` transport selection (lines 191-196) checks `torch.cuda.is_available()`, then `_has_cuda_input` gates again per-call. The double check is belt-and-suspenders — fine.

### 3. CUDA IPC transport — per-request handle creation

`_ipc_pickle` (lines 37-48) calls `ForkingPickler` which triggers `_share_cuda_()` internally — `cudaIpcGetMemHandle` + `cudaIpcGetEventHandle` on **every single request**. The deep research doc (Section 3) documents this costs 1-5us for the handle, but the real cost is on the receiver side: `cudaIpcOpenMemHandle` at 50-200us for first open per allocation.

The "THE KEY OPTIMIZATION" (pre-allocated CUDA buffer pool with fixed IPC handles) would eliminate all of this. Current hot path: pickle → create IPC handle → send → unpickle → open IPC handle → inference. Optimal hot path: copy to pre-allocated buffer → send 1-byte slot index → inference. That's the Tier 3 optimization — high effort, highest impact.

### 4. Single ZMQ PAIR socket = serialized inference

`_socket_lock` (threading.Lock) and `_async_lock` (asyncio.Lock) mean only one inference can be in-flight per backend at a time. This is by design (one model, one worker process), but it means **no pipelining**: you can't send request N+1's data while request N is being processed.

Triton achieves GPU overlap by running multiple model instances on separate CUDA streams. Here, you'd need multiple `SubprocessBackend` instances for the same model to get overlap, which the ModelManager could orchestrate.

### 5. JSON header encoding on hot path

Every request/response pair encodes/decodes a JSON header. BASELINE_BENCH_CONTEXT.md tested eliminating this and found "no measurable improvement", so this is a non-issue in practice. The JSON is tiny (~100 bytes) and `json.dumps`/`json.loads` on small dicts is <10us.

### 6. No batching awareness

Neither the Backend ABC nor SubprocessBackend has any concept of batching. Dynamic batching is the single highest-impact optimization (batch=16 → 8x throughput). Currently, batching would need to happen at a layer above (ModelManager or HTTP handler), collecting multiple `infer_async` calls into a single batched call. The worker's `model.infer()` already supports batch inputs (per the benchmark script), so the gap is purely in the collection/dispatch layer.

### 7. Worker crash detection is reactive

If the worker process dies, the only detection is a `zmq.Again` timeout on the next `recv_multipart`. There's no background health-check thread or process sentinel. The `_recv_with_timeout` loop (lines 315-321) will spin on 1-second timeouts until `_shutting_down` is set — but nothing sets it when the worker crashes. The caller will eventually get stuck.

**Recommendation**: Monitor `self._process.is_alive()` either lazily (check before send) or via a background sentinel thread.

### 8. Result SHM uses same size as input SHM

Line 224: `self._result_shm = SharedMemory(create=True, size=input_shm_size)` — 100MB default for both. For YOLO detection results (~2KB per image), this is wildly over-provisioned. The research doc suggests moving small results to CPU in the worker and using regular pickle (saves 100-300us per request by avoiding `cudaIpcOpenMemHandle`). Result SHM could be much smaller, or results could just go through ZMQ directly.

## Priority-ordered issues

| # | Issue | Impact | Effort |
|---|-------|--------|--------|
| 1 | No batching layer | **8x throughput** potential | Medium (new BatchCollector above Backend) |
| 2 | `torch.cuda.empty_cache()` every request | +20 RPS measured | 1-line delete |
| 3 | Per-request CUDA IPC handle creation | ~50-200us/req receiver-side | High (pre-allocated buffer pool) |
| 4 | No worker crash detection | Caller hangs on dead worker | Low (check `is_alive()` before send) |
| 5 | No health/stats on Backend ABC | Can't introspect or observe | Low (add property to ABC) |
| 6 | DirectBackend needs executor injection | Will break when implemented | Low (add parameter now) |
| 7 | Single in-flight request per backend | No GPU pipeline overlap | Medium (multi-instance in ModelManager) |

## Shared Executor — Why and How

The context doc specifies: "a single `ThreadPoolExecutor` is injected at construction and shared across all in-process backends."

### The problem it solves

`DirectBackend.infer_async()` needs to run blocking inference off the event loop. The naive approach is each backend creating its own `ThreadPoolExecutor`:

```python
class DirectBackend(Backend):
    def __init__(self, ...):
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def infer_async(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.infer_sync, *args, **kwargs)
```

This breaks down with multiple models. If you have 10 loaded DirectBackends, each with `max_workers=2`, that's 20 OS threads. Under load, these threads contend for:
- **The GIL** — only one thread runs Python at a time. 20 threads fighting for the GIL means 19 are always blocked on the GIL acquire, wasting context-switch overhead.
- **GPU compute** — if all 10 models push CUDA work concurrently, you get uncontrolled GPU contention. Unlike SubprocessBackend (which has process isolation and can use MPS for GPU partitioning), in-process threads all share the same CUDA context and default stream.
- **CPU cores** — thread oversubscription causes unnecessary scheduling overhead.

### The solution

A single `ThreadPoolExecutor` owned by `ModelManager`, injected into every `DirectBackend`:

```python
class ModelManager:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def load(self, model_id: str, api_key: str, backend: str = "in_process", **kwargs):
        if backend == "in_process":
            self._backends[model_id] = DirectBackend(
                model_id, api_key, executor=self._executor, **kwargs,
            )
```

```python
class DirectBackend(Backend):
    def __init__(self, model_id: str, api_key: str, *, executor: ThreadPoolExecutor, **kwargs):
        self._executor = executor  # shared, not owned
        ...

    async def infer_async(self, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.infer_sync, *args, **kwargs)

    def unload(self) -> None:
        # Does NOT shut down the executor — it's shared
        ...
```

### What this gives you

- **Controlled concurrency** — `max_workers=4` means at most 4 models run inference simultaneously, regardless of how many are loaded. The ModelManager tunes this based on deployment: CPU-only might use `max_workers=1` (sequential, no GIL contention), GPU machine might use `max_workers=2-4` (overlap H2D transfers with compute on different CUDA streams).
- **Implicit queuing** — when all executor threads are busy, new `infer_async` calls queue as suspended coroutines (via `run_in_executor`'s internal queue). No explicit batching needed at this level — the executor IS the admission controller.
- **Clean lifecycle** — the executor lives as long as the ModelManager. Individual backends come and go (load/unload) without executor churn.

### Why this doesn't apply to SubprocessBackend

SubprocessBackend already has process-level isolation — each worker is a separate OS process with its own GIL and (potentially) its own CUDA context. The executor pattern would add nothing: the async path already uses `run_in_executor` only for the blocking `recv_multipart`, not for inference itself. The worker process is the concurrency unit, not a thread.

### Interaction with batching

When adaptive batching is added above the Backend layer (a `BatchCollector` in ModelManager), the shared executor becomes even more important. The batcher collects N requests, then submits one batched `infer_sync(stacked_tensor)` to the executor. With a shared executor at `max_workers=1`, you get pure sequential batched inference — maximum GPU efficiency, zero GIL contention. The executor thread count becomes the knob for "how many concurrent batched forwards can run."
