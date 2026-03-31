# New Model Manager — Context & Requirements

## Problem with Current ModelManager

- Tightly coupled to Roboflow registry
- In-process, thread-locked — GIL contention between models
- No isolation — one model crash affects all
- Cannot be shared across FastAPI workers
- Will NOT be updated in inference 0.X.X (new manager is a separate package)

## Core Goals

Serve as the unified model management layer for:
- Inference server (HTTP)
- Workflows EE
- Inference Pipeline / WebRTC worker
- Async queue-based serving (worker processes listening to a queue)

## Architecture

The model manager exposes a **uniform proxy interface** to all clients. The execution backend is pluggable and selected at startup based on deployment context.

**One `Backend` instance per loaded model.** `ModelManager` holds `Dict[str, Backend]` and routes calls by `model_id`. Each backend is a handle to exactly one model — either a loaded object in the current process or a spawned worker process. This gives independent lifecycle, independent crash recovery, and allows mixing backend types per deployment (e.g. one model in-process, another in a worker).

```
ModelManager (same interface to all clients)
    │
    │   _backends: Dict[str, Backend]
    │       ├── "yolo"  → InProcessBackend   ← IS the loaded model
    │       ├── "sam"   → MultiProcessBackend ← IS the worker process handle
    │       └── ...
    │
    ├── InProcessBackend       ← inference pipeline, CPU-only, constrained hardware
    │     └── direct call / thread-pool executor; no IPC overhead
    │
    ├── MultiProcessBackend    ← HTTP server, GPU machines, full isolation
    │     ├── control plane: ZeroMQ
    │     └── data plane: ZeroMQ signal + SHM bulk payload (see below)
    │
    └── (future) RemoteBackend ← distributed / cross-machine
```

### Why pluggable backends

The HTTP server and Inference Pipeline have fundamentally different requirements:

| | HTTP server | Inference Pipeline |
|---|---|---|
| Consumers | Many concurrent clients | One pipeline, sequential per frame |
| Model lifecycle | Dynamic load/unload | Load once, run continuously |
| Bottleneck | Throughput, concurrency | Per-frame latency |
| IPC overhead | Amortized across requests | Paid every frame — budget is tight |
| Crash isolation | Critical | Less critical |
| GIL benefit | Yes (parallelism) | No (sequential) |

**Worst case for Inference Pipeline**: CPU-only hardware with no shared memory support. Here IPC overhead is a fixed per-frame tax with no GPU to hide it behind — in-process execution is the right choice. Multi-process would add serialization round trips for every model in the pipeline with no benefit.

**Worst case for HTTP server**: many concurrent clients, multiple FastAPI workers, GPU utilisation — in-process would reintroduce all the GIL and isolation problems the new manager is meant to solve.

### MultiProcessBackend — IPC Design

Worker processes are created with `start_method="spawn"`. Fork is not viable: CUDA cannot be used after `fork()`, and macOS has deprecated fork for multi-threaded programs.

**Shared executor for `InProcessBackend`**: a single `ThreadPoolExecutor` is injected at construction and shared across all in-process backends. Async calls dispatch to it via `run_in_executor`; sync calls block the calling thread directly.

**Two planes, two concerns:**

- **Control plane** (lifecycle signals, tiny messages): ZeroMQ — chosen over `multiprocessing.Pipe` / `Queue` for asyncio integration (`zmq.asyncio`) and thread safety.
- **Data plane** (bulk image/tensor data): hybrid approach — ZeroMQ carries a small envelope (`request_id`, `offset`, `size`); the actual payload moves through pre-allocated shared memory (zero-copy). ZeroMQ-only fallback for variable-size outputs or cross-machine cases.

```
caller writes image → SHM buffer
caller sends ZeroMQ msg: {request_id, offset, size}
worker reads SHM, runs inference
worker writes result → SHM buffer
worker sends ZeroMQ msg: {request_id, offset, size}
caller reads result from SHM
```

`DataChannel` is a pluggable abstraction (`ZeroMQDataChannel`, `SharedMemoryDataChannel`) injected into `MultiProcessBackend` at construction — the backend is unaware of which is active.

> Cross-platform requirement (Linux, macOS, Windows) still applies — verify `multiprocessing.shared_memory` on all three before committing SHM. ZeroMQ-only is the safe fallback.

Load-test both before committing to SHM as default.

### HTTP server technology note

If the HTTP server is written in Go or Rust, the Python orchestrator should still be a separate subprocess — not rewritten in Go/Rust. Rationale:
- Model registry access, metadata handling, auth middleware injection are all Python — reimplementing in Go/Rust provides no benefit
- Spawning and managing Python worker processes is natural from Python (`multiprocessing`), awkward from a foreign runtime
- The orchestrator is on the cold path (lifecycle management only) — its language overhead is irrelevant
- The HTTP server talks directly to workers for inference (hot path), only talking to the orchestrator for load/unload/health

## Functional Requirements

### Introspection & Observability
- Load model without running inference
- Manual unload
- Real-time status: what is loading, what is loaded
- Metrics: loading times, usage frequency, per-component breakdown
- Events emitted on state changes (for dashboards / alerting)
- GPU memory awareness — know which models can/cannot be loaded
- Cache introspection: see what's cached, purge items via API

### Model Lifecycle & Caching
- Custom weights (disk and cloud) are first-class citizens
- Explicit, controllable caching behavior
- Encapsulate model-specific behaviors: SAM cache control, TRT/CUDA stream management, auto-loader auth middleware injection
- Thread-safe interface for all model manipulation
- Load/unload must be coordinated — requests held during initialization
- Workers signal health state; manager prevents invalid operations

### Client Interface (Proxy Pattern)
- From the client's perspective it should look like direct model usage
- **Sync mode**: client thread blocks on event lock until result ready (infinite timeout, no cancellation) — simplest and most obvious
- **Async mode**: client schedules multiple model runs concurrently; can cancel early (discard queued inference if client disconnects before execution starts)
- Overwhelm prevention: configurable 429/503 rejection
- Configurable timeouts and memory limits for queues / shared memory

### Performance
- Binary serialization for data (numpy arrays); GPU pointer passing under certain conditions
- Multi-GPU: manager distributes across replicas (round-robin or weighted by GPU usage), transparent to client
- Configurable batching per worker (à la Triton)
- CUDA MPS worth evaluating for many-processes-per-GPU scenarios
- Thin IPC abstraction for Inference Pipeline — measure latency impact before adding overhead

### Error Handling
- Errors and exceptions must be explicitly representable and propagatable across process boundaries
- Workers must signal unhealthy state; manager surfaces this and blocks invalid operations

## Integration Points

- **HTTP server**: FastAPI (may not always be Python in the future)
- **Inference Pipeline / WebRTC**: must not degrade FPS or latency for live streams
- **Workflows EE**: models accessed exclusively through manager

## Load Testing Plan (before full engineering commitment)

Run all scenarios before committing to architecture:

1. **GPU saturation** (MultiProcessBackend): script pushes data with no feed bottleneck — verify N GPUs can be saturated with optimal worker config
2. **IPC throughput**: push data of varying sizes/types through ZeroMQ and SHM — understand real limits of each before choosing
3. **Inference Pipeline — GPU** (MultiProcessBackend): throughput under load with high-resolution footage; measure IPC overhead vs. baseline
4. **Inference Pipeline — CPU / no SHM** (InProcessBackend): same footage, constrained hardware — verify in-process backend maintains FPS and confirm multi-process would have been harmful here
5. **Server under load** (MultiProcessBackend): minimalistic FastAPI endpoint with pre-loaded models — measure latency, stability, throughput vs. old solution; start single GPU, then scale

## Reference Links

- CUDA VMM (zero-copy GPU memory, NCCL transfer): https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
- Triton shared memory protocol (control/data plane separation): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_shared_memory.html
- Triton model management: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html
- DCGM GPU metrics (per-device; per-process only with MIG): https://developer.nvidia.com/dcgm
- CUDA MPS (multi-process GPU saturation): https://docs.nvidia.com/deploy/mps/when-to-use-mps.html

## Current State

### `inference_models/inference_models/model_manager.py`

Stub `ModelManager` — uniform proxy interface. Holds `Dict[str, Backend]`, routes by `model_id`.

```python
def load(self, model_id: str, api_key: str, **kwargs) -> None
def unload(self, model_id: str, api_key: str) -> None
def clear(self) -> None
async def infer_async(self, model_id: str, *args, **kwargs) -> Any
def infer_sync(self, model_id: str, *args, **kwargs) -> Any
def stats(self) -> List[Any]
def __contains__(self, model_id: str) -> bool
def __len__(self) -> int
```

### `inference_models/inference_models/model_manager_utils/backends.py`

```python
class Backend(ABC):
    def unload(self) -> None
    async def infer_async(self, *args, **kwargs) -> Any
    def infer_sync(self, *args, **kwargs) -> Any

class InProcessBackend(Backend):
    def __init__(self, model_id: str, api_key: str, **kwargs)
    # loads model in current process; async dispatches to shared thread-pool executor

class MultiProcessBackend(Backend):
    def __init__(self, model_id: str, api_key: str, **kwargs)
    # spawns worker process (spawn, not fork); forwards inference over ZeroMQ + SHM data channel
```
