## Idea

At runtime, we want the model manager to read a **stored worst-case memory envelope** per model, runtime, precision, and input profile and decide if this model can be loaded on the GPU without OOM.

We need to build a profiling workflow that would provide the metrics for this admission, given model metadata.

## Admission Rule

```
available_gpu_memory >= new_model_residency + worst_case_incremental_peak + guard_band
```

## Metrics

1. **Model residency memory**
   Memory that must remain on the GPU after the model is loaded and idle. This is usually weights plus runtime-owned persistent buffers and, for TensorRT, engine/context-related memory.

2. **Incremental peak request memory**
   Additional peak memory needed when serving a request under a specific input profile.

3. **Global guard**

   Safety margin, the larger of:

   - 10–20% of total (`idle_after_load + peak_incremental`), or
   - a fixed floor such as 1–2 GiB

## Metadata

- model_id
- runtime: `pytorch`, `onnxruntime`, or `tensorrt`
- device type and GPU SKU
- precision: FP32 / FP16 / BF16 / INT8, etc.
- shape regime:
  - fixed shapes, or
  - dynamic shape bounds
- serving mode:
  - single-shot inference
  - autoregressive with KV cache
- concurrency assumptions:
  - number of simultaneous requests
  - number of execution contexts for TensorRT

## Profiling

#### Configurable data dimensions

- batch size
- sequence length
- image resolution
- number of generated tokens / max decode length
- dynamic-shape buckets
- MoE routing stress cases, if relevant
- concurrency level
- for TensorRT: optimization profile and number of execution contexts

For transformers and autoregressive models, include KV-cache growth in the sweep by testing decode lengths near the maximum you will permit in production.

#### Workflow

##### Phase 1: Static manifest extraction

Before any GPU profiling, extract cheap static metadata:

- parameter count
- dtype
- model architecture family
- nominal input schema
- runtime artifact path
- dynamic shape ranges
- whether KV cache exists
- whether model is dense, sparse, MoE, or retrieval-augmented

This gives you a prior estimate and helps choose what scenarios to benchmark.

##### Phase 2: Runtime-specific profiling harnesses

Build one harness per runtime, but make them emit the same normalized JSON result.

##### Phase 3: Worst-case envelope builder

Aggregate all runs and store:

- peak allocated memory
- peak reserved memory when available
- idle memory after load
- per-shape worst case
- recommended safety margin

## Normalized results schema

```json
{
  "model_id": "llama3-8b-fp16",
  "runtime": "tensorrt",
  "gpu": "NVIDIA L40S",
  "precision": "fp16",
  "artifact_hash": "sha256:...",
  "shape_profile": {
    "batch_size": 4,
    "seq_len": 2048,
    "max_new_tokens": 512
  },
  "concurrency": 1,
  "idle_after_load_bytes": 9432219648,
  "peak_incremental_bytes": 2785017856,
  "peak_total_process_bytes": 12217237504,
  "safety_margin_bytes": 1610612736,
  "recommended_admission_bytes": 13827850240,
  "notes": [
    "includes KV-cache growth to max_new_tokens",
    "measured with one execution context"
  ],
  "timestamp": "2026-04-23T08:00:00Z"
}
```

## Runtimes

#### PyTorch

##### Workflow

For each test case:

1. Start a clean worker process.
2. Initialize CUDA context.
3. Record baseline free memory from NVML or `nvidia-smi`.
4. Load model onto GPU.
5. Synchronize.
6. Record **idle-after-load** memory.
7. Reset peak stats.
8. Run warmup iterations.
9. Reset peak stats again.
10. Run measured iterations with worst-case-shaped inputs.
11. Synchronize.
12. Capture:
    - `torch.cuda.max_memory_allocated`
    - `torch.cuda.max_memory_reserved`
    - end-of-run reserved memory `torch.cuda.memory_reserved()`
13. Destroy process.

The separate process matters because PyTorch’s allocator caches memory, and that can contaminate later runs if you profile many variants in one long-lived worker. PyTorch’s docs note that unused memory managed by the caching allocator can still appear as used, and that `memory_allocated` and `memory_reserved` capture different things.  

##### What to store

- `idle_after_load_bytes`
- `peak_allocated_bytes`
- `peak_reserved_bytes`
- `delta_peak_reserved_bytes = peak_reserved - idle_after_load`
- `shape_signature`
- `warmup_iterations`
- `measured_iterations`

Use `peak_reserved` for admission control if you want to be conservative, because that better reflects allocator behavior.

##### Caveats

- PyTorch uses a **caching allocator**, so **reserved memory** can stay visible in tools like `nvidia-smi` even after tensors are freed
- PyTorch should be the “most conservative” runtime in many cases because of eager execution and allocator behavior.  Guard band toward the high end because allocator reservation and fragmentation can matter more.

##### Config

- profiler support with `profile_memory=True`
- CUDA memory snapshots for deeper debugging.

#### ONNX

##### Config

- Profiling can be enabled through `SessionOptions.enable_profiling = True`, and it emits a JSON trace you can inspect after the run
- Because ONNX Runtime’s trace is mostly execution-focused, you will usually want an external GPU memory observer too, such as NVML sampling or `nvidia-smi`, to capture process-level peaks. The ORT JSON trace is still useful for explaining which operators ran and when.  

##### Workflow

For each test case:

1. New worker process.
2. Create `SessionOptions`.
3. Set `enable_profiling = True`.
4. Create session with `CUDAExecutionProvider`.
5. Run a few warmups.
6. Measure:
   - GPU memory before session creation
   - idle memory after session creation
   - peak process GPU memory during inference
7. Close session and finalize profiling trace.

## What to store

- `idle_after_session_create_bytes`
- `peak_process_gpu_bytes`
- `delta_peak_bytes`
- trace file path
- execution provider and version

#### TensorRT

##### Config

- TensorRT should be profiled in two stages because **build-time** and **run-time** memory are different operational concerns.
- `trtexec` is explicitly intended for benchmarking networks, generating serialized engines, and profiling behavior during engine execution

##### Workflow

###### Stage A: Engine build profile

Measure peak GPU memory during:

- ONNX parse
- engine build
- engine serialization

This is relevant only if your production system builds engines dynamically. If you prebuild engines offline, the model manager does not need this number for admission.

###### Stage B: Engine runtime profile

Measure separately:

- engine deserialization
- execution context creation
- idle-after-load
- request-time incremental peak

For dynamic shapes, profile each optimization profile or shape bucket you will actually admit in production.

##### What to store

- `engine_size_bytes`
- `idle_after_deserialize_bytes`
- `execution_context_bytes`
- `peak_request_bytes`
- `optimization_profile`
- `max_workspace_setting`
- `num_contexts_profiled`

##### Caveats

- Profile **number of execution contexts** explicitly if you use more than one.
- Profile **one context per concurrent request class** if that matches your serving architecture.
- Keep build-time memory out of admission control unless builds happen on the serving GPU.