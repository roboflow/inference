# Using CUDA Graphs with TensorRT Models

CUDA graphs capture a sequence of GPU operations and replay them as a single unit, eliminating per-call
CPU overhead. For TensorRT models in `inference_models`, this translates to a **7–12% FPS improvement**
on repeated inference with the same input shape.

## Overview

When CUDA graphs are enabled, the first `forward()` call for a given input shape captures the TensorRT
execution into a CUDA graph. Subsequent calls with the same shape replay the captured graph instead of
re-launching individual GPU kernels. Captured graphs are stored in an LRU cache keyed by
`(shape, dtype, device)`.

CUDA graphs work with all TRT model classes that use `infer_from_trt_engine` — including object detection,
instance segmentation, keypoint detection, classification, and semantic segmentation models.

## Prerequisites

- A CUDA-capable GPU
- TensorRT installed (brought in by `trt-*` extras of `inference-models`)
- A TRT model package (`.plan` engine file)

## Quick Start

The simplest way to enable CUDA graphs is through the `USE_CUDA_GRAPHS_FOR_TRT_BACKEND` environment
variable:

```bash
export USE_CUDA_GRAPHS_FOR_TRT_BACKEND=True
```

With this set, all TRT models loaded via `AutoModel.from_pretrained` will automatically create a CUDA
graph cache and use it during inference. No code changes required.

```python
import torch
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    model_id_or_path="rfdetr-nano",
    device=torch.device("cuda:0"),
    backend="trt",
)

# First call captures the CUDA graph for this input shape
results = model.predict(image)

# Subsequent calls replay the captured graph — faster
results = model.predict(image)
```

## Manual Cache Control

For more control over cache behavior, create a `TRTCudaGraphCache` explicitly and pass it
to `AutoModel.from_pretrained`:

```python
import torch
from inference_models import AutoModel
from inference_models.models.common.trt import TRTCudaGraphCache

cache = TRTCudaGraphCache(capacity=16)

model = AutoModel.from_pretrained(
    model_id_or_path="rfdetr-nano",
    device=torch.device("cuda:0"),
    backend="trt",
    trt_cuda_graph_cache=cache,
)
```

The `capacity` parameter controls how many distinct input shapes can be cached simultaneously.
When the cache is full, the least recently used graph is evicted automatically.

### Inspecting the Cache

You can query the cache at any time to see what's been captured:

```python
# Check how many graphs are currently cached
print(cache.get_current_size())  # e.g. 3

# List all cached keys — each key is a (shape, dtype, device) tuple
for key in cache.list_keys():
    shape, dtype, device = key
    print(f"  shape={shape}, dtype={dtype}, device={device}")

# Check if a specific shape is cached
key = ((1, 3, 384, 384), torch.float16, torch.device("cuda:0"))
if key in cache:
    print("Graph is cached for this shape")
```

### Removing Specific Entries

Use `safe_remove()` to evict a single cached graph by its key. This releases the associated
CUDA graph, execution context, and GPU buffers immediately. If the key doesn't exist, the
call is a no-op:

```python
key = ((1, 3, 384, 384), torch.float16, torch.device("cuda:0"))
cache.safe_remove(key)
```

### Purging the Cache

Use `purge()` to evict multiple entries at once. When called without arguments, it clears the
entire cache. You can also pass `n_oldest` to evict only the N least recently used entries:

```python
# Evict the 4 oldest (least recently used) entries
cache.purge(n_oldest=4)

# Clear the entire cache
cache.purge()
```

`purge()` is more efficient than calling `safe_remove()` in a loop because it batches the
GPU memory cleanup — `torch.cuda.empty_cache()` is called once at the end rather than after
each individual eviction.

!!! tip "When to purge manually"
    Manual purging is useful when you know the workload is about to change — for example,
    switching from processing video at one resolution to another. Purging stale entries
    frees VRAM for the new shapes before they're captured.

### Sharing a Cache Across Models

Please **do not share single instance of `TRTCudaGraphCache`** to multiple models - as cache object is bound to 
specific model instance.

### Choosing Cache Capacity

Each cached graph holds its own TensorRT execution context and GPU memory buffers. A reasonable
default is **8–16 entries**. Consider:

- **Fixed input shape** (e.g. always 1×3×640×640): `capacity=1` is sufficient.
- **Variable batch sizes** (e.g. batch 1–16): set capacity to the number of distinct batch sizes
  you expect, or quantize to powers of two and set `capacity=4–5`.
- **Memory-constrained environments**: lower the capacity to reduce VRAM usage.

## Disabling CUDA Graphs Per Call

Even with a cache configured, you can bypass CUDA graphs for individual forward passes using the
`disable_cuda_graphs` flag:

```python
pre_processed, meta = model.pre_process(image)

# Standard path — uses CUDA graphs if cache is configured
output = model.forward(pre_processed)

# Bypass CUDA graphs for this specific call
output = model.forward(pre_processed, disable_cuda_graphs=True)
```

This is useful for debugging, benchmarking, or when you need to compare graph vs. non-graph outputs.


## How It Works

The lifecycle of a CUDA graph in `inference_models`:

1. **Cache miss** — `infer_from_trt_engine` detects that no cached graph exists for the current
   `(shape, dtype, device)` key. It creates a dedicated TensorRT execution context, allocates
   input/output buffers, runs a warmup pass, then captures the execution into a `torch.cuda.CUDAGraph`.
   The graph and its associated state are stored in the cache.

2. **Cache hit** — On subsequent calls with the same key, the cached graph's input buffer is updated
   via `copy_()`, the graph is replayed, and output buffers are cloned and returned. No TensorRT
   context setup or kernel launches happen on the CPU side.

3. **Eviction** — When the cache exceeds its capacity, the least recently used entry is evicted.
   The associated CUDA graph, execution context, and GPU buffers are released, and
   `torch.cuda.empty_cache()` is called to return memory to the CUDA driver.


## Important Considerations

### VRAM Usage

Each cache entry consumes GPU memory for input buffers, output buffers, and the TensorRT execution
context's internal workspace. With large models or high cache capacities, this can be significant.
Monitor VRAM usage when tuning `capacity`.

### Thread Safety

One may manage cache entries and eviction from separate thread compared to the one running forward-pass.
The cache state is synchronized with thread lock.

### Dynamic Batch Sizes

CUDA graphs are shape-specific — a graph captured for batch size 4 cannot be replayed for batch size 8.
If your application uses variable batch sizes, each distinct size will trigger a separate graph capture.
The LRU cache handles this transparently, but be aware that frequent shape changes will cause cache
churn and recapture overhead.

!!! tip "Quantize batch sizes for better cache utilization"

    If you control the batching logic, round batch sizes up to the nearest power of two
    (1, 2, 4, 8, 16). This reduces the number of distinct shapes and keeps the cache small.

### When CUDA Graphs Won't Help

- **Cold start / single inference**: The first call for each shape pays the capture cost, which is
  slower than a normal forward pass. CUDA graphs only pay off on subsequent replays.
- **Highly variable input shapes**: If every call has a unique shape, graphs are captured but
  never replayed.
- **CPU-bound pipelines**: If your bottleneck is preprocessing or postprocessing, the GPU-side
  speedup from graph replay won't be visible end-to-end.
