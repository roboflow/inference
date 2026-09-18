# RF-DETR preprocessor memory comparison on T4 — 2026-09-18

## Outcome

`triton-universal-v1` completed both memory scenarios with no fallback. Its GPU
peak grows with input resolution: **252 MiB at 640x480 and 318 MiB at 3840x2160**.
Pillow-SIMD measured **248 MiB in both scenarios**. Triton uses approximately one
CPU core during measured inference, versus 4.50 / 2.92 core equivalents for SIMD.

## Controlled setup

- Host: `gcp-gpu-t4` / `damian-gpu-machine-t4`, Tesla T4, `cuda:0`.
- Model: `rfdetr-small` (`coco/39`), TensorRT FP16 package
  `c70f32369a54d61e06ef4e6b56c82524`, static batch one, 512x512 network input.
- Runtime: `94002e982745b2f1cde53fd00c6b2dd3c0e90849`.
- Profiler: `31ae13531c1f9e9fd92449d3ff459c228d9dbf5d`
  from `codex/runtime-environment-v2`.
- Runner: existing isolated copy of `codex/autonomous-optimization-reports`,
  starting at `5656bf567b15a566a88ee7c4543c1ce3553f3b63`.
- Image: `sha256:d95bbc273b725e5951ddcca6fba621f4681c3fddda4f9e7842f57a5bd0b309d3`.
- Recipe: `profile-memory`, 5 warmup / 20 measured calls, one fresh worker for
  each scenario, 10 ms GPU/host sampling, one repetition.
- Production registry, no API key supplied; same persistent COCO input source.
- Fixed stages: buffer `base`, scheduler `base`, postprocessor `triton-fused-v1`,
  engine plugin `base`. Both compatibility and runtime fallback disabled.
- Base scenario: `camera_640x480_batch_1_base`, confidence 0.4, IoU 0.5,
  max detections 300. High: `camera_3840x2160_batch_1_high`, confidence 0.01,
  IoU 0.95, max detections 3000. Thus this is not a pure resolution-only sweep.

Only the requested preprocessor changes between runs. Provenance records agree
on image ID, profiler/runtime commits and `/runtime` import location. Runtime
environment, model metadata, input metadata and profiling-run settings match.
The unchanged adapter translates the plan argument and writes native selection
evidence after measurement; it does not wrap inference or alter the sampler.
Adapter SHA-256: `7912643ccb9ff682d15d6776c124b80690dfaab0f5c1ae1959887e05261b656b`.
No runtime/profiler source changes were needed for this follow-up.

## GPU memory

All values in MiB (bytes / 1,048,576), source `nvml_device`.

| Scenario | Preprocessor | After model load | Measured peak | Increment above load |
|---|---|---:|---:|---:|
| 640x480 base | pillow-simd-v1 | 222 | 248 | 26 |
| 640x480 base | triton-universal-v1 | 222 | 252 | 30 |
| 3840x2160 high | pillow-simd-v1 | 222 | 248 | 26 |
| 3840x2160 high | triton-universal-v1 | 222 | 318 | 96 |

Triton versus SIMD peak: **+4 MiB (+1.6%)** in the base scenario and
**+70 MiB (+28.2%)** in the high scenario. Triton's high-minus-base peak is
**66 MiB**; the corresponding SIMD difference is zero.

These are whole-device NVML measurements with the pre-load baseline subtracted,
not total device usage, isolated preprocessing allocations, or exact live tensor
bytes. Preflight found the GPU idle; competing workloads could still inflate
whole-device measurements. Sampling can miss peaks shorter than 10 ms. Measurement
starts after warmup, so retained buffers and allocator reservations contribute.

## Host RAM and CPU

| Scenario | Preprocessor | RSS after load MiB | Peak RSS MiB | Incremental RSS MiB | Average CPU % |
|---|---|---:|---:|---:|---:|
| 640x480 base | pillow-simd-v1 | 1013.129 | 1227.492 | 214.363 | 450.40 |
| 640x480 base | triton-universal-v1 | 1009.172 | 1200.539 | 191.367 | 102.17 |
| 3840x2160 high | pillow-simd-v1 | 1013.305 | 1244.121 | 230.816 | 291.76 |
| 3840x2160 high | triton-universal-v1 | 1011.516 | 1280.746 | 269.230 | 102.29 |

Triton host peak is **26.953 MiB lower** in the base scenario and **36.625 MiB
higher** in the high scenario. Its average CPU utilization is lower by 77.3% /
64.9%, respectively. This is utilization, not a measured reduction in CPU time
per request. 100% means one core equivalent, not the entire host.

Host RAM uses process RSS (`procfs_process_rss`); CPU uses process CPU time over
wall time (`process_time`). The host baseline is before input construction;
incremental RSS includes inputs and retained warmup allocations. RSS also includes
imported libraries and other process state, so differences cannot be attributed
entirely to preprocessing buffers.

## Why resolution affects Triton but not SIMD GPU memory

Pillow-SIMD resizes and normalizes the CPU image to 512x512 before GPU transfer:
see [CPU resize](https://github.com/roboflow/inference/blob/94002e982745b2f1cde53fd00c6b2dd3c0e90849/inference_models/inference_models/models/rfdetr/pre_processing.py#L350)
and [transfer](https://github.com/roboflow/inference/blob/94002e982745b2f1cde53fd00c6b2dd3c0e90849/inference_models/inference_models/models/rfdetr/pre_processing.py#L250).
The GPU therefore receives the same-sized float32 input for both source resolutions.

Triton performs resizing on the GPU. Its
[staging state](https://github.com/roboflow/inference/blob/94002e982745b2f1cde53fd00c6b2dd3c0e90849/inference_models/inference_models/models/rfdetr/triton_universal_preprocess_runtime.py#L122)
allocates two full-source-resolution uint8 GPU buffers, two corresponding pinned
host buffers, and a GPU scratch tensor shaped `(3, source_height, target_width)`.
Both ring slots are allocated even for batch one. The GPU staging tensors alone
require about **1.76 MiB at 640x480 versus 47.46 MiB at 4K**; scratch grows from
**0.70 to 3.16 MiB**. It also allocates resampling tables and the fixed-size output.

This source inspection explains the direction of the measured increase. It is
not a complete allocation trace or an exact decomposition of the 66 MiB increase;
allocator reservations, kernel/runtime state, and output/postprocessing can also
contribute. Lazy preprocessing state is allocated during inference, so the
post-load GPU residency remains 222 MiB for all four cases.

## Validation, limitations and artifacts

Both runs are `DONE`; all four results are `profiled`, with no failed/discarded
scenarios. Required GPU, host RAM and CPU fields are non-null. Native worker
proofs match profile IDs/scenarios, and verify requested/effective IDs for all
five stages without fallback at model selection and final execution. This is not
a per-call trace. Optional TensorRT context/workspace metadata is unavailable.

These are single runs in fresh workers, not repeated statistical trials. SIMD
was measured earlier; it was not rerun alongside this follow-up. Results describe
the two selected full-inference scenarios, not standalone preprocessing costs.

- Triton run: `rfdetr-t4-triton-memory-20260918`:
  required memory summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-memory-20260918/memory-profile.md`),
  raw memory JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-memory-20260918/output/memory.json`),
  provenance (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-memory-20260918/output/run-metadata.json`).
- SIMD baseline: `rfdetr-t4-simd-memory-20260918`:
  memory summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-memory-20260918/memory-profile.md`),
  raw memory JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-memory-20260918/output/memory.json`).
- [Earlier SIMD memory and output-delta report](rfdetr-simd-memory-output-delta-t4-2026-09-18.md).

Original artifacts remain at `/home/damiankosowski/remote-runner/runs/<run-id>/`
on the T4 host. Fetched artifacts are under the temporary runner's `run-results`.
