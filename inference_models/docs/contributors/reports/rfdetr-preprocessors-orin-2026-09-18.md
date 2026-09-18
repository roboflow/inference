# RF-DETR preprocessor comparison on Orin — 2026-09-18

This compares different preprocessors at **one runtime revision**, not before/after
code revisions. `threaded-exact-v1` is intentionally excluded at the user's request.
The requested `profile-latency` recipe measures complete `infer()` calls, so the
numbers are end-to-end inference latency with each preprocessor, not isolated
preprocessing-kernel latency.

## Results

All latency values below are milliseconds, lower is better. Both runs completed
with `DONE` and both scenario results in each run are `profiled`.

| Scenario | Preprocessor | Mean | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| camera_640x480_batch_1_base | base | 13.524 | 13.439 | 14.159 | 14.745 |
| camera_640x480_batch_1_base | triton-universal-v1 | 7.562 | 7.510 | 7.888 | 8.181 |
| camera_3840x2160_batch_1_high | base | 46.959 | 46.911 | 47.248 | 47.647 |
| camera_3840x2160_batch_1_high | triton-universal-v1 | 10.465 | 10.431 | 10.730 | 10.793 |
| Both | pillow-simd-v1 | Unsupported on aarch64 | — | — | — |

Relative to `base`, selecting `triton-universal-v1` reduced mean end-to-end latency:

- 640x480: **44.1%**, saving 5.962 ms (1.79x latency ratio).
- 3840x2160: **77.7%**, saving 36.495 ms (4.49x latency ratio).

All four results converged after five fresh workers per scenario. Metrics
concatenate the selected workers' latency samples, not worker percentile averages.
The profiler's convergence rule excluded repetition 3 for `base` at 640x480,
leaving 400 measured samples; the other three results each use all 500 samples.
This automatic exclusion is retained in the raw artifact, not manually chosen.

Run IDs:

- `base`: `rfdetr-preprocessors-base-v3-20260918`.
- `triton-universal-v1`: `rfdetr-preprocessors-triton-20260918`.

Per-run summaries and raw artifacts:

- Base summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-preprocessors-base-v3-20260918/latency-profile.md`),
  latency JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-preprocessors-base-v3-20260918/output/latency.json`).
- Triton summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-preprocessors-triton-20260918/latency-profile.md`),
  latency JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-preprocessors-triton-20260918/output/latency.json`).

Local artifacts live in the temporary runner clone. Remote originals remain under
`/home/roboflow/remote-runner/runs/<run-id>/` on `orin-agx-jp62`.

## Controlled setup

- Host: `orin-agx-jp62`, Jetson AGX Orin (aarch64), MAXN, CUDA device `cuda:0`.
- Model: production `rfdetr-small`, TensorRT FP16 package
  `bbf73c3014b86386434c8051f132cdf2`; no API key supplied.
- Runtime for every run: `94002e982745b2f1cde53fd00c6b2dd3c0e90849`.
  This includes the post-resize channel swap from PR #2988 in `base`.
- Profiler: `codex/runtime-environment-v2`, commit
  `31ae13531c1f9e9fd92449d3ff459c228d9dbf5d`.
- Runner: `codex/autonomous-optimization-reports`, commit
  `5656bf567b15a566a88ee7c4543c1ce3553f3b63`, in the isolated clone
  `/private/tmp/rfdetr-execution-plan-runner`, with experiment-local adjustments
  documented below. The original runner checkout was not changed.
- Image pinned by digest:
  `roboflow/roboflow-inference-server-jetson-6.2.0@sha256:e278db8bf64245d55f0c437a7e1e74a9afc2f1128468e13d51017226579be415`.
- CUDA 12.6.0, TensorRT 10.7.0, Torch 2.6.0, Triton 3.2.0, compute capability 8.7.
- Inputs: COCO validation images resized to the two requested fixed-camera
  workloads, batch size one. Input preparation outside the model is not timed.
- Twenty warmup / 100 measured iterations per fresh worker. Minimum five,
  maximum twenty repetitions, stable-recent-p50 threshold 0.05.
- Measurement: CUDA-synchronized wall-clock `infer()` latency.

The scenarios also differ in detection settings: the base scenario uses confidence
0.4, IoU threshold 0.5 and max detections 300; the high scenario uses confidence
0.01, IoU threshold 0.95 and max detections 3000. Compare preprocessors within each
scenario; the difference between scenario latencies is not solely image resolution.

Only the preprocessor differs:

| Stage | Selection |
|---|---|
| Preprocessor | `base` or `triton-universal-v1` |
| Buffer strategy | `base` |
| Scheduler | `base` |
| Postprocessor | `triton-fused-v1` |
| Engine plugin | `base` |
| Compatibility fallback | Disabled |
| Runtime-failure fallback | Disabled |

## Pillow-SIMD compatibility

`pillow-simd-v1` is **unsupported on this host**. Its metadata rejects the Orin
architecture before constructing the native extension:

```text
host architecture 'aarch64' is unsupported; requires ('x86_64', 'amd64')
```

No SIMD latency is reported and no fallback is timed under the SIMD label. A real
SIMD comparison requires a Linux x86-64/SSE4.1 host and the isolated Pillow-SIMD
installation. This experiment does not switch hosts.

## Selection verification and runner adjustments

The selected profiler revision predates its newer execution-plan CLI/proof support.
A profiling-only `sitecustomize.py` asset bridges the requested JSON plan to the
runtime's `RFDetrExecutionPlan` object via `rfdetr_execution_plan`. It:

1. Requires all five explicit stage IDs and disables both fallback policies.
2. Asserts that model initialization selected the exact requested plan.
3. After each worker's timed measurements, reads the runtime's native
   `optimization_runtime_metadata` and checks every stage's model-level and latest
   request-level requested/effective IDs and null fallback reason.
4. Saves unmodified native metadata, requested plan, scenario and the worker's
   `profile_id` to `output/selection-proof-<pid>.json`. Missing or mismatched
evidence raises an error instead of accepting a mislabeled measurement.

Post-run validation matched all **20 worker profile IDs** (ten per run), including
the excluded base repetition, to verified sidecars. Every model-level and final
request-level stage matched its requested implementation without fallback. Both
provenance files confirm the same runtime/profiler commits, pinned image and runtime
import location. Package, input and inference settings match between runs.

No preprocessing, forward, postprocessing or timed inference method is wrapped.
The checks and JSON writes happen outside the timed measurements. Strict runtime
fallback policy applies to every invocation; request metadata describes the last
invocation of each worker, not a separate trace of every iteration.

Adapter: `/private/tmp/rfdetr-execution-plan-runner/profiling-assets/sitecustomize.py`.
SHA-256: `467b8b50c6ec793e148cdbb4288648f3ff703899a2d1cde97a40724383a11a89`.
A local unit test validates plan translation, unchanged inference method,
fallback rejection and detection of mismatched native evidence.

The isolated recipe also selects the production registry and mounts the adapter.
The runner's launch script was locally corrected to parse its shell-quoted script
argument string with `shlex.split` into an argv array; the original version passed
literal quote characters to JSON arguments. No `eval` is used. Patched launch
script SHA-256: `60bbfb449b195afe768838ee166576d1284f640453aee3fa30560f84f2d3a29b`.

Two setup attempts produced no usable measurements and are excluded:

- `rfdetr-preprocessors-base-20260918`: unsupported newer
  `--execution-plan-json` flag. Corrected to the selected branch's
  `--from-pretrained-kwargs-json` argument.
- `rfdetr-preprocessors-base-v2-20260918`: JSON decode error caused by the runner's
  literal shell quotes. Corrected in the isolated launch script before measurement.

## Limits

The engine emits the same device-model portability warning in each run. Inventory
reports MAXN, but the read-only clock-state probe lacks root permission. No power
or clock settings are changed. Telemetry is kept separate from latency artifacts.
GPU peak temperatures were 47.312 C (`base`) and 48.187 C (Triton); CPU peaks
were 50.312 C and 50.375 C respectively. Neither telemetry artifact recorded
throttle/clock event evidence; this is not proof of locked GPU clocks.
This experiment compares preprocessing choices within the TensorRT path; it does
not benchmark Torch, ONNX or instance segmentation.
