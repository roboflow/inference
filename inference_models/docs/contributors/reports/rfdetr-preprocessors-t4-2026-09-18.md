# RF-DETR preprocessor comparison on T4 — 2026-09-18

## Results

All three runs completed with `DONE`; all six scenario results are `profiled`.
Values are end-to-end inference latency in milliseconds, lower is better.

| Scenario | Preprocessor | Mean | p50 | p95 | p99 |
|---|---|---:|---:|---:|---:|
| camera_640x480_batch_1_base | base | 14.719 | 14.549 | 16.463 | 17.138 |
| camera_640x480_batch_1_base | pillow-simd-v1 | 10.328 | 10.047 | 12.552 | 14.329 |
| camera_640x480_batch_1_base | triton-universal-v1 | 5.799 | 5.519 | 7.573 | 9.871 |
| camera_3840x2160_batch_1_high | base | 68.821 | 68.685 | 70.444 | 71.863 |
| camera_3840x2160_batch_1_high | pillow-simd-v1 | 32.024 | 31.911 | 32.864 | 34.485 |
| camera_3840x2160_batch_1_high | triton-universal-v1 | 12.336 | 12.120 | 13.831 | 14.563 |

Compared with `base`, mean latency decreased:

- **Pillow-SIMD:** 29.8% at 640x480 and 53.5% at 3840x2160
  (1.43x and 2.15x latency ratios).
- **Triton:** 60.6% at 640x480 and 82.1% at 3840x2160
  (2.54x and 5.58x latency ratios).

Triton was the fastest of the three in both scenarios. These are within-T4,
within-scenario comparisons, not speedups relative to the Orin runs or isolated
preprocessing-kernel speedups. Pillow-SIMD changes numerical behavior; see below.

### Repetition selection

The recipe automatically selects stable repetitions and concatenates their
individual measured latency samples. Nothing was manually excluded.

| Preprocessor | Scenario | Attempted workers | Selected repetitions | Measured samples |
|---|---|---:|---|---:|
| base | 640x480 | 12 | 8, 10, 11, 12 | 400 |
| base | 3840x2160 | 5 | 1–5 | 500 |
| pillow-simd-v1 | 640x480 | 11 | 7, 8, 10, 11 | 400 |
| pillow-simd-v1 | 3840x2160 | 5 | 1–5 | 500 |
| triton-universal-v1 | 640x480 | 5 | 1–5 | 500 |
| triton-universal-v1 | 3840x2160 | 5 | 1–5 | 500 |

The small scenario showed substantial variation across workers, especially early
in the base/SIMD runs. Percentiles and excluded worker measurements remain in the
raw artifacts. Treat the percentages as this experiment's observations, not a
locked-clock guarantee. Runs executed sequentially in SIMD → base → Triton order.

### Host telemetry limitations

The host retained its 70 W power limit and automatic clocks. Whole-run SM clock
ranges (including idle/initialization) were 300–1020 MHz for base and 300–1590 MHz
for SIMD and Triton. GPU peak temperatures were 46 C, 46 C and 49 C respectively.
Base and SIMD recorded no throttle-event evidence. Triton recorded one
`clocks_event_reasons.sw_power_cap=Active` sample at 10:10:30.999 UTC, during the
first small-scenario worker's lifetime. That worker includes initialization,
warmup and measurement, so this does not identify a specific affected timed call.
Its results remain included; the profiling run and convergence criterion succeeded.
Telemetry is reported separately and is not merged into the profiler's latency JSON.

### Artifacts

- Pillow-SIMD summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-pillow-simd-20260918/latency-profile.md`),
  latency JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-pillow-simd-20260918/output/latency.json`).
- Base summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-base-20260918/latency-profile.md`),
  latency JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-base-20260918/output/latency.json`).
- Triton summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-20260918/latency-profile.md`),
  latency JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-20260918/output/latency.json`).
- Environment inspection (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-environment-20260918/output/environment.json`),
  host inventory (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-environment-20260918/output/host-inventory.json`).

Each run also contains its log, provenance, native selection sidecars and separate
telemetry. Local raw artifacts are in the temporary runner copy; remote originals
remain under `/home/damiankosowski/remote-runner/runs/<run-id>/` on the T4 host.

## Scope and package discovery

Compare `base`, `triton-universal-v1`, and `pillow-simd-v1` on `gcp-gpu-t4`, using
the same two scenarios as the Orin experiment. `threaded-exact-v1` is excluded.
The recipe measures end-to-end `infer()` latency, not preprocessing-only latency.

Read-only production Firestore queries using `debug_firestore` at commit
`5c26e74a80a23965b81aeaea35a03e9cdc07e848` resolved:

- Public alias `rfdetr-small` → canonical model `coco/39`.
- Alias document: `models_aliases/983f35d84a3d9de82d6b21cc0f66dde866d6156c4c1a455ce97ff4ddfdc080f8`.
- Model document: `registered_models/60733eb9f6959af7769ab805e029c6de4ae2f0353ca004f05b1df270bc7d090a`.
- Package from its `model_packages` subcollection:
  **`c70f32369a54d61e06ef4e6b56c82524`**.
- Manifest: sealed, not deleted, TensorRT FP16, `tesla-t4`, compute capability
  7.5, GPU server, static batch 1, CUDA 12.4.127.post1, TensorRT 10.12.0.36,
  TRT forward-compatible, not same-CC hardware-compatible.

The CLI's `get-registered-model rfdetr-small` does not resolve aliases; it returns
no match. Resolve `models_aliases/sha256(alias)` first, then query the canonical
model. No Firestore writes were made.

## Controlled setup

- Host: `damian-gpu-machine-t4`, `us-central1-a`, Tesla T4, 15,360 MiB,
  x86-64, NVIDIA driver 550.163.01.
- Runtime: `94002e982745b2f1cde53fd00c6b2dd3c0e90849`, same as the Orin comparison.
- Profiler: `codex/runtime-environment-v2`, commit
  `31ae13531c1f9e9fd92449d3ff459c228d9dbf5d`.
- Runner: isolated copy of `codex/autonomous-optimization-reports`, commit
  `5656bf567b15a566a88ee7c4543c1ce3553f3b63`, with profiling-only adjustments below.
- Production registry, no Roboflow API key supplied. Device `cuda:0`.
- Scenarios: `camera_640x480_batch_1_base` and `camera_3840x2160_batch_1_high`.
- 20 warmup / 100 measured iterations per fresh worker; minimum 5 / maximum 20
  repetitions; stable-recent-p50 convergence, threshold 0.05, window 5.
- All stages fixed except preprocessor: buffer `base`, scheduler `base`,
  postprocessor `triton-fused-v1`, engine plugin `base`.
- Compatibility fallback and runtime-failure fallback both disabled.
- COCO validation inputs verified by `rfdetr-t4-coco-20260918`: `DONE`,
  `images_dir: /data/coco/val2017`.

Environment run `rfdetr-t4-environment-20260918` completed with `DONE`. Both
environment and host inventory use schema 2.0, contain no JSON nulls or diagnostics,
and identify the expected T4. The host inventory reports a 70 W power limit.
Provenance confirms the requested commits and imports below `/runtime`.
No power or clock settings were changed; locked clocks are not claimed.
Jetson model/power-mode fields do not apply to this discrete-GPU host.

The scenarios have different detection settings as well as image sizes: base uses
confidence 0.4 / IoU 0.5 / max detections 300; high uses 0.01 / 0.95 / 3000.
Compare preprocessors within each scenario, not resolution alone across scenarios.

## Isolated SIMD image

The existing GPU image had standard Pillow but no optional SIMD installation.
A local profiling image was built without replacing the existing image:

- Base: `roboflow/roboflow-inference-server-gpu:1.3.7`, pinned as
  `roboflow/roboflow-inference-server-gpu@sha256:bc07000849424983f8526979df643a8d49b1101d3b2e278104d41b53c2d543f1`.
- Profiling image ID:
  `sha256:d95bbc273b725e5951ddcca6fba621f4681c3fddda4f9e7842f57a5bd0b309d3`.
- Python 3.10.12, Torch 2.6.0+cu124, CUDA 12.4, TensorRT 10.12.0.36, Triton 3.2.0.
- Standard Pillow 12.3.0 remains the `PIL` package.
- Pillow-SIMD 12.3.0.post0 installed with `--no-deps --target /opt/pillow_simd`,
  from commit `d50c9d40e84b6cdbe3bcde19b6c77ccdff2421f7`.
- Built wheel SHA-256: `3e533ec5275980b871ec9697426039633e2d8c3639e5e06871a80af0cd2a4212`.
- The repository's `verify_pillow_simd.py` passed, checking separate extension
  modules, preserved standard Pillow, expected versions, and a resize smoke test.
- Build context: `/private/tmp/rfdetr-execution-plan-runner/t4-image/` locally,
  `/home/damiankosowski/remote-runner/rfdetr-simd-image-20260918/` remotely.
- Dockerfile SHA-256: `f734411173cb5d140df1e335288e2b23ac2f8f7057a1114d9d3d5ab89a4a9c03`.

This local image is pinned by immutable image ID, not a registry manifest digest.
The isolated runner was updated to inspect local `sha256:` image IDs instead of
trying to pull them. Its existing JSON-argv fix and profiling plan adapter are
retained. Original runner and debug-firestore checkouts are unchanged.

## Selection evidence and numerical behavior

The same profiling-only adapter as the Orin experiment converts JSON execution
plans to typed RF-DETR plans and verifies native selection metadata after each
worker's timed measurements. It does not wrap any timed inference method. Each
worker writes `selection-proof-<pid>.json`, keyed by its `profile_id`.

Post-run validation matched all **43 attempted worker profile IDs**: 17 base,
16 SIMD and 10 Triton, including excluded repetitions. All five stages matched
their requested implementation at model selection and the last execution, with
no fallback. Both fallback policies apply to every invocation; the metadata is
the last invocation per worker, not a per-iteration trace. All runs have identical
image ID, runtime/profiler revisions, package and input/inference settings, apart
from the requested preprocessor. No package results were skipped or failed.

The SIMD run's legacy helper log says `implementation=base max_workers=1` because
SIMD reuses the single-thread reference orchestration with an explicit SIMD
`image_module`. That line does not identify the resize backend. The execution-plan
metadata identifies `pillow-simd-v1`, and the source passes its isolated module to
`fromarray` / `resize`; fallback is disabled. Selection verification uses this
native stage metadata rather than the legacy log label.

- Adapter SHA-256: `467b8b50c6ec793e148cdbb4288648f3ff703899a2d1cde97a40724383a11a89`.
- Patched runner launch SHA-256: `c1fa22bb22e94e3685f5aa16851279ca31f045513897867ad23569e2dbaeceda`.

`pillow-simd-v1` is intentionally **not numerically identical** to standard Pillow.
Its metadata declares changed numerics: bilinear downscales can differ by one
uint8 intensity level before normalization. The PR's approximately 0.1% differing
pixels is workload-specific, not a guarantee. Latency profiling is not a detection
accuracy or numerical-parity evaluation.

## Reproduction

From the isolated runner copy, using its saved profiling adapter and the local T4
image above, the SIMD command is:

```bash
source .venv/bin/activate
./scripts/launch.sh \
  --host gcp-gpu-t4 --recipe profile-latency \
  --ref 31ae13531c1f9e9fd92449d3ff459c228d9dbf5d \
  --runtime-ref 94002e982745b2f1cde53fd00c6b2dd3c0e90849 \
  --docker-image sha256:d95bbc273b725e5951ddcca6fba621f4681c3fddda4f9e7842f57a5bd0b309d3 \
  --asset plan_adapter=/private/tmp/rfdetr-execution-plan-runner/profiling-assets/sitecustomize.py \
  --script-arg device=cuda:0 \
  --script-arg package-ids=c70f32369a54d61e06ef4e6b56c82524 \
  --script-arg profiling-scenarios=camera_640x480_batch_1_base,camera_3840x2160_batch_1_high \
  --script-arg 'from-pretrained-kwargs-json={"execution_plan":{"preprocessor":"pillow-simd-v1","buffer_strategy":"base","scheduler":"base","postprocessor":"triton-fused-v1","engine_plugin":"base","allow_compatibility_fallback":false,"allow_runtime_failure_fallback":false}}'
```

Change only `preprocessor` for `base` and `triton-universal-v1`. The launches in this
experiment used the named profiler branch, with the same commit verified in every
run's provenance; this reproduction command pins that commit directly.
