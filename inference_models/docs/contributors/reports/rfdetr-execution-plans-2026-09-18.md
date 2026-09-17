# RF-DETR object-detection execution plans — 2026-09-18

## Implementation

Branch: `codex/rfdetr-object-detection-execution-plans`.
Validated runtime: `94002e982745b2f1cde53fd00c6b2dd3c0e90849`.
Baseline: `b77b7a08cb1e484742d9eaf5b48e48b534081289`.

- Torch and ONNX object detection now resolve all five execution-plan stages.
  They share preprocessing selection, compatibility/runtime fallback handling,
  effective-ID metadata and warning deduplication with TensorRT. Their semantic
  forward and backend-specific postprocessing remain unchanged behind base adapters.
- Preprocessing `auto` prefers Triton Universal, then threaded exact, then base.
  CUDA-only metadata prevents constructing Triton for CPU/MPS. Unsupported model
  configurations, requests and custom Torch image-size overrides use reference
  preprocessing with an observable fallback reason. Strict plans reject fallback.
- Composed inference uses an exact-tensor CUDA event handoff; standalone
  preprocessing synchronizes before returning. Reference CUDA input storage and
  produced tensors are recorded on their consumer streams for allocator safety.
- The standard-Pillow base path incorporates PR #2988: resize first, then swap
  BGR/RGB on the resized image. The two-step dataset-version resize path is retained.
  This shared reference change also benefits callers outside the migrated backends.
- `pillow-simd-v1` is separate and explicit, never auto-selected. Metadata declares
  `changes_numerics=True`, uint8 NumPy input restrictions and x86 host compatibility.
  The loader checks Linux/SSE4.1, loads a distinct native extension under an isolated
  module name, and reports absent/disabled/incompatible builds as fallback reasons.
  Standard Pillow remains untouched.
- The optional SIMD build is pinned to Pillow-SIMD 12.3.0.post0, commit
  `d50c9d40e84b6cdbe3bcde19b6c77ccdff2421f7`, satisfying the standard Pillow >=12.3
  version floor discussed in Slack. Five x86 ONNX Dockerfiles install it separately
  in `/opt/pillow_simd`; a build verifier checks module isolation and resize tolerance.
  `INFERENCE_MODELS_PILLOW_SIMD_PATH` overrides the path; an empty value disables it.
- SIMD bilinear downscales can differ by one uint8 intensity level. The roughly
  0.1% differing pixels reported in PR #2989 are workload-specific, not a universal
  guarantee. Native numerical tests include no-op exactness and downscale tolerance.
- Instance-segmentation execution-plan migration is intentionally deferred.

Example (Torch/ONNX also support the preprocessor environment variable):

```python
from inference_models import AutoModel
from inference_models.models.rfdetr.optimization.execution_plan import RFDetrExecutionPlan

model = AutoModel.from_pretrained(
    "rfdetr-small",
    backend="torch",  # or "onnx"
    rfdetr_execution_plan=RFDetrExecutionPlan(
        preprocessor_id="triton-universal-v1",
        postprocessor_id="base",
    ),
)
# Inspect both model selection and effective last-request stages:
print(model.optimization_runtime_metadata)
```

For SIMD, explicitly select `preprocessor_id="pillow-simd-v1"`. On Jetson ARM it
resolves to `base`; no SIMD speedup is claimed for Orin.

## Validation

Command, with the source checkout first on `PYTHONPATH`:

```shell
python -m pytest inference_models/tests/unit_tests/models/rfdetr \
  inference_models/tests/unit_tests/models/optimization \
  -q --disable-warnings --maxfail=3
```

- Local macOS CPU environment: **212 passed, 30 skipped**, 7 warnings.
- Orin, same image digest as profiling: **239 passed, 3 skipped**, 28 warnings.
  All twelve new CUDA parity/event-handoff combinations passed (Torch/ONNX,
  base/Triton, NumPy/uint8 CUDA/float CUDA inputs).
- The three Orin skips are real native Pillow-SIMD tests: ARM is intentionally
  unsupported. Loader isolation, compatibility, numerical metadata and fallback
  tests use controlled module doubles and pass on both hosts.
- New backend integration tests use lightweight model/session doubles; real learned
  Torch/ONNX model end-to-end comparisons remain future work. Existing Triton and
  execution-plan GPU tests also passed in the Orin suite.
- Black and isort checks pass for the changed Python modules, and `git diff --check`
  is clean. Native Linux x86 SIMD execution and full Docker builds were not run:
  the available validation hosts are ARM and the local Docker daemon is stopped.

## Profiling setup

- Runner branch `codex/autonomous-optimization-reports`, commit
  `5656bf567b15a566a88ee7c4543c1ce3553f3b63`, recipe `profile-latency`.
- Profiler branch `codex/runtime-environment-v2`, commit
  `31ae13531c1f9e9fd92449d3ff459c228d9dbf5d`.
- Isolated runner clone: `/private/tmp/rfdetr-execution-plan-runner`. The recipe's
  only environment changes were `ROBOFLOW_ENVIRONMENT=prod` and
  `ROBOFLOW_API_HOST=https://api.roboflow.com`. No API key was supplied.
- Host `orin-agx-jp62`: Jetson AGX Orin, MAXN, integrated memory, compute 8.7;
  CUDA 12.6.0, TensorRT 10.7.0, Torch 2.6.0, Triton 3.2.0.
- Image `roboflow/roboflow-inference-server-jetson-6.2.0:latest`, digest
  `sha256:e278db8bf64245d55f0c437a7e1e74a9afc2f1128468e13d51017226579be415`.
- Production `rfdetr-small`, TensorRT fp16 package
  `bbf73c3014b86386434c8051f132cdf2`; device `cuda:0`.
- Twenty warmup and 100 measured iterations per fresh process; minimum five,
  maximum twenty repetitions, stable-recent-p50 threshold 0.05. CUDA-synchronized
  wall-clock `infer` latency includes preprocessing. COCO validation images are
  resized to each fixed source-camera scenario before the measured model call.
- No execution-plan override was passed; both revisions use the model defaults.
  This profiler only emits effective-stage proof for explicit plan requests, so
  these latency artifacts do not themselves prove the selected preprocessor ID.

## Results

Both final scenarios completed with `status=profiled` and run status `DONE`.
Each used five repetitions, 500 included measurements, stable-recent-p50
termination, and no excluded repetitions. Runtime/profiler SHAs, `/runtime` import
path and identical image digest were verified from the fetched run metadata.

| Scenario | Revision | Mean ms | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|
| camera_640x480_batch_1_base | Baseline | 7.533 | 7.475 | 7.891 | 7.988 |
| camera_640x480_batch_1_base | Final candidate | 7.507 | 7.473 | 7.867 | 7.906 |
| camera_3840x2160_batch_1_high | Baseline | 10.446 | 10.419 | 10.651 | 10.847 |
| camera_3840x2160_batch_1_high | Final candidate | 10.429 | 10.420 | 10.574 | 10.657 |

Mean-latency changes are **-0.34%** and **-0.17%**, respectively. These tiny
differences do not establish a meaningful speedup; no material mean/p50 regression
was observed. TensorRT was already using the execution-plan architecture before
this migration. An intermediate candidate measured 7.589 / 10.530 ms, illustrating
the run-to-run variability; it is not substituted for the final commit's results.

Telemetry completed without diagnostics or reported throttle/clock-event evidence.
Baseline/final GPU peak temperatures were 48.875°C / 47.968°C, with 106 samples in
each run. This is not proof of locked GPU clocks.

Runs and artifacts:

- Baseline: `rfdetr-execution-plan-baseline-prod-20260918`.
- Final: `rfdetr-execution-plan-final-prod-20260918`.
- Intermediate candidate: `rfdetr-execution-plan-candidate-prod-20260918`.
- Environment inventory: `rfdetr-execution-plan-environment-20260918`.
- Local artifact root: `/private/tmp/rfdetr-execution-plan-runner/run-results`.
  Each profiling run has a `latency-profile.md` summary, `run.log`, and
  `output/{latency.json,run-metadata.json,status.txt,host-telemetry.json}`.
- Remote root: `/home/roboflow/remote-runner/runs`, on `orin-agx-jp62`.

Reproduction command from the isolated runner clone (production recipe settings
above):

```shell
source .venv/bin/activate
./scripts/launch.sh --host orin-agx-jp62 --recipe profile-latency \
  --ref codex/runtime-environment-v2 \
  --runtime-ref 94002e982745b2f1cde53fd00c6b2dd3c0e90849 \
  --script-arg device=cuda:0 \
  --script-arg profiling-scenarios=camera_640x480_batch_1_base,camera_3840x2160_batch_1_high
```

Use baseline runtime SHA `b77b7a08cb1e484742d9eaf5b48e48b534081289` for the
before measurement. Keep the recorded image digest fixed for later comparisons.

## Caveats and diagnostic attempts

- The staging recipe's `rfdetr-small` package allow-list returned no measurements.
  Production resolved the expected TensorRT package, as the user suggested.
- A staging nano diagnostic run found a TensorRT package but failed because the
  profiler worker rejects bare `cuda`; all reported runs use indexed `cuda:0`.
  Neither staging attempt is included in the comparison.
- An additional `AutoModel` metadata smoke check was blocked by the package's
  `trusted_source` flag. The trust guard was not disabled. The requested profiler
  uses its own backend-class loader and completed independently of that check.
- TensorRT warns that the engine was built on a different device model; both
  compared runs use the exact same production engine.
- Inventory reports MAXN, but `jetson_clocks --show` requires root and failed its
  read-only probe. No power or clock settings were changed. Telemetry is preserved
  separately from latency data; it does not establish locked GPU clocks.
- This is a TensorRT regression comparison. It does not measure the speedup from
  migrating Torch/ONNX, nor isolate the base-channel-swap or x86 SIMD improvement.
