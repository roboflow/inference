# RF-DETR SIMD memory and output comparison on T4 — 2026-09-18

## Outcome

All three requested runs completed successfully with the expected preprocessors
and no fallback. Pillow-SIMD measured 222 MiB GPU residency and 248 MiB GPU peak
in both scenarios. The saved model outputs are **not numerically identical**.
The base scenario preserves all 16 same-class detections with at most a one-pixel
coordinate difference. The high scenario's low-confidence predictions differ more,
and a confidence change crosses a 0.4 threshold in the diagnostic view below.

## Controlled setup

Follow-up to the [T4 latency comparison](rfdetr-preprocessors-t4-2026-09-18.md),
using the same host, package, image, runtime revision, and two scenarios.

- Host: `gcp-gpu-t4` / `damian-gpu-machine-t4`, Tesla T4.
- Public model: `rfdetr-small` → canonical `coco/39`.
- TensorRT FP16 package: `c70f32369a54d61e06ef4e6b56c82524` (T4, static batch one).
- Runtime: `94002e982745b2f1cde53fd00c6b2dd3c0e90849`.
- Profiler: `codex/runtime-environment-v2`, commit
  `31ae13531c1f9e9fd92449d3ff459c228d9dbf5d`.
- Runner: isolated copy of `codex/autonomous-optimization-reports`, starting at
  `5656bf567b15a566a88ee7c4543c1ce3553f3b63`.
- Image: `sha256:d95bbc273b725e5951ddcca6fba621f4681c3fddda4f9e7842f57a5bd0b309d3`,
  standard Pillow 12.3.0 plus isolated Pillow-SIMD 12.3.0.post0.
- Production registry, no API key supplied; device `cuda:0`.
- Fixed stages: buffer `base`, scheduler `base`, postprocessor `triton-fused-v1`,
  engine plugin `base`. Both fallback policies disabled. `threaded-exact-v1` excluded.
- Scenarios: `camera_640x480_batch_1_base` (confidence 0.4, IoU 0.5, max detections
  300) and `camera_3840x2160_batch_1_high` (0.01, 0.95, 3000).

The recipes were adjusted only in the temporary runner copy to use production and
the explicit-plan adapter. No runtime/profiler production source or original runner
checkout was changed. The adapter translates JSON to the typed RF-DETR plan and
validates native model-selection and final-execution metadata **after** each worker
finishes. It neither wraps inference methods nor alters sampler/output results.
Memory evidence is keyed by profile ID; snapshot evidence by bundle directory.

Adapter: `/private/tmp/rfdetr-execution-plan-runner/resource-snapshot-assets/sitecustomize.py`.
SHA-256: `7912643ccb9ff682d15d6776c124b80690dfaab0f5c1ae1959887e05261b656b`.
Its unit test covers all three worker hooks, returned-record preservation, strict
plans, and unchanged inference methods. Three comparison tests cover reordering,
coordinate changes, class mismatches, empty outputs, and differing inputs.

## Pillow-SIMD memory profile

Run: `rfdetr-t4-simd-memory-20260918`, recipe `profile-memory`. The recipe defaults
were retained: 5 warmup / 20 measured calls, one fresh worker per scenario, 10 ms
sampling. Both results are `profiled`, `status.txt` is `DONE`, and both native
selection proofs validate all stages without fallback.

MiB = bytes / 1,048,576. These are complete inference resource measurements,
not isolated preprocessing allocations.

| Scenario | GPU residency MiB | GPU peak MiB | GPU incremental MiB | Host RSS after load MiB | Host RSS peak MiB | Host incremental MiB | Average CPU % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 640x480 base | 222.000 | 248.000 | 26.000 | 1013.129 | 1227.492 | 214.363 | 450.40 |
| 3840x2160 high | 222.000 | 248.000 | 26.000 | 1013.305 | 1244.121 | 230.816 | 291.76 |

All required metrics are present. GPU source is `nvml_device`: whole-device NVML,
with the pre-load baseline subtracted. Competing GPU workloads could inflate this
measurement; preflight found the GPU idle and the requested runs were sequential.
Sampling can miss short-lived peaks. Host RAM is process RSS (`procfs_process_rss`)
and CPU uses process CPU time / wall time (`process_time`); 100% is one core, so
450% means roughly 4.5 core equivalents, not 450% of the machine.

The host-RAM baseline is taken before input construction. Its incremental value
includes input materialization and allocations retained from warmup, not only
temporary inference buffers. No memory comparison to Triton is claimed because
only Pillow-SIMD memory profiling was requested in this initial experiment.
A subsequent requested Triton run is documented in the
[memory comparison](rfdetr-preprocessors-memory-t4-2026-09-18.md).

- Memory summary (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-memory-20260918/memory-profile.md`).
- Raw memory profile (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-memory-20260918/output/memory.json`).

## Snapshot comparison method

Recipe `snapshot-model-output` uses one warmup and captures one full postprocessed
result per scenario. It stores the exact invocation and lossless NPY arrays. COCO
images are selected deterministically by sorted path, batch one. This is a small
output-delta experiment, not COCO mAP evaluation or a dataset-wide parity guarantee.

The comparison first requires identical input arrays (shape, dtype, every element)
and inference kwargs. It compares detection arrays in returned order, then performs
same-class, one-to-one matching at IoU >= 0.5, maximizing match count first and total
IoU second. Matched box-coordinate deltas are in source-image pixels; confidence
deltas are absolute score differences on the 0–1 scale. Signed deltas use SIMD
minus Triton. Reordered detections and unmatched detections are reported separately.
Canonical JSON/NPY data, not JPEG previews, determine every numerical result.

Comparison script: `/private/tmp/rfdetr-execution-plan-runner/resource-snapshot-assets/compare_snapshots.py`.
Pillow-SIMD declares changed numerics; this comparison neither assumes exact parity
nor treats Triton as ground-truth labels. It does not snapshot preprocessed tensors
or raw model logits, and a single capture cannot independently quantify engine
run-to-run nondeterminism.

## Snapshot delta results

Both runs captured COCO image `000000000139.jpg`, resized outside the model to the
scenario dimensions. Input arrays and all invocation kwargs are exactly identical
between preprocessors in each scenario. SHA-256 of the uint8 image bytes:

- 640x480: `96bf9837e55a7a5134550b40d8cc396c0d3906c91ed21cd4c7b7884d515f8b0b`.
- 3840x2160: `8e2133da6685c41b3913bb1b41c5270275e454817d3e770c768fda5a3746ee5e`.

### Full captured outputs

| Metric | 640x480 base | 3840x2160 high |
|---|---:|---:|
| Triton / SIMD predictions | 16 / 16 | 300 / 300 |
| Same-class IoU >= 0.5 matches | 16 | 275 |
| Unmatched predictions, each side | 0 | 25 |
| Per-class counts identical | Yes | No |
| Matched predictions with changed coordinates | 9 / 16 | 215 / 275 |
| Max absolute matched coordinate delta | 1 px | 203 px |
| Mean absolute matched coordinate delta | 0.171875 px | 4.608182 px |
| Minimum / mean matched IoU | 0.958333 / 0.988222 | 0.504825 / 0.918523 |
| Max absolute matched-pair confidence delta | 0.024318 | 0.359525* |
| Mean absolute matched-pair confidence delta | 0.004965 | 0.008284* |
| Matched predictions changing rank | 3 | 243 |

Confidence deltas use the 0–1 score scale: 0.024318 is 2.4318 percentage points,
not a relative-percent change. Unmatched predictions are unmatched under the
specified class/IoU rule, not ground-truth false positives or missed objects.

**Important matching ambiguity at 4K:** the low-confidence output contains duplicate
and overlapping boxes. Geometry-only assignment does not recover RF-DETR query
identity. The largest 0.359525 confidence delta pairs duplicate class-86 boxes:
Triton indices 9 and 31 have identical integer boxes but scores approximately
0.59843 and 0.24113. The geometry matcher pairs them with SIMD scores 0.24221 and
0.60066 in reversed order to maximize IoU. Thus the starred numbers are properties
of this matching rule, **not evidence that one identifiable object's confidence
changed by 0.36**. The 203 px coordinate example is a low-confidence pair, roughly
0.048 on both sides, matched at IoU 0.637685.

Returned-order comparisons also exaggerate geometric changes when ranking shifts:
max rankwise coordinate differences are 328 px and 2724 px, respectively. Max
rankwise confidence deltas are 0.016157 and 0.018414, but those sorted-score deltas
likewise need not describe the same detection. Both views are kept in the JSON.

At 4K, SIMD-minus-Triton prediction-count changes by class ID are: 44: -1,
46: +1, 47: +3, 62: +2, 64: -3, 67: -3, 79: +1; other class counts are unchanged.

### Additional high-confidence view of the captured 4K outputs

This is a post-capture filter of both existing 300-prediction arrays at confidence
**> 0.4**, followed by the same matching algorithm. It does not rerun inference,
change the requested high scenario, or replace the full results above.

- Triton: **17** predictions; SIMD: **16**.
- **16** same-class matches; one unmatched Triton prediction and none on SIMD.
- Matched box coordinates differ by at most **2 px**, mean absolute 0.171875 px;
  five matched boxes have at least one changed coordinate.
- Matched confidence delta: max absolute **0.017218**, mean absolute **0.003381**.
- Matched IoU: minimum **0.979452**, mean **0.996528**.
- The extra Triton prediction is class 86 at `[1444, 999, 1518, 1076]`, confidence
  **0.4006907344**. Its full-output SIMD match is `[1444, 999, 1518, 1077]`,
  confidence **0.3999288380**, IoU **0.987179**. This is a concrete example of a
  small score difference changing thresholded output membership.

For the base scenario, this diagnostic filter leaves the original 16 predictions
unchanged because its invocation already used confidence 0.4.

## Verification and artifacts

All three provenance records match on image ID, runtime/profiler commits and runtime
import location. Snapshot manifests match on model, input source, scenario and
workload settings; only creation timestamps differ. Both snapshot runs have two
`snapshotted` results, no failures, and no preview warnings. All six native worker
proofs (two memory plus four snapshot) identify the expected preprocessor, fixed
other stages, and no fallback. Metadata covers model selection and the last call
per worker; it is not a per-call trace.

Snapshot runs:

- SIMD: `rfdetr-t4-simd-snapshot-20260918`:
  snapshot-result.json (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-snapshot-20260918/output/snapshot-result.json`),
  640x480 manifest (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-snapshot-20260918/output/model-output-snapshots/c70f32369a54d61e06ef4e6b56c82524/default/camera_640x480_batch_1_base/manifest.json`),
  3840x2160 manifest (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-simd-snapshot-20260918/output/model-output-snapshots/c70f32369a54d61e06ef4e6b56c82524/default/camera_3840x2160_batch_1_high/manifest.json`).
- Triton: `rfdetr-t4-triton-snapshot-20260918`:
  snapshot-result.json (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-snapshot-20260918/output/snapshot-result.json`),
  640x480 manifest (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-snapshot-20260918/output/model-output-snapshots/c70f32369a54d61e06ef4e6b56c82524/default/camera_640x480_batch_1_base/manifest.json`),
  3840x2160 manifest (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-triton-snapshot-20260918/output/model-output-snapshots/c70f32369a54d61e06ef4e6b56c82524/default/camera_3840x2160_batch_1_high/manifest.json`).

Computed deltas, including every matched pair and unmatched indices:

- 640x480 comparison JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-output-comparison-20260918/camera_640x480_batch_1_base.json`).
- 3840x2160 comparison JSON (local artifact: `/private/tmp/rfdetr-execution-plan-runner/run-results/rfdetr-t4-output-comparison-20260918/camera_3840x2160_batch_1_high.json`).

Canonical bundle files are `input.json`, `output.json`, and `arrays/*.npy`.
Each bundle also has exact input PNGs and annotated JPEG previews. Local artifacts
are in the temporary runner copy; original run artifacts remain under
`/home/damiankosowski/remote-runner/runs/<run-id>/` on the T4 host.
