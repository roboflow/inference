# Semantic segmentation post-processing benchmarks

Developer-only scripts backing the measurements in
[#2728](https://github.com/roboflow/inference/pull/2728) (present_class_ids
hint + single F-order conversion) and
[#2729](https://github.com/roboflow/inference/pull/2729) (in-process numpy
mask fast path). Not part of the shipped package, mirroring the convention of
`development/profiling/`.

Both scripts import the production block
(`RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections`) from
the checked-out tree and inline the pre-optimization implementation verbatim
as the baseline/reference, so they run on any branch: modes a tree does not
support (e.g. numpy masks before #2729) are detected and skipped, never
silently substituted.

## `benchmark_postprocessing.py`

Per-frame latency of the two optimized stages, with mean / p50 / p95 / p99:

- `model_side` — producing the response payload: PNG+base64 encode of both
  full-resolution masks (base64 transport) or just the `present_class_ids`
  bincount (numpy transport).
- `convert` — the workflow block's `_convert_to_sv_detections`.

```bash
# from the repo root, with inference installed (or PYTHONPATH=.)
python development/benchmarks/semantic_segmentation/benchmark_postprocessing.py \
    --iterations 100 --json /tmp/semseg_bench.json

# memory: per-mode subprocesses report tracemalloc peak + RSS
python development/benchmarks/semantic_segmentation/benchmark_postprocessing.py --memory
```

Defaults synthesize a 5320x3032 (16.1 MP) uint8 label map with 4 blobby
foreground classes plus a smooth confidence field — the shape of the camera
workload that motivated the PRs (the report records exact input metadata,
including PNG payload sizes, so runs are comparable). `--label-map` /
`--confidence-map` accept a grayscale PNG or `.npy` to measure real frames.
`--height/--width/--foreground-classes/--seed` vary the synthetic input.

The stdout table is markdown-ready; `--json` adds raw per-iteration samples,
environment/library versions, and the memory table.

Scope note: this measures the post-processing path in isolation. The
end-to-end on-device tables in the PR descriptions come from the
`SEMSEG_TIMING` stage profiler running inside the full video pipeline on a
Jetson AGX Orin; expect different absolute numbers on different CPUs, with
the same shape (PNG round-trip and full-frame scans dominating the baseline).

## `check_output_parity.py`

Byte-identical output check of the optimized conversion against the inlined
pre-optimization reference, on a full 16.1 MP map by default (`--quick` for
a small one):

- class ids/names, exact xyxy, RLE `counts` byte equality + `size` fields,
  exact float32 confidences, `confidence_mask` byte equality
- across transports and hint states: base64 no-hint, base64 hint, numpy
  hint, numpy no-hint (numpy rows auto-skip on trees without #2729)
- edge cases: all-background, background-only hint, stale hint id, empty
  hint fallback; plus `present_class_ids_from_label_map` vs `np.unique`
  when torch is available

```bash
python development/benchmarks/semantic_segmentation/check_output_parity.py
echo $?  # non-zero on any mismatch
```

The committed unit tests cover the same properties on small inputs (see
`tests/workflows/unit_tests/core_steps/models/roboflow/semantic_segmentation/`);
this script exists to run them at full production resolution and across
branch/mode combinations from one command.
