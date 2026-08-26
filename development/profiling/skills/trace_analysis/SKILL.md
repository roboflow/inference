# Nsight trace analysis

Use this skill when a profiling run already contains `manifest.yaml` and
`trace.nsys-rep`. The analysis workflow exports supported `nsys stats` reports
and writes a stable `analysis.json` artifact. It does not rerun the profiling
target.

## Preconditions

Before analyzing a run:

1. Read `manifest.yaml` and identify the target, device, workload settings,
   capture range, record IDs, Git commit, and expected trace path.
2. Confirm that `trace.nsys-rep` belongs to the same run directory.
3. Run `nsys --version`. Use a version that can read the trace and supports the
   `nvtx_pushpop_trace` and `nvtx_gpu_proj_sum` reports.
4. Treat generated traces, SQLite exports, report CSV files, and analysis JSON
   as local artifacts under the ignored `inference_profiling/` directory.

Do not parse `.nsys-rep` directly. Use NVIDIA's supported `nsys stats` reports.

## Generate the analysis

From the repository root, run:

```bash
PYTHONPATH=./ uv run python development/profiling/analyze.py \
  --run-dir RUN_DIRECTORY
```

Replace `RUN_DIRECTORY` with the directory containing `manifest.yaml` and
`trace.nsys-rep`.

The command creates these files:

- `stats/nsys_nvtx_pushpop_trace.csv`
- `stats/nsys_nvtx_gpu_proj_sum.csv`
- `analysis.json`

If the command fails, report the exact Nsight version and error. Check for a
missing executable, an incompatible trace, missing NVTX data, or report-column
changes before modifying the parser.

## Read the result

Read `analysis.json` in this order:

1. Check `provenance` for the manifest, trace, report paths, and Nsight version.
2. Check `run` for the device, Git commit, workload settings, capture range, and
   record count.
3. Read `warnings`. Resolve missing iterations or capture ranges before drawing
   performance conclusions.
4. Use `iteration_summary` to understand timing level and variance across the
   measured passes.
5. Use `iterations` to locate an unusually slow or incomplete pass.
6. Use `host_ranges` for CPU-side orchestration and nested-range costs.
7. Use `gpu_projected_ranges` for GPU work associated with each NVTX range.

All durations use nanoseconds and include the `_ns` suffix.

## Preserve timing semantics

Interpret the fields as follows:

- `host_ranges[].inclusive` measures CPU time between an NVTX push and pop and
  includes nested child ranges.
- `host_ranges[].exclusive` subtracts nested child-range duration from the host
  range.
- `gpu_projected_ranges[].projected` measures the interval covered by GPU
  operations launched from inside that NVTX range.
- `iterations[].host` and `iterations[].gpu_projection` describe different
  clocks and must remain separate.

Do not:

- call host-inclusive or host-exclusive duration GPU latency
- sum inclusive timings from parents and children
- treat GPU projection as the sum of kernel durations
- infer a regression from one run
- compare runs with different targets, inputs, devices, synchronization
  policies, capture ranges, or Nsight versions without stating the difference

## Report findings

When summarizing a run, include:

- profile name, run ID, Git commit, device, and Nsight version
- warmup, iteration count, record count, and synchronization policy
- host-inclusive and GPU-projected iteration statistics with units
- the ranges that account for the largest host and projected-GPU totals
- warnings, missing data, first-iteration effects, or large variance
- the exact `analysis.json` path

Describe measurements before suggesting optimizations. If a range is slow only
on the host, inspect preparation, Python orchestration, synchronization, or I/O.
If projected GPU time is high, inspect its associated kernels in the Nsight UI
or add a kernel-specific report in a separate analysis step.
