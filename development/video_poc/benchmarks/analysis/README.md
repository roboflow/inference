# Workflow corpus report analysis

Analyze one or more JSON reports written by `run_api_workflow_corpus.py`:

```bash
python development/video_poc/benchmarks/analysis/report.py \
  development/video_poc/benchmarks/results --format markdown
```

The analyzer derives delivered FPS from frame-counter deltas rather than the
worker's `stats.fps`. The latter describes workflow processing time and can be
higher than the source frame rate. It discards the first 10 seconds of the
measurement phase by default.

`decodeToResultLatencyMs` is a rolling EMA sampled by the runner. Consequently,
the reported latency p50/p95 values are percentiles of sampled EMAs, not true
per-frame latency percentiles. True latency SLOs require processor histograms or
per-frame observations to be attached to the report.

Capacity summaries are only generated for homogeneous runs (one profile, tier,
mode, output setting, and `maxFps` setting). Controlled-FPS and unbounded runs
are never combined into one curve. The lowest successful tested concurrency is
used as the FPS/latency baseline. A run is certified only when it passes every
configured gate, including successful execution, steady-state sample coverage,
FPS retention, latency, fairness, startup, and single-processor placement.
