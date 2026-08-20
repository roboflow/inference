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
FPS retention, controlled target-FPS attainment, latency, fairness, startup,
and single-processor placement. This prevents an under-performing concurrency-1
baseline from making the entire controlled-FPS curve appear healthy.

Recovery-tolerant fault runs are summarized separately (event count and observed
control-plane recovery interval) and are deliberately excluded from capacity
certification. This interval starts at the runner's first non-running/requeue
observation, not pod deletion, and ends only after running state plus subsequent
frame progress are verified; it must not be presented as end-to-end media
downtime. Otherwise a successful requeue could hide an unstable worker inside
an apparently passing capacity curve.

For a fault-injection run, `recovery.py` verifies the controller evidence hash
chain and successful outcome, then binds the exact benchmark job, captured
processor pod, replacement processor pod, and runner recovery assertions. It
joins the pre-delete `fault-requested` timestamp to the runner's
`progressVerifiedAt`. The result is labeled as a verified frame-recovery upper
bound rather than exact media downtime. For a relay fault, the same command
binds the old/new relay pods and controller revision to measurement samples. It
requires every job to stay on the same processor and attempt while frames and,
when requested, published outputs advance after replacement readiness. Relay
certification additionally requires a report-bound telemetry artifact whose
pod-identity series match the injected old/new relay UIDs and whose
post-replacement `out-*` path count/ingress and aggregate reader/traffic samples
come from the replacement pod after the one-minute rate window is uncontaminated.
That is still a poll-derived upper bound plus downstream relay-acceptance proof,
not proof of gapless media or workflow-state continuity.

`soak.py` is the fail-closed promotion gate for the checked-in GPU and CPU soak
ladder. It hash-binds the API report, exact matrix/corpus/workflow profiles, and
Prometheus resource artifact; verifies the independently approved cluster
identity against the collector's observed API server and immutable namespace
UID and reopens its hash-bound source file; verifies output watch renewal,
published-counter continuity,
frame-histogram latency tails, restart-counter stability, relay traffic/loss,
and host/relay/VRAM growth and slope; recomputes job/sample coverage; requires
stable processor/attempt
identity, monotonic advancing frame and output counters, terminal cleanup,
relay traffic with zero observed packet-loss/error rate, telemetry coverage,
the expected count of active `out-*` paths with positive ingress, one exact
relay pod UID spanning the full window, and the stage's
memory-growth/slope guardrails. A 1-hour, 4-hour, or 12-hour
certification also reopens and recomputes every shorter stage's raw artifacts.
The checked-in thresholds are initial regression guardrails, not measured
production limits; change them only by changing and re-reviewing the matrix
before starting a ladder.

`recommendations.py` compares strict and explicitly relaxed latency gates for
each homogeneous curve and emits a provisional streams-per-worker class. It
labels right-censored curves, repetitions, failing boundaries, and missing FPS,
resource, soak, output, and cost evidence. It never derives pricing from frame
throughput alone.
