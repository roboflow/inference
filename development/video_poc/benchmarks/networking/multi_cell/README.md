# Multi-cell staging campaign

This directory turns Track C of
[`MULTI_CELL_SCALING_RFC.md`](../../../MULTI_CELL_SCALING_RFC.md) into a
fail-closed, paired experiment contract. It does not deploy a second cell or
change the control plane. It renders one bounded publisher or probe Job at a
time and certifies only complete staging evidence.

The committed JSON is intentionally a **non-runnable template**. Its snapshot
identities illustrate the required shape; they are not claims about the live
clusters. `render_campaign.py` refuses it until an operator captures a fresh,
six-hour-or-shorter `identitySnapshot` and replaces every cell, relay, node,
network, LoadBalancer, connector, source-placement, policy, processor, replica,
fixture, and image identity from staging.

No command in this directory reads or writes a cluster, cloud API, video API,
DNS, or production environment. Rendering and analysis operate on files only.

## What is compared

Each pair uses at least three repetitions and alternates AB/BA order:

| Arm | Connector | Source relay | Processor/probe | Purpose |
|---|---|---|---|---|
| `same-cell` | home cell | home cell | home cell | paired control |
| `cross-cell-ingest` | remote cell | home cell | home cell | connector-to-relay WAN cost |
| `cross-cell-direct` | home cell | home cell | remote cell, direct origin read | per-reader WAN cost |
| `cross-cell-replica` | home cell | home cell | remote cell, local replica read | one-WAN-copy fan-out shape |
| `cross-cell-ingest-and-execution` | remote cell | home cell | another allowed cell | later combined path |

The source home cell and relay shard never change inside a pair. Remote
execution is valid only when the snapshotted workspace policy explicitly
allows the execution cell. A dedicated policy must have `allowSharedFallback`
false. Separate placement cases prove sticky preview/job reactivation and prove
that an unauthorized dedicated-cell request produced neither an execution cell
nor a job claim.

## Why the evidence contract is strict

A node selector, Deployment name, image tag, or requested `cell` label is not
placement evidence. The campaign binds and later revalidates:

- cluster UID/config hash and Cilium/network revision;
- node name, node UID, instance type, NIC/network identity, and documented VPC
  ceiling for connector, relay, and reader nodes;
- MediaMTX pod UID/revision, config hash, image digest, shard, LB ID/config hash,
  and a provider- or experiment-derived LB bandwidth ceiling;
- source record revision, home cell, relay shard, placement generation, and
  connector build/session identity;
- workspace placement-policy revision/hash;
- processor and optional read-replica runtime identities; and
- processor workflow/runtime-input hashes and first-attempt claim evidence; and
- probe image digest plus the CBR pixel-clock fixture identity.

The analyzer rejects changed or absent identities, partial run/repetition/action
coverage, counter resets, telemetry that does not cover the measurement window,
or a run outside the identity snapshot validity window. Every range result also
binds the canonical query hash and a run-specific stream-scope hash, preventing
an aggregate cell series from being substituted for the planned stream/path
evidence. LB and VM counters are necessarily resource aggregates; the renderer
therefore requires dedicated `exclusiveBenchmarkCell` resources and serial
run/return-to-baseline barriers before those ranges are attributable to a run.
The evidence JSON is only an index/normalized view: certification also opens and
hashes the raw publisher/probe termination reports, processor and placement API
responses, and every Prometheus/Crusoe range response. Missing, changed,
path-escaping, wrong-kind, or normalized-vs-raw disagreement fails closed.

## Authoritative measurements

`multi_cell_probe.py` generates a zero-B-frame H.264 CBR pixel-clock stream. A
PyAV reader records each decoded frame's monotonic arrival, decodes the embedded
publisher wall clock, and sums encoded packet sizes. The controller must join
that report with synchronized-clock and range-query evidence:

- TTFF: first decoded-frame monotonic time minus connection-start monotonic time;
- stalls: derived from every consecutive decoded-frame arrival, not log text;
- delivered FPS: decoded arrivals divided by the declared measurement window;
- encoded throughput: PyAV packet sizes divided by that window;
- latency: per-frame `pixel-clock-v1` values, accepted only with before/after
  chrony/NTP/PTP uncertainty within the campaign threshold;
- loss/errors: MediaMTX RTP received/lost/input-error counter deltas;
- ingress/egress: MediaMTX bytes and Crusoe LB bytes/peak bit rates;
- VPC pressure: separate Crusoe VM ranges for connector, relay, and reader nodes;
  and
- network health: separate connector-, relay-, and reader-node drop and TCP
  retransmit deltas from Prometheus.

The current Crusoe public documentation does not state an LB bandwidth ceiling.
Therefore `bandwidthCeilingBps` must cite either a provider-confirmed value or a
prior staging calibration in `limitSource`; it must not be inferred from the
virtual NIC link-speed metric. A measured ceiling is a staging test boundary,
not a provider guarantee.

## Snapshot and render

1. Copy `multi-cell.staging.template.json` outside the repository.
2. Using **staging contexts only**, capture all identities above. Hash canonical
   exported objects/configuration rather than terminal formatting. Record a
   snapshot expiry no more than six hours after capture.
3. Pin the exact probe and MediaMTX image digests. Create short-lived,
   run-prefix-scoped media Secret keys; never paste URL values into JSON.
4. Set `identitySnapshot.status` to `captured`. Keep its collector identity and
   canonical source hash in the run artifact.
5. Render without contacting either cluster:

```bash
python development/video_poc/benchmarks/networking/multi_cell/render_campaign.py \
  --campaign /secure/path/multi-cell.staging.json \
  --execution-id stg-two-cell-20260813-a7f91c \
  --output-dir /secure/path/results/stg-two-cell-baseline-001
```

The output contains:

- `campaign-manifest.json`: immutable identities, counterbalanced sequence,
  thresholds, readiness barriers, exact per-run apply commands, and cleanup;
- one single-Job JSON file per run/role/context, so applying a directory cannot
  accidentally launch all paired repetitions concurrently; and
- `evidence-template.json`: complete run/action coverage marked
  `not-collected`, which the analyzer refuses to certify.

The execution ID must be globally unique for the staging campaign window. It is
part of every run/job/file/label, and cleanup selects it rather than the reusable
campaign name, so concurrent operators cannot overwrite or delete one another's
Jobs. Do not bulk-apply the output directory. A controller or operator executes the
manifest sequence serially: publisher, readiness barrier, create the real
processor job, verify its exact claim/node/cell and frame progress, start the
independent path probe, collect, cleanup, return-to-baseline, next arm. The
probe intentionally adds one measured reader; processor-only capacity curves
must use the processor campaign instead of treating this diagnostic topology as
zero overhead. Immediately before each write the controller must
revalidate context API endpoint/cluster UID, namespace, node UID, workload UID,
source placement, policy, and resolved image ID. Any mismatch aborts the
campaign before the next Job.

The readiness and processor-claim work must finish within
`startupGraceSeconds`. The publisher itself runs for that entire grace plus the
probe warmup and measurement windows and a bounded media tail of at most five
seconds. Kubernetes adds a larger, separate `shutdownMarginSeconds` so ffmpeg
can exit and the final report can be flushed before the hard deadline.

## Readiness and stop gates

The readiness barrier requires the publisher on its exact node, the expected
path on the exact relay workload, unchanged source placement, and baseline
samples from Prometheus and Crusoe. The operator then starts the reader and
enforces the manifest thresholds. Abort and clean up the current run on:

- any identity or placement drift;
- publisher/probe failure, missing frame samples, reconnect, or counter reset;
- TTFF, stall, FPS, latency, loss, input-error, node-drop, VPC, or LB breach;
- missing/stale metrics or excessive clock uncertainty; or
- a credential/snapshot expiry.

The manifest uses `backoffLimit: 0`, an active deadline, TTL cleanup, non-root
containers, dropped capabilities, and a read-only root filesystem. Those are
last-resort bounds, not a replacement for run-scoped cleanup and credential
revocation.

## Analyze

After filling observed placement, agent samples, placement-case actions, and
metric ranges, mark each fully collected item `observationStatus: collected`:

```bash
python development/video_poc/benchmarks/networking/multi_cell/analyze_campaign.py \
  --manifest /secure/path/results/stg-two-cell-baseline-001/campaign-manifest.json \
  --evidence /secure/path/results/stg-two-cell-baseline-001/evidence.json \
  --artifact-root /secure/path/results/stg-two-cell-baseline-001/raw \
  --output /secure/path/results/stg-two-cell-baseline-001/report.json
```

Exit code 0 means every absolute gate and the paired latency-regression gate
passed. Exit code 2 means complete evidence was analyzed but at least one SLO
failed. Invalid or incomplete evidence is an error and produces no
certification.

Certification is deliberately narrow: the exact fixture, identities,
topologies, load, and staging window in the report. Capacity ramps repeat the
same paired contract at increasing source/readers/bitrate levels; a one-hour and
then 24-hour soak are still required before recommending an operating limit.

## Implementation boundary

The transport harness can run once a second staging benchmark cell, metrics,
Secrets, and immutable probe image exist. The source/dedicated placement cases
also require the RFC Phase 1 cell-aware contracts: persisted home cell/shard,
generation-aware connector commands, job execution cell, and claim filtering by
cell. Until that lands, those cases should fail closed rather than being marked
`collected` from requested fields.
