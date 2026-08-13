# RFC: Multi-cell video processing and workload-aware capacity

**Status:** Draft

**Last updated:** 2026-08-13

**Scope:** Live connector sources, MediaMTX relay capacity, processor placement,
multi-stream GPU allocation, dedicated cells, and remote execution experiments

**Related:**
[HANDOFF.md](https://github.com/roboflow/inference/blob/hansent/video-poc/development/video_poc/HANDOFF.md),
[DEPLOY_PLAN_STAGING.md](https://github.com/roboflow/inference/blob/hansent/video-poc/development/video_poc/DEPLOY_PLAN_STAGING.md), and
[README.md](https://github.com/roboflow/inference/blob/hansent/video-poc/development/video_poc/README.md)

## Summary

The first production video-processing cell is live in Crusoe US East. It proves
the intended cell shape: an outbound-only connector publishes an encoded source
stream to MediaMTX, and colocated CPU/GPU processor pools fan out from that relay.
The next phase is to determine its safe capacity and make placement explicit so
the same architecture can support:

- multiple shared cells in different regions or providers;
- workspace-dedicated cells and reserved processor pools;
- more than four light streams per GPU when measurements support it;
- heavy workflows that require most or all of a GPU;
- deliberate experiments where media and processing are in different locations;
- eventually, customer-managed or on-premises cells.

The central proposal is:

> A live source is lazily assigned to a sticky **home cell** when it first needs
> to stream. Jobs normally execute in that same cell. Remote execution is an
> explicit placement mode, not an accidental outcome of a global queue.

This RFC intentionally starts with instrumentation and benchmarks. Relay shard
size, per-GPU concurrency, workload weights, and operating headroom must be
derived from measured capacity curves rather than fixed in the design.

## Status of the current implementation

As of 2026-08-10, production is intentionally a single cell:

- `crusoe-use1` runs one MediaMTX relay plus ready GPU and CPU pools.
- MediaMTX is a single `Recreate` Deployment. Multiple replicas behind the
  existing Service would be incorrect because a publisher and its readers can
  land on different pods.
- A connector receives a full ingest URL for each `start_stream` command. Video
  flows only while a preview or active job needs the source.
- Functions construct URLs from one global set of `VIDEO_PROC_*` variables.
- Pub/Sub wake-ups carry one global `cell` attribute and subscriptions are
  filtered by it, but the transactional claim query does not filter by cell.
- Fleet processors claim across workspaces and filter only by CPU/GPU tier.
- The claim payload gives a processor a cluster-local MediaMTX URL, which assumes
  the worker and source relay are colocated.
- Production workers accept up to four jobs. Admission is job-count based;
  workflow cost and workspace fairness are not considered.
- Processor Prometheus metrics expose active-job count and configured capacity.
  Draft infra [#2443](https://github.com/roboflow/roboflow-infra/pull/2443)
  adds internal-only MediaMTX scraping and a first relay dashboard; it is not
  active until the chart is merged and applied. pprof remains disabled.

Phase 0 implementation has also started in draft inference
[#2616](https://github.com/roboflow/inference/pull/2616): the processor exports
bounded aggregate capacity, lifecycle, frame, publisher, and latency metrics;
`benchmarks/` contains a reproducible relay harness and a provisional workflow
corpus. These are measurement tools, not certified capacity settings. No relay or
processor limit should change until baseline curves and SLOs exist.

These are safe assumptions for one cell. They are unsafe for two: a processor in
cell B could claim a job whose source only exists on the relay in cell A, then
attempt to read it from cell B's cluster-local MediaMTX service.

## Goals

1. Preserve media/compute locality by default.
2. Make every placement decision durable, inspectable, and enforceable at claim.
3. Scale relay origins without relying on random load balancing of stateful
   stream paths.
4. Admit multiple streams per processor according to measured workload cost.
5. Provide workspace fairness and explicit dedicated-capacity policies.
6. Quantify the performance and cost of non-colocated media and processing.
7. Roll out cell awareness without changing behavior for the existing East cell.
8. Keep connector networking outbound-only.

## Non-goals for the first iteration

- Seamless migration of an actively streaming source between cells.
- Live migration of tracker/counter state between workers.
- Automatic per-block remote execution inside the Workflows engine.
- Selecting a final WAN media protocol before testing the current RTSP/TCP path.
- Designing recording retention and historical playback.
- Replacing Firestore job ownership or the ready-pool lifecycle.

## Terminology

**Cell**
: A deployable media-processing location with cell-specific endpoints, one or
  more relay shards, CPU/GPU processor pools, dispatch subscriptions, monitoring,
  and a declared region/provider.

**Relay shard**
: One stateful MediaMTX origin responsible for a disjoint set of source and output
  paths. A shard can have read replicas, but a source has one origin at a time.

**Home cell**
: The cell that receives a live connector source. It is selected lazily on first
  activation, persisted on the source, and kept sticky across idle periods.

**Execution cell**
: The cell in which a job's processor runs. For live sources this defaults to the
  home cell. Uploaded-file jobs can be placed independently because their input
  comes from object storage, not a connector relay.

**Placement generation**
: A monotonically increasing source value that distinguishes the current ingest
  assignment from stale connector commands or reports.

**Workload profile**
: A versioned estimate of the resources a workflow consumes at a declared input
  resolution and target FPS.

## Placement invariants

1. A live source has at most one active relay origin.
2. Preview and job creation racing on an unassigned source select one home cell
   transactionally.
3. All ordinary jobs for a live source inherit its home cell.
4. A worker can only claim jobs assigned to its own cell and pool/tier.
5. Pub/Sub cell filters and Firestore claim filters must agree; Pub/Sub is only a
   wake-up mechanism, while Firestore remains the source of truth.
6. Cell-specific URLs are resolved from persisted placement, never process-global
   defaults after the migration period.
7. Active placement is sticky. The first implementation only reassigns an idle
   source, except for an operator-directed recovery.
8. Cross-cell execution must be represented on the job and authorized by the
   workspace's placement policy.

## Proposed architecture

```text
                                     global control plane
                         cell registry · placement · job ownership
                                        /             \
                                       /               \
customer connector                   v                 v
  cameras ── encoded push ──> cell A relay ──> cell A processors
                                  │
                                  └── optional encoded WAN copy ──> cell B proxy
                                                                      │
                                                                      └─> cell B processors
```

The normal path stays entirely in one cell. If several processors in a remote
cell need the same source, cell B should pull one encoded copy into a local proxy
or read replica and fan out locally instead of opening one WAN reader per job.

Decoded frames must not become the cross-cell transport. Encoded video is orders
of magnitude smaller, and decode/pre/post-processing should remain close to the
source unless a specific Workflows capability deliberately externalizes a model
forward pass.

## Source placement lifecycle

### Registration

Connector healthchecks continue to register source metadata without sending
video. A newly registered source has no committed home cell. The workspace or
connector may have a preferred cell, but this is policy input rather than active
placement.

### First activation

A preview request or first live job performs a placement transaction:

1. Read the source and workspace placement policy.
2. If the source already has an eligible home cell, reuse it.
3. Otherwise select an eligible cell and relay shard.
4. Persist `homeCell`, `relayShard`, `placementGeneration`, and the decision
   reason.
5. Build ingest and WHEP URLs from that placement.
6. Reconcile the connector into publishing to the selected ingest URL.
7. Assign jobs to the same cell unless remote execution was explicitly requested.

### While active

Placement is locked while any preview lease or active job depends on the source.
Additional workflows reuse the one source publication and become additional relay
readers in the same cell.

### Idle and reactivation

When the final preview/job lease ends, the connector stops publishing as it does
today. The home cell remains as a sticky preference. On later activation it is
reused if healthy and eligible; it may be replaced while idle for capacity,
policy, residency, or maintenance reasons.

### Migration

The first supported migration is idle-source reassignment. Active migration
requires a generation-aware handshake:

1. stop or fence the old publisher;
2. increment the placement generation and issue the new ingest command;
3. confirm the connector is publishing the new generation;
4. re-place or reconnect jobs;
5. retire the old relay path.

Seamless active migration additionally depends on externalizable state for
stateful Workflow blocks. Until that exists, an active migration can restart the
workflow but cannot promise continuity of trackers or counters.

## Workspace and connector policies

Placement is persisted per source, but workspace and connector policies constrain
and bias selection:

```json
{
  "mode": "shared | dedicated | customer-managed",
  "preferredCells": ["crusoe-use1"],
  "allowedCells": ["crusoe-use1", "crusoe-ussc1"],
  "allowRemoteExecution": false,
  "allowSharedFallback": false,
  "residency": ["US"],
  "reservedCapacity": { "gpuUnits": 100 }
}
```

- Sources from one connector should normally share a preferred cell to keep
  customer networking predictable.
- Per-source placement allows large connectors to span relay shards or cells.
- A strict dedicated workspace has one or more allowed dedicated cells and no
  silent shared fallback.
- A dedicated processor pool may share a relay when compute isolation is enough.
- A full dedicated cell has its own relay, processor pools, endpoints, and
  capacity telemetry.
- A customer-managed/on-prem cell uses the same placement and job contracts; it
  is not a separate product architecture.

## Cell registry and data model

The control plane needs an authoritative cell registry. It can start as deployed
configuration, but callers should consume it through one interface so it can later
become dynamic.

```json
{
  "id": "crusoe-use1",
  "provider": "crusoe",
  "region": "us-east1",
  "environment": "production",
  "state": "accepting | draining | unavailable",
  "endpoints": {
    "rtspIngest": "rtsp://video-ingest...:8554",
    "rtspConsumeExternal": "rtsps://...",
    "whep": "https://video-relay...",
    "processorGateway": "https://video-processors..."
  },
  "capabilities": {
    "tiers": ["gpu", "cpu"],
    "remoteSourceRead": false
  }
}
```

Proposed additions:

- `video_sources`: `homeCell`, `relayShard`, `placementGeneration`,
  `placementReason`, `placementUpdatedAt`.
- `video_jobs`: `executionCell`, `sourceCell`, `remoteExecution`,
  `workloadProfileId`, `reservedResources`.
- processor claim: `cell`, `tier`, worker capacity/free resources, cached model
  keys, and eventually a worker generation/image version.
- connector stream report/commands: placement generation so stale commands cannot
  revive an old path after migration.

The initial migration backfills or interprets missing placement as the configured
default cell (`crusoe-use1`). The cell-aware code should run in production with
only that cell registered before a second cell is allowed to claim work.

## Cell selection

Version 1 should be intentionally simple and explainable:

1. Filter cells by environment, health, workspace allowlist, residency, and
   required tier/capabilities.
2. Reuse an eligible sticky source assignment.
3. Prefer a workspace/connector preferred cell.
4. Reject cells above a conservative relay or processor admission threshold.
5. Choose the lowest-cost eligible cell using coarse relay headroom, queue delay,
   and available compute.
6. Persist the decision and a machine-readable reason.

Connector-to-cell RTT and upload probes can later inform the preference, but
placement must not oscillate in response to noisy measurements. Hysteresis and
minimum assignment age are required before automatic rebalancing.

## Relay scaling and bandwidth accounting

MediaMTX routes encoded media without transcoding, so the first expected capacity
limit is network throughput. For source bitrate `B`, `N` processing readers, `V`
source-preview viewers, and watched output bitrates `O[i]`:

```text
source external ingress      ~= B
source relay internal egress ~= N * B
source preview egress        ~= V * B
output internal ingress      ~= sum(O[i])
output viewer egress         ~= sum(O[i] * viewers[i])
```

At 100 sources × 5 Mb/s × four workflows, connector ingress is roughly 0.5 Gb/s
but relay-to-processor traffic is roughly 2 Gb/s. If all four 5 Mb/s outputs are
also watched, the relay sees another roughly 2 Gb/s in and 2 Gb/s out, before
protocol overhead. Public LB ingress, node NICs, CNI/east-west traffic, and WHEP
egress must be measured separately.

Scaling patterns:

- **Do not** increase identical MediaMTX origin replicas behind the existing
  Service. Stream paths are stateful.
- **Shard origins by source.** Persist the shard and issue shard-specific ingest
  and WHEP routing.
- **Use read replicas/proxies for fan-out.** They help with many viewers or with
  one source consumed by multiple processors in another cell, but the origin-to-
  replica link can itself become the bottleneck.
- **Keep the relay transcode-free by default.** Connector-side target FPS,
  resolution, or camera substream selection is preferable when a workflow does
  not need the full source rate.

MediaMTX documents bandwidth as the common scaling bottleneck and supports
Prometheus byte/path/session metrics, pprof, and read-replica topologies:

- https://mediamtx.org/docs/features/scalability
- https://mediamtx.org/docs/features/metrics
- https://mediamtx.org/docs/features/performance

## Workload-aware processor admission

### Current behavior

`MAX_CONCURRENT_JOBS` is a hard count. A fresh ready worker briefly yields claim
races so jobs pack into a partially filled process, allowing shared model loads.
This is useful, but four light workflows and four heavy workflows are treated as
equivalent. GPU compute is not the only shared resource: decode, preprocessing,
tracking, visualization, output encoding, CPU, RAM, and VRAM can each be limiting.

### Workload profiles

A profile is keyed by a canonical workflow fingerprint plus runtime inputs such as
resolution, target FPS, tier, model versions, and output-publishing mode. It should
record at least:

- expected delivered FPS and latency at concurrency one;
- normalized GPU compute units;
- peak VRAM and host memory;
- CPU/decode/encode demand;
- cached model keys and load cost;
- whether the workflow is stateful;
- confidence and provenance of the estimate.

The first implementation can use conservative classes:

| Class | Example admission behavior |
|---|---|
| Light | Several per GPU if the measured mix remains within SLO |
| Medium | Two or a small number per GPU |
| Heavy | One per GPU by default |
| Exclusive | Entire GPU/worker reserved |

The class values remain TBD until benchmarks exist. `maxJobs` stays as a hard
safety ceiling, not the scheduler's primary capacity model.

### Multidimensional admission

Each worker exposes a resource budget and current reservations. A job fits only if
adding its reservation leaves headroom in every relevant dimension. Model affinity
can break ties, but must not override safety or workspace fairness.

When no worker safely fits a job, the job queues and the ready pool grows if
capacity is available. Existing healthy streams are protected from new work.
Preemption or live migration should not be required for the first scheduler.

### Workspace fairness

Global FIFO is insufficient because one workspace can occupy every available
slot. Candidate selection should evolve to weighted fair queuing:

- per-workspace concurrent resource-unit limits;
- reserved units for dedicated or contracted capacity;
- fair selection among workspaces before oldest-job selection within one
  workspace;
- explicit priority classes for interactive versus background work;
- no silent spill from strict dedicated cells into shared capacity.

Fairness here is admission fairness, not guaranteed GPU time slicing. Runtime
telemetry remains necessary because colocated kernels and CPU work can interfere
even when static estimates fit.

### Runtime feedback

Workers should stop accepting jobs before an SLO breach, based on reserved capacity
and coarse health signals. If an admitted workload is more expensive than its
profile predicted:

1. mark the profile observation;
2. stop further admission to that worker;
3. preserve existing streams where possible;
4. queue or re-place new work;
5. only reduce FPS when the workflow explicitly declares an elastic target range.

## Observability required before scaling decisions

### Relay and network

- active paths, publishers, readers, and protocol/session counts;
- per-path and aggregate ingress/egress bitrate derived from byte counters;
- RTSP packet errors/discards and WebRTC loss/jitter/discards;
- MediaMTX CPU, heap, goroutines, file descriptors, and restarts;
- LoadBalancer bytes/connections/errors;
- relay-node NIC saturation, drops, retransmits, conntrack pressure, and CNI
  east-west bandwidth;
- external versus internal versus inter-cell bandwidth and estimated egress cost.

Draft infra [#2443](https://github.com/roboflow/roboflow-infra/pull/2443)
implements the first part: metrics are exposed only on the internal Service,
scraped by a PodMonitor with a bounded allowlist, and visualized in a Grafana
dashboard. pprof should remain internal and be enabled only for controlled
performance work.

### Processor and job

- queued-to-claim and claim-to-first-result latency;
- delivered FPS, frames processed, source FPS, and frame drops;
- decode-to-result p50/p95/p99 latency;
- per-pod active jobs, reservations, tier, and model-cache affinity;
- GPU utilization/VRAM, CPU utilization/throttling, RAM, and network;
- output publisher state and bitrate;
- workflow fingerprint/profile and job failure/requeue reason.

GPU metrics are naturally pod-level. Controlled mixed-workload experiments are
needed to attribute interference to individual jobs.

### Cell

- eligible/accepting/draining state;
- relay headroom;
- ready and working processors by tier;
- reserved and free resource units;
- queue depth and oldest age by workspace/tier;
- placement decisions, reasons, and rejected-cell reasons.

## Benchmark program

Benchmarks run in staging or a dedicated performance cell before production load
tests. Synthetic publishers should replay pre-encoded media with `-re -c copy` so
the load generator's encoder is not the bottleneck. Readers should run outside the
relay pod and consume/discard frames while recording timing and continuity.

### Track A: MediaMTX and cell network capacity

Axes:

- 720p, 1080p, and representative 4K streams;
- representative 1/5/10/20 Mb/s bitrates and GOP sizes;
- increasing source count;
- 1/2/4/8 readers per source;
- no preview, source WHEP preview, and multiple watched output streams;
- publishers outside the cluster and readers on processor nodes;
- 10-15 minute ramps followed by 1-hour and 24-hour soaks.

Record the first bottleneck and the point at which latency, errors, disconnects,
or resource saturation exceed the agreed SLO. Test the Crusoe LB, relay pod/node,
CNI, and WHEP media path independently where possible.

### Track B: Multi-stream processor packing

Axes:

- light/medium/heavy/exclusive workflow corpus;
- same model versus different models;
- 640/720/1080 input and 5/15/30 target FPS;
- concurrency 1/2/4/6/8/12;
- output publishing off versus watched;
- homogeneous mixes and adversarial mixes such as three light streams followed
  by one heavy stream;
- CPU and GPU tiers.

Measure each stream independently. A total-throughput improvement is not acceptable
if one stream's latency or delivered FPS collapses.

### Track C: Cross-cell and WAN execution

Compare:

1. relay and processor colocated;
2. East relay with a direct remote processor reader;
3. East relay with one remote-cell proxy and local processor fan-out;
4. GCP relay with Crusoe processing;
5. cloud relay with customer/on-prem processing;
6. a complete on-prem mini-cell with only control traffic leaving the site.

Sweep RTT, jitter, loss, and available bandwidth. A useful impairment matrix starts
around 20/50/100/200 ms RTT, 0/0.1/0.5/1/2% loss, and bandwidth caps close to and
above the encoded source rate. Compare RTSP/TCP with secure WAN transport candidates
before approving a production remote path.

### Track D: Failure and recovery

- processor crash/OOM with several colocated jobs;
- GPU node reclaim or drain;
- relay restart;
- connector disconnect/reconnect;
- Pub/Sub delay or duplicate delivery;
- cell marked draining or unavailable;
- inter-cell partition;
- idle-source reassignment;
- operator-directed active-source migration.

Measure interruption, duplicate work, job attempts, stale publisher behavior, and
state loss. Stateful workflows require separate continuity tests.

### Capacity certification

For each tested workload, define:

- the load where the first agreed SLO fails (the capacity knee);
- a lower certified operating limit with substantial headroom;
- the limiting resource and required alerts;
- the software, instance, node, network, codec, and workflow versions;
- confidence based on ramp and soak duration.

The initial headroom factor is a decision to make after the first curves; operating
at roughly one-half to two-thirds of the measured knee is a reasonable hypothesis,
not yet an approved target.

## Rollout plan

### Implementation checkpoint (2026-08-13)

The summary invariants above were checked against the current code rather than
treated as an implementation specification. Registration was already
metadata-only and the connector already consumed a complete server-issued ingest
URL. The minimum safe change was therefore to make placement, identity, relay
authorization, URL resolution, and claims explicit—not to add cell selection to
the connector.

Phase 1 code is in control-plane draft
[roboflow#14444](https://github.com/roboflow/roboflow/pull/14444) and
processor/harness draft
[inference#2793](https://github.com/roboflow/inference/pull/2793), but is not
deployed. It uses
an East-only explicit registry first, transactional first activation, persisted
placement metadata, generation-aware connector reports/commands, per-cell fleet
credentials, a deployment-fixed relay auth cell, transactional cell+tier claim
filters, wrong-cell processor rejection, legacy default-cell interpretation, and
a bounded scheduled reaper. Emulator-backed tests exercise simultaneous preview
and job activation against two eligible test cells and prove one committed source
assignment; local execution still depends on the repository's current Node
dependency overlay being available.

Phase 2 infrastructure is prepared in draft
[roboflow-infra#2460](https://github.com/roboflow/roboflow-infra/pull/2460), not
applied. Stacked staging-config draft
[roboflow#14445](https://github.com/roboflow/roboflow/pull/14445) registers the
South cell as `unavailable` and binds only its cell-specific secret name after
that infra prerequisite. The chosen cell is `crusoe-ussc1` on
`ck8s-stg-us-southcentral1`. Its own stack begins
`unavailable` with both processor pools disabled, non-runnable image placeholders,
cell-specific credentials/subscription, fixed c1a.16x relay/gateway placement,
fixed processor node classes, `.one` endpoints, metrics/dashboard coverage, and
a default-off RTSPS listener for an explicitly authorized South-origin WAN read.
The production Helm render is byte-for-byte unchanged. The East render changes
only by adding immutable processor identity; the active L40S capacity settings
and source are outside this workstream.

The two-cell validation renderer is staging-only and offline. It emits separate
East/South Job lists and requires actual pod/node/cluster/cell, connector route,
preview, claim, MediaMTX session/reader, processor, latency/loss/bandwidth/egress,
and failure/recovery evidence. Its cross-cell case is South origin to an East
CPU reader; it never mutates the East relay or schedules onto L40S. Applying the
cell, DNS, functions, benchmark prerequisites/Jobs, network impairment, or
failure experiment still requires the documented staging approval.

### Phase 0: RFC, instrumentation, and baseline

- agree on terminology, invariants, and SLOs;
- merge and apply the internal MediaMTX metrics/dashboard from infra
  [#2443](https://github.com/roboflow/roboflow-infra/pull/2443), staging first;
- validate the aggregate processor metrics in inference
  [#2616](https://github.com/roboflow/inference/pull/2616);
- run the synthetic relay harness and provisional workflow corpus in
  [`benchmarks/`](benchmarks/), then version the resulting environment manifests;
- measure the current East topology.

**Gate:** relay and processor capacity curves exist and identify the first
bottlenecks.

### Phase 1: Cell-aware contracts with one cell

- introduce the cell registry;
- persist source and job placement;
- resolve URLs from placement;
- require processors to identify their cell;
- filter claims transactionally by cell and tier;
- align Pub/Sub attributes/subscriptions with job placement;
- treat missing legacy placement as `crusoe-use1` during migration.

**Gate:** all existing behavior works with one registered cell, and a synthetic
worker from another cell cannot claim East work.

**Current status:** implemented and locally validated; review, Firestore index,
East-only functions rollout, East processor identity rollout, and live synthetic
claim-isolation evidence remain.

### Phase 2: Second non-production cell

- deploy a second staging/performance cell;
- test connector placement, preview, and processing in both cells;
- validate strict workspace pinning and dedicated cell policy;
- test cell draining, idle reassignment, and failure behavior;
- measure East-to-second-cell remote execution without making it the default.

**Gate:** deterministic two-cell operation with no cross-cell misclaims or missing
local streams.

**Current status:** repeatable South cell and deterministic validation contracts
are prepared with processors default-off. No South resource/DNS/function binding
or benchmark Job has been applied, and no two-cell validation result is claimed.

### Phase 3: Workload-aware admission and fairness

- profile the benchmark workflow corpus;
- introduce conservative workload classes/resource units;
- add worker fit checks, model affinity, and workspace fairness;
- test concurrency above four for light workloads;
- add runtime feedback without preempting healthy streams.

**Gate:** mixed workloads remain within SLO and one workspace/heavy workflow cannot
materially degrade unrelated streams.

### Phase 4: Relay sharding and remote/dedicated placements

- shard origins when measured relay capacity requires it;
- test read replicas/proxies for viewer and remote-cell fan-out;
- productionize approved dedicated/shared placement policies;
- validate secure cross-cloud and on-prem transports;
- add active migration only after state continuity is supportable.

**Gate:** each offered placement mode has a documented performance, availability,
security, and cost envelope.

## Security, privacy, and residency

- Placement policy is an authorization boundary, not only a scheduler hint.
- Strict dedicated/residency-constrained work never silently spills elsewhere.
- Cross-cell consume URLs must be short-lived or job-scoped and encrypted in
  transit; the current plain RTSP external path is not the final WAN contract.
- Relay auth continues to validate per-source/per-job credentials.
- Worker identity must be bound to its cell; callers cannot self-select a more
  privileged or dedicated cell in a claim body.
- Metrics must avoid workspace/source identifiers where high-cardinality or
  customer-sensitive labels would leak data; join through controlled logs/traces.
- Dedicated and customer-managed cells need explicit upgrade, draining, secret
  rotation, and support ownership.

## Open questions

1. What initial SLOs define acceptable preview startup, delivered FPS,
   decode-to-result latency, relay loss, and recovery time?
2. ~~Is the first second cell another Crusoe region, GCP, or a performance-only
   environment?~~ **Resolved for Phase 2:** Crusoe South Central staging,
   `crusoe-ussc1`; this is not a production-region commitment.
3. ~~Where should the cell registry live initially: deployed configuration,
   Firestore, or an existing infrastructure/service registry?~~ **Resolved for
   Phase 1:** validated deployed configuration behind one registry adapter.
4. ~~What is the stable cell identity presented by a fleet worker, and how does the
   platform verify it rather than trusting request data?~~ **Resolved for Phase
   1:** immutable `VIDEO_PROC_CELL` plus a cell-scoped fleet secret; the body cell
   is only a matching assertion and cannot grant placement.
5. What workspace placement modes are product commitments versus internal testing
   controls?
6. Which metrics are available from Crusoe LoadBalancers and GPU nodes, and what
   network limits must be validated with the provider?
7. What benchmark workflow corpus represents real customer monitoring workloads?
8. How should a workload profile be invalidated when a workflow, model, runtime,
   resolution, or target FPS changes?
9. When does a source receive a new relay shard within the same cell?
10. Which secure WAN transport should follow the RTSP/TCP baseline?
11. What stateful Workflow blocks must support checkpoint/restore before active
    migration is considered safe?
12. How are reserved and dedicated capacity represented in billing and metering?

## Documentation ownership

- [HANDOFF.md](HANDOFF.md) remains the end-to-end architecture and implementation
  context.
- This RFC is the source of truth for multi-cell placement, relay capacity, and
  workload-aware scheduling while it is under design.
- [DEPLOY_PLAN_STAGING.md](DEPLOY_PLAN_STAGING.md) is retained as the historical
  first-cell deployment rationale; the roboflow-infra chart README is authoritative
  for current deployment mechanics.
- Benchmark commands, fixtures, corpus definitions, results, and environment
  manifests live in [`benchmarks/`](benchmarks/). Results and fixtures remain
  untracked; representative reports should be published with their exact commit,
  environment, and fixture hashes.
- Material architecture decisions and completed milestones must update both this
  RFC and the handoff in the same change.
