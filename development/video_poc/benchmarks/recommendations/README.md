# Video benchmark decision framework

This is the living evidence-to-recommendation layer for the staging campaign.
It answers two separate questions:

1. **What did a specific experiment prove?**
2. **What architecture, allocation, or product policy may safely follow?**

The machine-readable registry is
[`decision-framework.json`](decision-framework.json), with its structural
contract in
[`decision-framework.schema.json`](decision-framework.schema.json). It
deliberately records unknowns and pending gates. A measured fact stays scoped
to its fixture, runtime, topology, SLO, and environment; it is never silently
promoted into a general product limit.

## Evidence language

| Term | Meaning |
|---|---|
| `verified_observation` | A committed campaign summary records the exact experiment and result, and the ignored raw artifacts were inspected during that campaign. It may still lack the evidence needed for a broader decision or independent re-analysis from Git alone. |
| `preliminary_observation` | Useful ordering evidence, but missing immutable raw reports, current images, or another required validity property. |
| `design_only` | A proposed contract or architecture, not runtime evidence. |
| `pending` | The described experiment has not produced an accepted result. |
| `certified` | Every requirement for that exact decision is satisfied by verified evidence. No decision currently has this state. |
| `provisional` | A bounded direction is supportable, but one or more decision gates remain. |
| `design_candidate` | A design is concrete enough to test, but no verified runtime observation yet supports selecting it. |
| `insufficient_evidence` | No outcome may yet be selected. |

There are also four useful maturity levels that should not be conflated:

| Level | Minimum proof |
|---|---|
| Capacity observation | Exact workload/runtime/environment and resource telemetry for a completed point. |
| Capacity boundary | Two passing repetitions at the selected point and an observed repeated failing point above it, under the same SLO. |
| Operational candidate | Capacity boundary plus output/mix, cleanup, failure recovery, and progressive soak evidence. |
| Product class | Operational candidate plus admission policy, metering, environment cost, support/fallback semantics, and explicit product review. |

## Current evidence snapshot

| Area | What is established | What is not established |
|---|---|---|
| Threaded L40S A/B/C | For native H.264 1280x720@60, YOLOv8 Nano, output off, all three variants pass strict c2 and fail repeated c4. C/NVDEC improves throughput above the strict knee but not the strict maximum. | Controlled 5/10/15 FPS packing, output cost, heavy workflows, process topology, manager/MPS, or long-run safety. |
| `maxFps` validity | The repaired c1 path delivered 4.840/4.913 FPS for uploaded replay and 4.999 FPS for connector replay at a 5 FPS request. | Any multi-stream controlled-FPS capacity limit. |
| D/E/F process images | Immutable D/E/F images passed import/spawn and disposable L40S smokes. | Job claim execution, distinct c2 children, selective cancellation, crash containment, cleanup, or capacity. |
| CPU | An older Burstable curve suggests c2 strict, c3 a latency near miss, and c4 saturation. | A current certified CPU class, process comparison, Guaranteed-QoS curve, heavy-workflow viability, or economics. |
| MMP | Single-workspace controls localize a 4K subprocess regression to frame transport; 640p removes that regression. | Cross-workspace authorization, result separation, fairness, failure containment, MPS, or a production manager topology. |
| Multi-workspace, recovery, soak, networking, cost | Harnesses and plans exist. | Accepted outcome evidence. |

The checked-in framework is auditable from the committed campaign summaries,
but most raw benchmark reports remain deliberately ignored. Before a production
proposal, publish a sanitized evidence bundle (or immutable artifact URI plus
SHA-256) for every result that drives a selected policy so an independent
reviewer can recompute it.

The c15 entry in the workload matrix is a commercial hypothesis supplied by
the product discussion. It is not benchmark evidence, a capacity requirement,
or permission to relax latency and reliability gates.

## Product hypotheses to preserve during benchmarking

The active internal
[product discussion](https://roboflow.enterprise.slack.com/archives/C0B6H2KJS2J/p1786394686066709)
does not contain a final packaging decision. It surfaces alternatives the
campaign must keep distinguishable:

- **Managed per-stream offer:** the customer declares concurrency, FPS,
  resolution, output, and a versioned workflow, and receives a predictable
  stream/hour quote. The platform recommends and may change the underlying
  worker allocation. Editing the workflow creates an explicit re-profile and
  re-price boundary.
- **Worker-class launch step:** expose a small set of classes with measured
  example envelopes and have the platform recommend the class. This can launch
  before a reliable workflow-cost predictor, but it must not make the customer
  guess whether an arbitrary workflow fits or imply that a physical machine is
  the product.
- **Dedicated production capacity:** strict single-workspace pools or cells may
  be appropriate for steady 24/7 loads, reservations, residency, or isolation.
  This is a placement/reliability contract even if the customer buys a stream
  service rather than a machine.
- **Shared preview capacity:** preview is bursty and user-experience driven.
  Its admission, noisy-neighbor, and burst SLO should be tested separately from
  steady paid processing economics.
- **Separated media and processing:** encoded ingest/relay cost follows bitrate,
  viewer/output fan-out, and placement; compute cost follows the workflow and
  target cadence. The product may meter or package them separately, including a
  streaming/scheduling-only service for customer-managed workers.

The benchmark output should make either of the first two launch contracts
possible. It should not encode a packaging preference into capacity gates.

## Decision summary

| Decision | Current bounded recommendation | Status |
|---|---|---|
| Worker topology | One process per job is the leading candidate because it isolates decoder, workflow, model, CUDA context, and publisher without copying frames through supervisor IPC. Select only after D/E/F lifecycle and capacity comparisons. | Provisional |
| GPU streams/SLO | c2 is the strict maximum only for the exact native-rate A/B/C profile. The product packing limit is unknown. | Provisional |
| CPU versus GPU | Keep routing experimental and profile-specific. Run current Guaranteed CPU classes and a matching GPU control before selecting a tier or class. | Insufficient |
| Trusted shared model manager | A trusted pod-global manager may share backends across jobs/workspaces. It still needs per-job authorization, identity, accounting, fair queues, and result isolation. A manager per workspace is not required merely because tenant IDs differ. | Design candidate |
| Fairness/admission | The job is the execution and runtime-SLO unit. Workspace identity is relevant to auth, quota, reservations, dedicated placement, admission ordering, accounting, and reporting—not automatically to process topology. | Insufficient |
| Dedicated/multi-cell | Sticky lazy home-cell placement with colocated processing is the leading design. Workspace/connector policy may constrain or prefer cells, including strict dedicated cells without fallback. | Design candidate |
| MPS | Test only as an optional scheduler inside one exclusive GPU pod. It is not a security or memory/fault-isolation boundary. | Insufficient |
| Recovery/soak | Keep recovery out of capacity certification; measure faults separately and advance 15m -> 1h -> 4h -> 12h only after each gate passes. | Insufficient |
| Relay/network | Isolate every edge, then measure the combined cell. Do not call a plateau a MediaMTX limit until generators, node entitlement, LB, and CNI have headroom. | Insufficient |
| Pricing/capacity | Separate resource allocation from customer pricing. Measure cost per successful service unit at a certified operating point; pricing remains an explicit product choice. | Insufficient |

## Exact selection rules

### Worker topology and maximum streams

Choose a job-process D/E/F topology only if all of these are true:

- c2 shows one supervisor and one distinct child PID per job;
- selective cancellation and a killed child leave the sibling advancing and
  return `activeJobs` to zero;
- two identical repetitions pass at the selected capacity point and an observed
  repeated failure bounds it;
- the chosen topology does not regress target FPS, histogram p95/p99 latency,
  startup, output behavior, resource headroom, or cleanup versus its threaded
  control; and
- the selected point passes the progressive soak.

For the native-rate L40S curve, the strict per-stream gates remain at least 90%
of the variant c1 baseline, cohort spread at most 10%, Jain fairness at least
0.99, histogram p95 at most 50 ms, p99 at most 150 ms, and TTFR at most 30
seconds, with no retry, recovery, reset, restart, CUDA error, output stall, or
relay loss. Controlled-FPS service tiers additionally require at least 90% of
the requested FPS at c1 and at every certified point. Never combine native,
5-FPS, 10-FPS, and 15-FPS curves.

### CPU versus GPU routing

Routing is keyed by the complete versioned workload profile: workflow
fingerprint, model versions, resolution, target FPS, output mode, runtime, and
SLO. A CPU class qualifies only with immutable-image evidence, Guaranteed QoS,
CFS throttling at most 1%, CPU p95 at most 90% of the class, memory at most 80%
of its limit, exact process identity, clean final state, and the ordinary FPS,
latency, fairness, and boundary gates. Compare it with a GPU run that changes
none of the workload inputs.

Possible valid outcomes include one CPU class, several size classes, or GPU-only
routing for a profile. Do not preselect one. Rank candidates by successful
stream-hours and successful frames per worker-hour, startup/tail latency,
memory per job, headroom, and failure containment—not raw inference speed.

### Trusted shared model manager

Treating the manager as trusted removes the need for a process or manager per
workspace; it does not remove tenant-aware control-plane checks. Certification
requires:

- authorization before every model route/cache reuse, including a negative
  test against a model another workspace already warmed;
- opaque client/job identity on every request, SHM slot, response, cancellation,
  error, metric attribution, and output route;
- watermark tests proving no cross-client result or preview crossover;
- bounded per-client inflight work or fair queuing so one high-rate stream
  cannot dominate a shared model queue;
- backend death/reload, cancellation-during-load/inference, SHM exhaustion,
  and long-lived model churn tests; and
- bounded batch wait that preserves the low-latency stream SLO.

Backend sharing should follow model/runtime compatibility and measured value,
not workspace identity by default. A workspace-specific instance remains a
policy tool for dedicated capacity, incompatible settings, or an observed
fairness/security need.

### Job fairness and workspace admission

Runtime certification starts per job: every job must attain at least 90% of its
target, retain histogram p95 at most 50 ms, keep stable identity and counters,
and avoid migration. Cross-workspace runs additionally require tenant target
attainment spread at most 10% and delayed incumbents to retain at least 90%.

Workspace identity matters when allocating scarce shared capacity: quotas,
weighted admission, reserved units, dedicated cells/pools, priorities, billing,
and reporting. It does not change the fact that a job is what executes. Static
admission cannot guarantee runtime GPU fairness, so heavy-neighbor experiments
must either validate the fit or cause the scheduler to reject it.

### Dedicated cells and remote execution

The one-cell migration must first prove persisted source/job placement,
placement-derived URLs, verified worker cell identity, and transactional
cell-scoped claims. Only then can a second staging cell prove deterministic
first activation, stickiness, strict workspace pinning, draining, idle
reassignment, and absence of cross-cell misclaims.

Ordinary live jobs inherit their source home cell. Remote execution is explicit,
authorized policy and transports encoded media. Compare direct WAN fan-out with
one encoded origin-to-replica copy plus local fan-out. A dedicated policy must
state whether relay, processors, or the entire cell is dedicated; reserved
capacity, upgrades, draining, secret rotation, support ownership, billing, and
fallback must all be explicit.

### MPS

Compare the same model, process topology, warm/cold/cache order, target FPS, and
manager settings with MPS off/on. Test no cap and 50%/25% active-thread caps,
same-model batching, and different-model backends. A throughput win is accepted
only if target attainment, p95/p99 latency, batch wait, startup, and incumbent
fairness stay within SLO.

Client termination, backend failure, device fault, MPS-server fault, cleanup,
and restart must establish the blast radius. MPS remains a scheduling option,
never the cross-workspace security boundary. MIG is not available on the L40S
fleet being tested.

### Relay, recovery, and soaks

Capacity suites allow no recovery. Fault suites bind the exact old/new pod to
verified post-fault frame progress and report a recovery upper bound, not exact
media downtime. Relay liveness alone never proves continuity.

Network certification separates publish-copy, read-copy, and decode validation,
then isolates same-node, CNI cross-node, public LB, relay node, processor node,
and WAN edges. Every curve must identify the first saturated resource and a
failing boundary. The whole-cell run combines processors, relay fan-out,
outputs, and viewers; isolated maxima must not be multiplied.

### Capacity economics and pricing

Define a successful stream-hour by its versioned workflow plus promised
FPS/resolution/output envelope, latency SLO, and availability. Also retain
successful processed frames for profile comparison. Failed, retrying, or
out-of-SLO time is not delivered service.

For each certified profile calculate:

```text
environment cost per successful stream-hour =
  (worker + ready headroom + relay + bandwidth/egress + control plane
   + observed failure overhead)
  / successful stream-hours
```

Also report cost per successful processed frame so different FPS and workflow
profiles remain comparable, and retain media/relay cost separately from
processing cost. These are cost and allocation inputs, not automatic customer
prices. Packaging, workflow-edit repricing, margins, included units, overages,
quotas, and reserved/dedicated commitments require a separate product decision.

## Updating the framework

After an accepted experiment:

1. Commit the exact report summary, immutable image/config identity, fixture,
   SLO, repetitions, failing boundary, telemetry, and cleanup outcome. For any
   production-driving decision, also publish sanitized raw evidence or an
   immutable artifact URI and digest.
2. Change the corresponding evidence item from `pending` only when the cited
   artifact actually proves its full scope.
3. Mark a requirement `satisfied` only when every cited evidence item is a
   `verified_observation`.
4. Narrow or remove unknowns; never delete one merely because the result is
   plausible.
5. Set a decision to `certified` only after every requirement is satisfied and
   every decision-scoped unknown is resolved. A `provisional` decision must cite
   at least one verified observation; use `design_candidate` when the direction
   is based only on design or preliminary evidence.
6. Run the framework validation test before committing.

The validator intentionally rejects a certified decision backed by preliminary,
design-only, pending, missing, or external evidence.
