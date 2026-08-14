# Distributed relay benchmark harness

This directory contains the dry-run-first implementation of the distributed
client design in
[`STAGING_NETWORK_BENCHMARK_PLAN.md`](STAGING_NETWORK_BENCHMARK_PLAN.md). It is
separate from the single-host smoke runner in the parent directory.

The harness is restricted to an explicitly named staging context and a
dedicated namespace containing `bench`. The renderer never invokes `kubectl`.
It writes a Kubernetes JSON `List` for operator inspection plus a redacted run
manifest. Kubernetes accepts the JSON list anywhere it accepts YAML.

`render_two_cell_validation.py` adds the Phase 1/2 routing-validation contract.
It remains render-only and writes one Job list per staging cell, a redacted
collection manifest, and an initially empty validation-report template. The
South+East example is `two-cell-south-east.staging.example.json`.
It uses the canonical `crusoe-use1` / `crusoe-ussc1` cell IDs and
`ck8s-stg` / `ck8s-stg-us-southcentral1` contexts. The explicit WAN case is
South-origin to an East `c1a.16x` CPU reader through the South public RTSPS
endpoint; it neither schedules on nor requires changes to the active East L40S
capacity fleet.

## Contract

The version 1 scenario declares:

- one immutable agent image digest and fixture identity;
- cell, provider, relay/LB/CNI versions or hashes, and each location's expected
  instance type and documented VPC bandwidth;
- publisher and reader groups, including `publish-copy`, `read-copy`, and
  `read-decode` roles;
- an explicit node selector or node name for every logical location;
- fixture and media URL templates through existing Kubernetes Secret key
  references (a non-credentialed fixture path/URL is also allowed);
- run duration, hard client deadlines, TTL cleanup, and stop thresholds; and
- bounded resource requests/limits for each client group.

Reader groups default to a five-second `startAfterSeconds` delay so publishers
can establish paths first. Set the delay explicitly for calibration and retain
small reconnect allowances when testing connection churn; this delay is not a
substitute for the future controller readiness barrier.

`maxAgents` is an explicit scenario safety limit, with an absolute renderer cap
of 5,000 Jobs. A scenario that expands past its declared limit is rejected
before any manifests are written.

The renderer expands every client into a separate Job. All objects receive a
run-ID label used for cleanup. `activeDeadlineSeconds` stops clients even if the
operator disappears, and `ttlSecondsAfterFinished` removes completed Jobs.
The run manifest also records a safe, argument-vector cleanup command for
interrupt/error handling by the future executing controller.

The agent obtains media URLs from environment variables and expands only the
`{stream}` placeholder. URLs therefore never appear in Job arguments. Its
report retains only redacted URL forms, requested and observed placement,
ffmpeg progress, startup/stall/reconnect outcomes, delivered frames/FPS,
process resource usage, and environment/image identity. The report is written
to `/dev/termination-log` and logged once with the
`BENCHMARK_FINAL_JSON=` prefix. A small `/metrics` endpoint on port 9091 exposes
running, progress, frame, and reconnect measurements while the Job is active.
Those metrics use only role/location labels; the run ID is carried by the Pod
target labels and report rather than emitted as another high-cardinality series
label.

The Pod downward API supplies the actual observed `nodeName`. It cannot expose
arbitrary node labels, so the agent records the scenario's
`requestedNodeInstanceType`; the executing controller must join the observed
node name against Kubernetes node inventory and add the actual instance type to
the aggregate run report. Child maximum RSS is explicitly reported in KiB,
matching the Linux load-agent contract.

For two-cell runs the agent also records the expected cell, expected cluster
context, media-path name, and whether the path is same-cell or explicitly
cross-cell. These are requested assertions, not observed truth. The collection
contract requires the controller to join each Pod UID and node name with the
actual cluster, node labels, image ID, and cell release before marking placement
as proven.

`read-copy` keeps decode CPU out of capacity curves. Use a small
`read-decode` group to enforce the fixture FPS and catch media corruption or
continuity problems.

## Render a staging run

Copy the example, replace the image/fixture identities, selectors, resources,
and Secret references, then render it:

```bash
python development/video_poc/benchmarks/networking/render_distributed_relay_benchmark.py \
  --scenario /path/to/distributed-relay.staging.json \
  --run-id relay-calibration-001 \
  --output-dir development/video_poc/benchmarks/results/relay-calibration-001
```

Inspect `run-manifest.json` and `jobs.json`. In particular, verify the current
Kubernetes context independently, requested node placement, immutable image,
fixture hash/FPS, job count, active deadline, resource bounds, and cleanup
selector. Rendering makes no cluster changes.

## Render a deterministic two-cell validation

Copy the South+East example and replace every placeholder, including the agent
digest, cell endpoint Secret references, fixture hash, cluster contexts,
instance types, release/config hashes, and evidence artifact paths. Endpoint,
fixture, monitoring, and control-plane values are accepted only through
Kubernetes Secret name/key references.

```bash
python development/video_poc/benchmarks/networking/render_two_cell_validation.py \
  --scenario /path/to/two-cell-south-east.staging.json \
  --run-id south-east-routing-001 \
  --output-dir development/video_poc/benchmarks/results/south-east-routing-001
```

Inspect `jobs-east.json`, `jobs-south.json`, `validation-manifest.json`, and
`validation-report.template.json`. The scenario must explicitly declare:

- exactly two recognizable staging contexts and dedicated benchmark namespaces;
- cell-specific ingest, internal consume, public consume, preview, processor,
  and Prometheus endpoint Secret references;
- first-activation, reconnect-stickiness, and dedicated-workspace evidence
  cases;
- both local media paths and a separately authorized
  `explicit-cross-cell`/`remoteExecution=true` experiment;
- baseline and impaired latency/loss/bandwidth/egress profiles; and
- an approval-gated failure experiment with recovery timeline and rollback
  artifacts.

The rendered manifest contains read-only placement collection argument vectors,
required MediaMTX/processor/Prometheus/network evidence, per-cell cleanup
commands, and the failure-event timeline schema. Rendering does not execute any
of those commands. Applying either Job list, changing network impairment,
running a failure experiment, or revoking credentials remains an operator-owned
staging mutation requiring the normal approval and rollback review.

Once the dedicated staging benchmark cell and short-lived media Secret exist,
an operator can apply the inspected manifest:

```bash
kubectl --context ck8s-stg apply -f jobs.json
```

Observe placement and collect final agent reports before TTL expiry:

```bash
kubectl --context ck8s-stg --namespace video-proc-bench get pods \
  --selector benchmark.roboflow.com/run-id=relay-calibration-001 \
  -o wide

kubectl --context ck8s-stg --namespace video-proc-bench logs \
  --selector benchmark.roboflow.com/run-id=relay-calibration-001 \
  --prefix=true
```

On interruption, run the exact cleanup argument vector recorded in
`run-manifest.json`. Do not paste or commit Secret contents. Deleting Jobs
deletes their Pods; the short-lived credential issuer must revoke the run
credential separately when that control-plane capability is implemented.

## Current boundary

This increment deliberately renders rather than applies. It provides the
distributed placement, load-agent, progress, report, threshold, and cleanup
contracts needed for Phase 0/1 calibration. The executing controller still
needs to add readiness barriers, Prometheus/Crusoe query-range collection,
aggregate stop enforcement, automatic report collection, and short-lived
credential issue/revoke calls before unattended capacity ramps are safe.

The file-only multi-cell successor lives in [`multi_cell/`](multi_cell/). It
adds counterbalanced same-cell/cross-cell pairs, connector/relay/processor and
dedicated-workspace placement cases, immutable network/runtime identities, a
pixel-clock probe, exact per-run manifests, and a fail-closed evidence analyzer.
Its committed campaign is intentionally non-runnable until a fresh staging
identity snapshot is captured. It performs no cluster or cloud operations.
