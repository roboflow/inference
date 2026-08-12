# Staging failure-injection controller

This controller exercises the video benchmark recovery path after the loss of
one processor or MediaMTX pod. It is deliberately narrower than a general
chaos-testing tool:

- staging is hard-coded as the only environment;
- the kubectl context must be the exact allowlisted staging context, must
  exactly match the active context, and its kubeconfig cluster name and API
  server must match the allowlisted immutable identity;
- the namespace must be `video-proc` or start with `video-proc-bench`;
- one invocation deletes at most one exact, controller-owned pod;
- there are no video service API writes;
- default operation only renders a plan;
- execution requires an exact run-ID confirmation and a new evidence directory.

The preferred and unambiguous API identity is the exact
`us-central1-roboflow-staging.cloudfunctions.net` hostname used by the corpus
runner. `api.roboflow.one` is also retained because the existing staging runner
explicitly recognizes it, but it is a less self-describing alias; the controller
does not allow arbitrary subdomains, suffix matches, `api.roboflow.com`, or
caller-provided host allowlists.

Run the startup and steady-state cases separately with distinct benchmark run
IDs. This avoids ambiguous recovery evidence and makes cleanup ownership clear.

## Dry run

Copy one of the example scenarios and replace the run ID and checkpoint path.
The checkpoint must be the atomically written report for that exact API corpus
run, and its filename must contain the run ID.

```bash
python development/video_poc/benchmarks/fault_injection/run_staging_fault.py \
  development/video_poc/benchmarks/fault_injection/processor-startup.staging.example.json \
  --output /tmp/video-fault-plan.json
```

This validates the scenario and writes a credential-free plan. It does not call
kubectl or the video service API.

## Execute after inspection

For a startup fault, start this controller **before** the API benchmark. Missing
checkpoint files are retried. Run a single-job corpus with a lowercase DNS-safe
run ID (63 characters maximum), `--startup-fault-ready-seconds 60`,
`--recovery-timeout-seconds 180`, and a startup timeout of at least 360 seconds.
Once the service reports a claimed processor, the runner persists an explicit
`fault-ready` checkpoint and pauses status polling for the bounded window. The
controller targets only that claimed ordinal; it will not reinterpret an already
running job as startup. For a steady-state fault, either process may start first;
the controller waits for a `measurement` checkpoint in which every captured job
is running.

Startup controller, launched first with a scenario whose checkpoint names the
same `recovery-startup-001` run:

```bash
python development/video_poc/benchmarks/fault_injection/run_staging_fault.py \
  /absolute/path/processor-startup-staging.json \
  --execute \
  --confirm-run-id recovery-startup-001 \
  --evidence-dir /absolute/path/evidence/recovery-startup-001
```

Matching one-job startup corpus in the other terminal:

```bash
python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  --workspace WORKSPACE \
  --source-id SOURCE_ID \
  --profile single-detection \
  --run-id recovery-startup-001 \
  --duration-seconds 60 \
  --startup-timeout-seconds 360 \
  --recovery-timeout-seconds 180 \
  --startup-fault-ready-seconds 60 \
  --execute
```

For a steady-state fault, use a distinct run ID and the steady scenario; it
waits for the runner's measurement checkpoint before deleting the selected pod:

```bash
python development/video_poc/benchmarks/fault_injection/run_staging_fault.py \
  /absolute/path/processor-steady-staging.json \
  --execute \
  --confirm-run-id recovery-steady-001 \
  --evidence-dir /absolute/path/evidence/recovery-steady-001
```

Processor targeting is derived from the selected job ordinal and its reported
processor ID. Relay targeting uses only equality labels and must resolve to one
pod. In both cases the pod must have exactly one ReplicaSet or StatefulSet
controller owner. The controller captures the name, UID, resource version,
labels, controller identity, readiness, and immutable image IDs, then reads the
exact pod again immediately before deleting it by name.

The evidence directory is created exclusively. `events.jsonl` is a flushed,
fsynced, SHA-256 hash chain and becomes read-only at completion. It records the
dry-run plan, trigger, captured and verified target, fault, replacement, and
outcome. Copy it to durable object storage after the run if filesystem-level
immutability or retention is required.

## Recovery and cleanup boundary

All trigger and recovery waits have explicit deadlines. A processor case passes
only after the exact benchmark job is running with a different processor ID and
that processor ID resolves to one new ready managed pod. This remains correct
when pool labels change from `working` to `ready` and when other ready workers
already exist. A relay case passes after exactly one new ready pod owned by the
same captured controller revision exists; the benchmark checkpoint is still used
to prove the requested phase, but the current video API does not expose a relay
reconnect generation to prove media continuity. Pair relay injection with the
frame-identity/continuity analyzer when that evidence is required.
Set the controller recovery deadline longer than heartbeat/loss detection plus
the runner recovery deadline and poll allowance; the processor examples use 300
seconds around a 180-second runner window. The measurement duration must also be
long enough for injection and verified post-recovery frame progress.

For steady-state processor recovery timing, opt the matching corpus run into a
bounded requeue window instead of its normal capacity-test fail-fast behavior:

```bash
python development/video_poc/benchmarks/run_api_workflow_corpus.py \
  ... \
  --recovery-timeout-seconds 180 \
  --execute
```

The report retains every non-running poll under phase `recovery` and records a
bounded before/after event with attempt counts and processor IDs. Recovery is
not accepted on `state=running` alone: the runner requires a later poll with
frame progress and a changed processor or advanced attempt. The measured value
is the runner-observed control-plane recovery interval; the joined upper bound
starts immediately before the delete request at `fault-requested` in the
controller evidence. A terminal job or
an expired recovery deadline still fails the benchmark and reaches cleanup.

After both processes finish, verify the controller's hash chain and join its
pod-deletion timestamp to the runner's verified frame-progress timestamp:

```bash
python development/video_poc/benchmarks/analysis/recovery.py \
  --report development/video_poc/benchmarks/results/api-corpus-RUN_ID.json \
  --evidence /absolute/path/evidence/RUN_ID/events.jsonl \
  --output /absolute/path/evidence/RUN_ID/recovery-summary.json
```

The joined value is explicitly an upper bound from the recorded deletion
request to a poll proving replacement-frame progress, not an exact per-frame
outage duration.

Deleting a pod is the cleanup action: its existing Deployment/StatefulSet must
replace it. The controller never scales, patches, or restarts a workload. If the
replacement deadline expires, the evidence ends as failed and an operator must
inspect that exact controller. The API corpus runner remains responsible for
cancelling benchmark jobs, including after the fault controller exits.

## Known limitations

- `kubectl delete pod` has no CLI flag for a server-side UID precondition. The
  controller minimizes the name-reuse race by re-reading and comparing the UID
  immediately before deletion, but cannot eliminate it. Do not use this tool for
  stable same-name standalone pods; standalone pods are refused anyway.
- A startup fault can only be injected once the service has reported a processor
  ID and the runner has persisted `fault-ready`. It does not cover loss between
  server acceptance and the first durable processor assignment.
- Recovery tolerance is opt-in. Ordinary capacity and fairness runs retain the
  fail-fast default (`--recovery-timeout-seconds 0`) so a requeue cannot hide an
  unstable worker from a capacity result.
- Relay recovery proves Kubernetes replacement readiness, not decoded-frame
  continuity, connector retry behavior, or stream-key preservation.
- Do not run relay injection during a rollout. A Deployment rollout can produce
  multiple matching relay pods or change the ReplicaSet controller UID; both are
  deliberately treated as ambiguous rather than successful recovery. Processor
  recovery is bound to the service-reported replacement processor ID instead.
- This harness does not inject node, GPU-device, network-partition, API, Firestore,
  or Pub/Sub failures.
