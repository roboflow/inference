# Staging failure-injection controller

This controller exercises the video benchmark recovery path after the loss of
one processor or MediaMTX pod. It is deliberately narrower than a general
chaos-testing tool:

- staging is hard-coded as the only environment;
- the kubectl context must contain `stg` or `staging`, must not contain `prod`,
  and must exactly match the active context;
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

Start the matching API benchmark first. For a startup fault, the controller
waits until the selected ordinal has a `processorId` while the checkpoint is in
`started`, `startup`, or `arrival`. It refuses to inject if all jobs are already
running. For a steady-state fault, it waits for a `measurement` checkpoint in
which every captured job is running.

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
  ID. It does not cover loss between server acceptance and the first durable
  processor assignment.
- The corpus runner currently treats a non-running job during measurement as a
  failed benchmark and may begin cancellation before worker recovery completes.
  That is useful failure evidence, but a dedicated recovery-run mode will be
  needed to measure retry time without the runner ending the run.
- Relay recovery proves Kubernetes replacement readiness, not decoded-frame
  continuity, connector retry behavior, or stream-key preservation.
- Do not run relay injection during a rollout. A Deployment rollout can produce
  multiple matching relay pods or change the ReplicaSet controller UID; both are
  deliberately treated as ambiguous rather than successful recovery. Processor
  recovery is bound to the service-reported replacement processor ID instead.
- This harness does not inject node, GPU-device, network-partition, API, Firestore,
  or Pub/Sub failures.
