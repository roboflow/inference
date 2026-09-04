# Worker integration: lifecycle probe and per-job execution experiment

The POC worker now has a deliberately narrow, feature-gated execution-domain
seam. Its default remains `PROCESSOR_EXECUTION_DOMAIN_MODE=in_process`, which
creates no child processes and preserves the current deployment behavior.

For a staging experiment only, set:

```text
PROCESSOR_EXECUTION_DOMAIN_MODE=workspace_probe
```

This starts one empty spawned child process for each workspace represented by
the worker's active jobs. The parent alone retains the workspace-to-domain and
job-to-domain maps. Children receive only `stop` or test-only `crash` lifecycle
commands: they do not receive workspace IDs, job payloads, source URLs, API
keys, access tokens, workflow definitions, frames, or models.

The worker monitors child liveness. If a child exits unexpectedly, it reports a
credential-free error and stops only the parent-side `JobRun` objects owned by
that workspace domain; sibling workspace probes and jobs remain running. The
aggregate health/status document exposes only the mode, whether it is
experimental, and the active domain count.

Containment has a bounded deadline (10 seconds by default, configurable with
`PROCESSOR_DOMAIN_CONTAINMENT_TIMEOUT_S`). If a parent-side pipeline is wedged
and cannot stop inside that window, the experimental worker hard-exits instead
of wedging its only domain monitor. Kubernetes then restarts/retires the spent
pod and the platform heartbeat reaper requeues its held jobs.

## What this proves

- Feature selection fails closed on unknown values and is disabled by default.
- Jobs from one workspace share a lifecycle domain while different workspaces
  do not.
- A hard child exit produces one bounded notification covering exactly that
  domain's jobs.
- A wedged parent-side stop cannot permanently disable monitoring of later
  domain failures; it retires the whole experimental worker at the deadline.
- Removing a domain's final job reaps the otherwise-idle child.
- Child arguments and public diagnostics contain no workspace identifier or
  job credential.
- The existing worker has concrete start, release, monitor, and shutdown hooks
  that a real child runtime can keep using.

## What it does not prove

This mode is **not process isolation and not a tenant-security boundary**.
`JobRun`, `InferencePipeline`, model registries, decoders, publishers,
credentials, frames, and result buffers still live in the parent worker. The
empty child is a control-plane probe, and killing it intentionally tears down
the corresponding parent-side jobs so failure ownership can be exercised.

MPS would not change that conclusion. MPS can share and schedule GPU work, but
it neither isolates tenants nor authorizes model or shared-memory access.

## Production blockers beyond the staging capacity experiment

1. Extend the experiment's versioned, bounded status/control protocol for batch
   results and any browser-facing event stream that production requires. Do not
   add raw frames or unbounded per-frame events to the parent connection.
2. Decide the production contract for debug MJPEG, structured per-frame events,
   and completed batch result files. Live output publishing already stays in
   the child and goes directly to MediaMTX.
3. Remove broad workspace credentials from long-lived parent and cross-process
   state. A child must use a short-lived, job-scoped model authorization or a
   broker that re-authorizes every model operation; merely sending the current
   workspace API key over IPC is process separation, not credential isolation.
4. Make model authorization part of every MMP request before cache lookup or
   route reuse. Shared model processes and shared-memory slots need client
   ownership checks, generation IDs, bounded leases, and cleanup after crashes.
5. Start all CUDA/model runtimes only after `spawn`. The supervisor must remain
   CUDA-free, and image tests must verify this for both current inference and
   the new model manager.
6. Reconcile child completion with platform heartbeats exactly once. Claims,
   retry attempts, cancellation, processor access tokens, and terminal/failing
   reports stay parent-owned and must tolerate a crash at every start/stop
   transition.
7. Add end-to-end tests for unauthorized warm-model reuse, cross-client shared
   memory reads, simultaneous workspace crashes, stuck-child escalation,
   parent restart, result cleanup, and repeated workspace/model churn.
8. Measure process startup, RAM/VRAM duplication, file descriptors, `/dev/shm`,
   decode-to-result latency, throughput, and fairness before selecting
   workspace-process versus job-process granularity.

Until those items are complete, reports for the empty lifecycle mode must call
it `workspace_probe`; they must not label that mode process-isolated execution.
Reports using `PROCESSOR_JOB_EXECUTION_MODE=process` may accurately describe
per-job execution isolation, but not a hardened tenant-security boundary.

## Real per-job execution experiment

The worker now also has a separate staging topology switch:

```text
PROCESSOR_JOB_EXECUTION_MODE=process
```

Unlike `workspace_probe`, this mode moves the real decoder, `InferencePipeline`,
workflow/model execution, CUDA context, and output publisher into one spawned OS
process per live job. It is independent of the inference runtime and ingest
mode, enabling the D/E/F matrix documented in
[`JOB_PROCESS_MATRIX.md`](JOB_PROCESS_MATRIX.md). The supervisor retains
claims, access tokens, heartbeats, cancellation, aggregate metrics, and durable
failure reporting. Only bounded control and telemetry cross the connection;
frames and tensors never do.

Do not combine `workspace_probe` and per-job process mode. The former remains a
lifecycle-only prototype; the latter is the actual execution-isolation capacity
experiment. It is not yet a production tenant sandbox or a replacement for the
authorization/fairness work required by a shared model manager.
