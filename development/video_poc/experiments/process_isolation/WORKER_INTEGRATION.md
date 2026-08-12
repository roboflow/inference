# Worker integration: lifecycle-only workspace probe

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

## Blockers before executing workflows in children

1. Define a versioned, bounded IPC protocol for start, cancel, status,
   structured errors, aggregate counters, frame/event delivery, and graceful
   shutdown. It needs explicit backpressure so a slow browser or parent cannot
   grow child memory without bound.
2. Decide which side owns decoders and output publishers. Moving the full
   `JobRun` into a child is the clearest fault boundary, but the parent HTTP API
   then needs bounded access to events, previews, and completed result files.
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

Until those items are complete, benchmark reports must call this mode
`workspace_probe`; they must not label it process-isolated execution.
