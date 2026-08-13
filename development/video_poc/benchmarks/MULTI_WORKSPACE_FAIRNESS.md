# Multi-workspace fairness scenarios

Object-based workloads opt a matrix scenario into the staging-only
multi-workspace runner. Existing string workloads keep the original
single-workspace behavior unchanged.

Each workload can independently select its workspace, source, credential
environment variable, tier, FPS limit, processing mode, and output publishing:

```json
{
  "schemaVersion": 1,
  "environment": "staging",
  "defaults": {
    "apiBase": "https://api.roboflow.one",
    "durationSeconds": 300,
    "maxPlannedJobs": 16
  },
  "scenarios": [
    {
      "name": "two-tenant-noisy-neighbor",
      "workloads": [
        {
          "profile": "single-detection",
          "count": 4,
          "workspaceLabel": "tenant-a",
          "workspace": "STAGING_WORKSPACE_A",
          "sourceId": "STAGING_SOURCE_A",
          "apiKeyEnv": "VIDEO_BENCHMARK_API_KEY_A",
          "tier": "gpu",
          "maxFps": 15,
          "mode": "stream",
          "publishOutput": false
        },
        {
          "profile": "instance-segmentation",
          "count": 1,
          "startAfterSeconds": 60,
          "workspaceLabel": "tenant-b",
          "workspace": "STAGING_WORKSPACE_B",
          "sourceName": "Stable uploaded fixture",
          "apiKeyEnv": "VIDEO_BENCHMARK_API_KEY_B",
          "tier": "gpu",
          "maxFps": 5,
          "mode": "stream",
          "publishOutput": false
        }
      ]
    }
  ]
}
```

Set the referenced environment variables and run the normal matrix command.
The parent suite command contains only the matrix path and scenario name; key
values never appear in argv. Dry-run output and result reports use
`workspaceLabel` and safe source metadata, not workspace routing IDs,
credential-variable names, or credential values.

Use labels that are meaningful in benchmark analysis but do not identify a
customer. Inline `apiKey`, `token`, or authorization fields are rejected. Every
API base is independently restricted to staging hosts, including workload-level
overrides.

`requireSingleProcessor` still defaults to true. That assertion is useful for a
same-worker noisy-neighbor test, but it is a placement assertion, not tenant
isolation. Use separate scenarios for same-worker fairness and scheduler-level
distribution.

The production execution unit remains a job: one source, workflow, and runtime
configuration. Workspace identity does not define a process or model-cache
boundary. The analyzer always certifies per-job progress first. Its per-workspace
aggregation is an explicit noisy-neighbor/admission-policy view that answers
whether one tenant submitting many jobs harms another tenant; it is not a
requirement to create one worker or model manager per workspace.

## Current staging fixtures

The 2026-08-13 preflight verified two feature-enabled staging workspaces through
the dedicated `https://api.roboflow.one` video surface:

- label `tenant-a`: workspace route `thomas-workspace`, connector source ID
  `9g7UzPcDyVBFBJ0dLei6`, `traffic.mp4`, H.264 1280x720 at 60 FPS, credential
  environment variable `VIDEO_BENCHMARK_API_KEY_A`;
- label `tenant-b`: workspace route `rf-inference-benchmark`, connector source
  ID `d5XmPQAZssPpE3clCmcY`, `vehicles.mp4`, dedicated connector
  `bench-l40s-capacity`, credential environment variable
  `VIDEO_BENCHMARK_API_KEY_B`.

These identifiers and environment-variable names are safe to check into a
staging matrix; key values are not. The two source contents and native frame
rates differ, so use an admitted `maxFps` limit when comparing tenant fairness
and treat model/post-processing content sensitivity as a remaining covariate.

The checked-in code now propagates `maxFps` through create, read, and fleet
claim, but deployment history has previously mixed independent function refs.
Do not certify controlled-FPS fairness from source inspection alone. Immediately
before the first fairness run, create one 5 FPS job and require both the public
job response and worker runtime report to echo 5 while delivered FPS remains
within the configured attainment bounds. Native-rate A/B/C capacity runs
intentionally omitted `maxFps` and are unaffected.

Analyze a completed report with:

```bash
python -m development.video_poc.benchmarks.analysis.fairness \
  development/video_poc/benchmarks/results/api-multi-workspace-RUN_ID.json
```

The analyzer derives per-job and per-tenant delivered FPS, target attainment,
within-tenant and cross-tenant Jain fairness, target-attainment spread,
incumbent baseline retention after a later tenant arrives, p95 latency, and
processor placement/migrations. It fails a same-worker fairness run if the
planned, started, and sampled jobs do not match exactly; any job lacks a target
or enough steady samples; a frame counter resets; the configured FPS limit was
not propagated; either job misses its target; tenant attainment differs by more
than 10%; a delayed-arrival scenario lacks an incumbent baseline or loses more
than 10% of it; frame-histogram p95 latency is unavailable or exceeds 50 ms;
jobs migrate; or the tenants did not share one processor. Sampled EMA latency
is retained as diagnostic telemetry but is not accepted as a frame-level p95
SLO. Use `--allow-distributed` only for scheduler-placement scenarios; that mode
retains every other fairness and stability gate.

Executed multi-workspace runs attach this analysis to the final report. The
runner preserves `operationalSuccess` separately and sets overall `success`
only when both lifecycle execution and the fairness SLO pass. The parent matrix
therefore stops before longer or more expensive scenarios when a short fairness
gate is operationally healthy but unfair.

Executing runs atomically checkpoint after every poll and handle `SIGINT` or
`SIGTERM` by cancelling all captured jobs. After an abrupt runner loss, inspect
the exact checkpoint and original matrix without network access:

```bash
python development/video_poc/benchmarks/cleanup_api_multi_workspace_run.py \
  --matrix /path/to/matrix.json \
  --scenario two-tenant-noisy-neighbor \
  --run-id RUN_ID
```

The janitor joins each captured job to its plan ordinal and requires the current
matrix's canonical SHA-256 to match the one captured by the run. The checkpoint
therefore does not persist workspace IDs, credential environment names, or
keys, while a later routing edit cannot retarget a colliding job ID. Add
`--execute` only after inspecting the exact run-scoped plan; credentials are
then resolved from the original matrix's environment-variable declarations.
When exact cleanup succeeds, the janitor also atomically finalizes the original
checkpoint as a complete failed run. The parent suite can then reconcile it on
`--resume`: the normal fail-fast policy stops there, while an explicitly
unchanged `--continue-on-error` suite may advance. Retrying the failed scenario
uses a new suite/run ID rather than reusing its idempotency keys.
