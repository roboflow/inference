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

Do not run the controlled-FPS fairness matrix against the current internal
fleet claim surface yet. The dedicated public API persists `maxFps`, but the
currently deployed `light-v2-device` claim handler predates propagation of that
field to workers; a 5 FPS control smoke therefore ran at the native 60 FPS.
Either deploy the merged claim propagation change or move the fleet claim route
to `light-v2-video` before using these fairness results. Native-rate A/B/C
capacity runs intentionally omit `maxFps` and are unaffected.

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
