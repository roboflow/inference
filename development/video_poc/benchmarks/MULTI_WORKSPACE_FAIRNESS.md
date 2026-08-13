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
