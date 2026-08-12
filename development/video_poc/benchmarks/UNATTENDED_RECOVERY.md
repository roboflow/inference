# Unattended run recovery

The staging API corpus runner atomically replaces its JSON report after job
creation and after every status poll. A reader therefore sees either the prior
complete checkpoint or the new complete checkpoint, never partially written
JSON. Reports remain credential-free: API keys are read only from the configured
environment variable, and job/source fields use explicit allowlists.

During an executing run, the runner handles both `SIGINT` and `SIGTERM` as a
graceful stop request. It reaches a safe point, cancels every job whose ID was
captured, waits for terminal states, writes the final checkpoint, and exits
non-zero. The polling interval bounds how long a stop request can take to reach
that safe point; cleanup has its own timeout.

If the runner is killed without an opportunity to execute cleanup, inspect the
exact run-ID checkpoint before doing anything:

```bash
python development/video_poc/benchmarks/cleanup_api_benchmark_run.py \
  --run-id RUN_ID \
  --output-dir development/video_poc/benchmarks/results
```

The janitor is deliberately staging-only and dry-run by default. It does not
list or sweep a workspace. It validates the checkpoint API host and run ID, then
plans cleanup for only the job IDs already recorded in that checkpoint. Execute
that exact plan with the same dedicated staging service identity used by the
runner:

```bash
export VIDEO_BENCHMARK_API_KEY='...'

python development/video_poc/benchmarks/cleanup_api_benchmark_run.py \
  --run-id RUN_ID \
  --output-dir development/video_poc/benchmarks/results \
  --execute
```

The janitor first refreshes each captured job, skips jobs already terminal,
cancels the rest, and waits for all captured jobs to become terminal. It writes
`cleanup-api-corpus-RUN_ID.json` with the typed expected and actual recovery
states. An inspect, cancel, poll, or timeout error makes the command exit
non-zero for operator follow-up.

The recovery boundary is intentionally narrow: a job that was accepted by the
service but whose response was never received cannot appear in the local
checkpoint. Server-side run ownership or a list-by-idempotency-prefix endpoint
would be required to make that rare ambiguous-acceptance case automatically
recoverable.
