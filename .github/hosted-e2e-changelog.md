# CI infrastructure changelog

## Unreleased

### Persistent model cache on the GPU runner — October 21, 2026

- Mount a per-leg directory under the runner user's home as the server test
  container's `MODEL_CACHE_DIR` in every stage of the shared GPU action, so
  later stages and later runs load weights from disk instead of downloading
  them again. Image reuse does not cover weights, since the image holds none.
- Validate the cache before each run with `ci_model_cache.py`: discard it when
  `MODEL_CACHE_KEY` changes, when it is older than `MODEL_CACHE_MAX_AGE_DAYS`
  (14), when a previous run was interrupted, or on the `reset_model_cache`
  dispatch input; wipe through the already-pulled test image rather than
  sudo, and log reuse, size and free disk. Skip the mount instead of failing
  when free disk stays below `MODEL_CACHE_MIN_FREE_GIB` (20) after a wipe.
  Mark the directory in use for the job and release it in a final `always()`
  step, so a second run of the same leg on the host falls back to the
  container-local cache instead of sharing files with a live server. Mark it
  dirty for the job and clean it only when the job was not cancelled, and
  turn on `ATOMIC_CACHE_WRITES_ENABLED` in the test containers, so a
  container killed mid-download cannot leave a truncated or half-extracted
  artifact in the cache.
- Make `/model/clear` honour `DISK_CACHE_CLEANUP`, so the per-module clears in
  the integration tests no longer delete cached weights when the flag is off.
- Add offline tests for the cache script and the action wiring. A live T4
  dispatch pair (populate, then measure) still needs to confirm the timing
  gain.

### Main image reuse through Depot — October 20, 2026

- Supersede the fast-track Depot saved-image trial with image reuse for every
  push to main (`ci_image_reuse.yml`): build once per family with
  `depot build --save` under GitHub OIDC, then pull the exact digest in both
  model modes. The trial workflows, their approval variables and reviewer
  environments are removed; a release branch proves the same path by
  dispatching the orchestrator. GCP Artifact Registry stays the release mirror.

### Depot saved-image trial — October 16, 2026

- Add an opt-in fast-track path that saves CPU and GPU images in Depot once
  and tests their exact digests in both model modes. Share the existing server
  test stages with the legacy workflows and use `fail-fast: true`.
- Require same-repository manual execution, an explicitly approved branch and
  commit, and separate protected builder/reader environments. Authenticate to
  Depot with GitHub OIDC through the org-level connection, as every other Depot
  workflow at Roboflow does; store no Depot token anywhere. Keep the project and
  registry as repository variables, mint one-hour read-only pull tokens in the
  job for registry logins, remove them before tests and withhold raw build/test
  output from public logs.
- Keep the existing default path and release publishing unchanged. The trial
  remains disabled unless configured; Depot permissions, retention, runner
  isolation and a live build/pull/test run still require verification.
- Add offline regression coverage for authorization, image identity, credential
  cleanup, output suppression, complete test coverage and legacy-path exclusion.
- Validate the Depot transport locally against project `inference-ci-test`
  (since renamed `manual-inference-ci-test`):
  `--save`, digest-addressed manifest inspection, pull by child digest, the
  identity checks and push denial for pull tokens all pass. The Actions OIDC
  path still needs one live run.

### Review follow-ups — October 16, 2026

- Record a redacted one-line summary of each failing case's final attempt in
  shard evidence and `acceptance.json`, and print those summaries in the visible
  log when a shard or aggregation stage fails. Credential values are replaced
  before recording; full tracebacks stay only in the encrypted stage log.
- Keep manual dispatch available from any same-repository branch or tag, as
  before sharding; drop the default-branch/fast-track restriction and the
  umbrella-PR lookup. The guard only confirms a manual same-repository dispatch
  and pins one commit across jobs. It is accident prevention, not a credential
  boundary: secrets are repository-level and reachable by anyone with write
  access. Enforcement through GitHub Environments (deployment-branch rules or
  required reviewers) is a separate follow-up needing repository admin changes.

### Repository activation — October 16, 2026

- Connect the existing staging and production manual entrypoints to preparation,
  shared warmup, four parallel shard jobs, and unconditional result aggregation.
- Require a manual same-repository dispatch, use one dispatch SHA in every job
  and serialize both target workflows. Any branch or tag may run, as before.
- Preserve the operator's skip-warmup option and explicitly allow only the five
  cases already disabled in the reviewed suite. Reject empty provider credentials.
- Capture detailed test output and transfer authenticated encrypted evidence
  between jobs. Keep visible output to fixed statuses and aggregate counts;
  reuse existing hosted credentials with run-specific, domain-separated key
  derivation. Detailed timing profiles can be supplied through repository secrets.
- Preserve the private deployment-acceptance workflow. The repository manual
  path makes no deployment-image identity claim, matching the previous workflow.
- Add subprocess integration tests for isolated job directories, encrypted
  transfers, four-shard coverage, retries, skip handling, secret-bearing failures,
  tampering, incorrect contexts and absent evidence. Test the operator-facing
  entrypoints so a disconnected runner cannot pass integration validation.
- Live rollout remains pending until a reviewed staging run
  verifies preparation, shards A–D and aggregation. Offline tests alone do not
  demonstrate activation or a performance improvement.


### Review follow-ups — October 15, 2026

- Reject unauthorized collection-time skips, including module-level skips and
  missing optional dependencies. Record collector IDs in private evidence,
  require an exact separate allowlist, and reject skip drift after preparation.
- Reject allowlisted skipped retries following an unresolved call failure.
  Preserve successful retries and clean deliberate skips, with subprocess
  regression coverage for both review findings.
- Record team agreement on the shared-CI-infrastructure approach while retaining
  private-caller configuration and live-validation rollout prerequisites.
- Clarify that missing provider credentials may skip tests, while empty or
  invalid credentials may cause authentication failures. Unexpected skips and
  failed tests both reject acceptance.
- Exercise explicitly skipped cases through real pytest subprocesses, checking
  both acceptance with an exact skip allowlist and rejection without it.
- Document complete-workflow reruns: preparation and shard evidence must share
  the same attempt; rerunning only failed jobs cannot reuse an earlier plan.

### Added — October 15, 2026

- Opt-in private-caller acceptance workflow with four serial pytest shards,
  one shared warmup, deployment identity checks and complete result accounting.
- Deterministic scheduling by estimated runtime, keeping all cases from each
  test file together, including authentication and parameter variants.
- Separate private staging and production timing profiles. Changed source or
  case counts invalidate file estimates; new or changed files use a fallback.
- Offline coverage for scheduling, retries, fixture failures, incomplete
  evidence, identity drift and private-caller guards.

### Why the shard case counts differ

Balance estimated execution time rather than the number of cases: expensive
provider tests can take longer than many model checks combined. Assign the
longest files first to the shard with the least estimated work.

The initial production proposal uses the
[v1.7.4 production run, attempt 1](https://github.com/roboflow/inference/actions/runs/38001719494/attempts/1).
Its hosted-test directory matches the implementation's base revision. The
previous release, v1.7.3, covered 190 cases rather than 270 and is excluded from
these scheduling weights.

| Shard | Cases | Estimated test work |
|---|---:|---:|
| A | 20 | 186 seconds |
| B | 30 | 186 seconds |
| C | 107 | 185 seconds |
| D | 113 | 186 seconds |

For example, the 8-case SpaceXAI file took approximately 177 seconds, whereas
the 51-case Roboflow model file took approximately 28 seconds. Equal case
counts would therefore not mean equal runtime. These assignments are an
initial proposal derived from the profile, not hardcoded shard membership.

**Estimate limits:** the source is one serial production run with timestamped
file-progress output, not per-case duration reports. Core-model timing uses a
fallback because its logged interval includes warmup. The estimates above
exclude shared warmup, runner setup and aggregation. Actual parallel balance
and provider load remain unvalidated; these figures are not measured sharding
performance or a runtime guarantee. Staging uses its own profile.

### Security and rollout

Detailed timing profiles, raw logs, case evidence, credentials and deployment
metadata remain private. This changelog contains only the aggregate scheduling
rationale and public test-suite examples. Private deployment-acceptance execution
requires the private caller and deployment-verification adapter. The repository
manual workflows are now connected to sharding; live rollout still requires
maintainer review and load/isolation validation.

See [PR #3211](https://github.com/roboflow/inference/pull/3211) for the repository
activation approach, validation and rollout requirements. Timing profiles are
optional; without them, the four shards use case-count balancing. Rerun the
complete workflow after a failure because evidence is bound to its run attempt.
