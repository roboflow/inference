# CI infrastructure changelog

## Unreleased

### Depot saved-image trial — October 16, 2026

- Add an opt-in fast-track path that saves CPU and GPU images in Depot once
  and tests their exact digests in both model modes. Share the existing server
  test stages with the legacy workflows and use `fail-fast: true`.
- Require same-repository manual execution, an explicitly approved branch and
  commit, and separate protected builder/reader environments. Supply credentials
  and private resource configuration only through secret references; remove pull
  credentials before tests and withhold raw build/test output from public logs.
- Keep the existing default path and release publishing unchanged. The trial
  remains disabled unless configured; Depot permissions, retention, runner
  isolation and a live build/pull/test run still require verification.
- Add offline regression coverage for authorization, image identity, credential
  cleanup, output suppression, complete test coverage and legacy-path exclusion.

### Repository activation — October 16, 2026

- Connect the existing staging and production manual entrypoints to preparation,
  shared warmup, four parallel shard jobs, and unconditional result aggregation.
- Restrict credentialed execution to manual runs on the default branch or
  approved release fast-tracks. Check the umbrella PR's exact head SHA, approval
  label and release ancestry before accessing hosted secrets. Use one dispatch
  SHA in every job and serialize both target workflows.
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
- Live rollout remains pending until a reviewed main/fast-track staging run
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
