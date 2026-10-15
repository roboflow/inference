# Hosted platform E2E changelog

## Unreleased

### Review follow-ups — October 15, 2026

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
rationale and public test-suite examples. Live execution requires the private
caller, deployment-verification adapter, maintainer review and load/isolation
validation. Existing manual workflows remain unchanged pending migration.

See the [implementation and integration guide](hosted-e2e.md).
