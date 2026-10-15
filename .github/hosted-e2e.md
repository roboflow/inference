# Private four-shard hosted acceptance

See the [changelog](hosted-e2e-changelog.md) for changes and the rationale behind
the initial runtime-balanced shard proposal.

`hosted_inference_e2e_sharded.yml` is an opt-in reusable workflow for a private
repository's protected manual workflow. It collects the complete hosted pytest
manifest, verifies serving identity, warms up once, runs four serial pytest
processes, and requires complete successful evidence from all four shards.
It does not deploy, publish or make the release decision. Existing manual
workflows are unchanged; migrate their use before enabling parallel acceptance.

## Scheduling

All cases in a file stay together, including authentication and parameter
variants. Files are sorted by descending estimated duration and assigned to
the least-loaded shard, with deterministic tie-breaking. A few expensive cases
can therefore occupy a whole shard while another shard holds many cheap cases.
This is a scheduling heuristic, not a proof of optimal runtime or safe load.

The private caller can provide `.github/hosted-e2e/timing_profile.json`:

```json
{
  "schema": 1,
  "target": "staging",
  "files": {
    "tests/inference/hosted_platform_tests/example.py": {
      "seconds": 30.0,
      "cases": 6,
      "source_hash": "SHA256_OF_FILE_CONTENTS"
    }
  }
}
```

Use the latest release's matching suite as the baseline and separate staging
and production profiles. Record source run, attempt and estimation method in
additional profile metadata. Historical progress timestamps provide approximate
file costs only: include wrapped progress lines and retry time, and exclude the
first file when its interval includes session warmup. Do not manufacture
per-case timings from those file estimates.

File estimates apply only when both source hash and collected case count match.
Missing or changed files use the median estimated seconds per case from valid
file estimates, multiplied by their own case count. With no timing data, the
fallback is case count; it is not a measured duration. An optional private
`durations.json` maps node IDs to measured seconds, including retry overhead.
Complete case timings override a file estimate. Refresh these case timings when
test code changes. The plan records weight provenance, assignments and their
hash; no assignment changes during a run.

File grouping preserves known same-file writes but does not establish provider
quota independence. Validate shared credentials, provider fan-out, dataset
writes and external locks before setting `load_validated: true`. Concurrency
serializes this workflow only within one caller repository. Other writers and
other caller repositories must share an external lock or be disabled during
acceptance. Splitting a slow file further requires an isolation review.

## Private caller contract

1. Use `workflow_dispatch` from the private repository's protected default
   branch. Protect that branch and restrict who may dispatch and change its
   workflow, secrets and verification adapter. The reusable workflow rejects
   public callers, other events, other refs and an unset load-validation gate.
2. Pin the reusable workflow to a reviewed full commit SHA. Pass that same SHA
   as `runner_sha`, and a full reviewed `test_sha`. Both must be reachable from
   `roboflow/inference` main. Implementation must be merged before live use;
   unmerged feature branches and pull-request code cannot access credentials.
3. Pass `target: staging` or `production` and `load_validated: true` only after
   validating the environment. Keep deployment identity, endpoints and project
   identifiers in private configuration/secrets, not public workflow inputs.
4. Explicitly map the secrets declared by the reusable workflow. Use narrow,
   short-lived credentials where supported. Do not use `secrets: inherit`.
   Missing provider credentials can cause skips; empty or invalid credentials
   can cause authentication failures. Unexpected skips and failed tests both
   prevent acceptance. Tests checking `KEY is None` do not skip for an empty
   string supplied through an environment variable.
5. Supply the private adapter below and an explicit `allowed_skips.json` list
   of collected node IDs for deliberately disabled tests. The default allowlist
   is empty. Stale allowlist entries and unexpected skips fail acceptance.
   Collection-time skips (for example, module-level `pytest.skip` or
   `pytest.importorskip`) require a separate `allowed_collection_skips.json`
   list of exact collector node IDs, such as
   `["tests/inference/hosted_platform_tests/test_example.py"]`. This list also
   defaults to empty. Preparation records collector skips in `collection.json`
   and rejects unexpected skips or stale entries before creating the plan.
   Approved collector skips are frozen in the plan and must match during
   warmup, every shard and aggregation; they do not represent executed cases.
6. Place the selected environment's timing profile and skip allowlists in
   `.github/hosted-e2e/` at the private caller's pinned commit. The adapter must
   use the standard library or the already installed test dependencies.

Example job in the private caller (replace the placeholder with a reviewed SHA):

```yaml
jobs:
  acceptance:
    permissions:
      contents: read
    uses: roboflow/inference/.github/workflows/hosted_inference_e2e_sharded.yml@REVIEWED_COMMIT_SHA
    with:
      runner_sha: REVIEWED_COMMIT_SHA
      test_sha: ${{ inputs.test_sha }}
      target: staging
      load_validated: ${{ inputs.load_validated }}
    secrets:
      DEPLOYMENT_REFERENCE: ${{ secrets.DEPLOYMENT_REFERENCE }}
      EXPECTED_CANDIDATE_SHA: ${{ secrets.EXPECTED_CANDIDATE_SHA }}
      EXPECTED_IMAGE_DIGEST: ${{ secrets.EXPECTED_IMAGE_DIGEST }}
      DEPLOYMENT_VERIFICATION_TOKEN: ${{ secrets.DEPLOYMENT_VERIFICATION_TOKEN }}
      HOSTED_PLATFORM_TESTS_PROJECT: ${{ secrets.HOSTED_PLATFORM_TESTS_PROJECT }}
      HOSTED_PLATFORM_TESTS_API_KEY: ${{ secrets.HOSTED_PLATFORM_TESTS_API_KEY }}
      OPENAI_KEY: ${{ secrets.OPENAI_KEY }}
      GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      GOOGLE_VISION_API_KEY: ${{ secrets.GOOGLE_VISION_API_KEY }}
```

## Serving identity and readiness

The private caller provides `.github/hosted-e2e/verify_deployment.py`. It must
independently verify the deployment serving the configured test target, its
routing, readiness, source revision and image digest against a trusted control
plane. Echoing environment variables is not verification. For a multi-image
deployment, provide a trusted immutable digest binding the complete deployment
and reject mixed revisions. No generic endpoint adapter can establish this.

The adapter exits nonzero on uncertainty and emits exactly one JSON object:
`ready` (true), `deployment_ref`, `candidate_sha` (40 lowercase hex characters)
and `image_digest` (`sha256:` followed by 64 lowercase hex characters). The
runner compares these with the expected private inputs. Adapter stdout/stderr
is captured and is not echoed on failure. It runs before collection, before
and after warmup, and before and after each shard. Point-in-time checks cannot
prove no transient routing changes; the caller must hold the deployment stable
for the entire acceptance run.

The frozen context binds run ID/attempt, target, test SHA, orchestration SHA,
private policy SHA, observed identity and installed dependency versions. Shards
install preparation's frozen versions and reject environment drift. This is
version reproducibility, not a hash-locked supply-chain guarantee.

## Evidence and acceptance

Preparation uploads its plan, readiness, warmup evidence and frozen dependency
versions. Shards retain per-case setup/call/teardown outcomes and durations,
including failed retries. The pinned pytest-retry integration preserves the
suite's existing retry policy. Aggregation checks exact coverage, readiness,
context, exit codes, fixture outcomes and allowed skips. Missing, cancelled,
duplicated or inconsistent results cannot pass. A successful retry remains
visible in the private report. Aggregation is scheduled after failed shards;
hard cancellation or runner loss may prevent artifact creation and requires a
fresh complete run, never acceptance from partial evidence.

A skip allowlist cannot turn an unresolved test failure into acceptance. A
failed call followed by a skipped retry is rejected even when the case is
allowlisted. Successful retries and clean, explicitly allowed skips remain valid.

After a failed or incomplete run, use **Re-run all jobs** or start a new complete
workflow run. Do not use **Re-run failed jobs**: a rerun increments `run_attempt`,
but preparation evidence from successful jobs remains under the previous
attempt's artifact name. Restarted shards cannot find a plan for their new
attempt, and copying the earlier artifact would still fail the context check.
Preparation, warmup, all four shards and aggregation must belong to the same
attempt. These checks intentionally prevent mixing evidence across attempts.

Logs, case IDs, timing profiles, dependency lists, deployment metadata and
artifacts remain in the private caller; artifacts expire after three days.
Only the fixed `accepted`/`failed` status is suitable for public export. There
is no automatic public export in this workflow. Never copy raw traces or
payloads into public issues, PRs or artifacts. These controls do not retroactively
change existing public-workflow logs.

Offline validation runs synthetic tests without hosted credentials or network:

```sh
python -m unittest discover -s .github/scripts -p 'test_hosted_e2e.py'
```

Use Python 3.12 with pytest 8.4.2, pytest-retry 1.6.3, Click and PyYAML. Live
rollout still requires private caller configuration, maintainer review,
deployment verification and measured load/isolation validation.
