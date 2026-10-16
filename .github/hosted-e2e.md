# Hosted platform E2E execution

See the [changelog](hosted-e2e-changelog.md) for changes and the rationale behind
the initial runtime-balanced shard proposal.

## Existing manual staging and production workflows

The existing **E2E TEST - hosted inference - STAGING** and **PRODUCTION** manual
workflows now call `hosted_inference_e2e_repository.yml` in this repository.
Dispatch from the default branch, select `serverless-v2`, and leave
`skip_warmup` false unless deliberately skipping service warmup. Execution is:

1. Prepare the complete manifest and frozen dependencies; warm services once.
2. Run shards A, B, C and D concurrently, with serial pytest inside each job.
3. Aggregate all four reports, rejecting missing evidence, failures and
   unexpected skips, even when another job failed.

Credentialed jobs require a manual run on the default branch or an approved
release fast-track head.
All checkouts use the dispatch SHA. Both targets share one concurrency group.
PR events and arbitrary feature-branch dispatches do not run credentialed tests. This does not
block a separate external system from writing to the same test resources.

For release fast-tracks, use `fast-track/post-vX.Y.Z-ptN` cut from the matching
release tag. The gate requires an open, non-draft, same-repository umbrella PR
to main carrying `fast-track-deploy-approved`, whose head SHA exactly matches
the dispatch SHA. It verifies ancestry against the release tag. Missing approval,
a moved PR head, wrong ancestry, or API failure fails the guard before hosted
credentials are available. This follows the existing release approval label;
it does not automatically dispatch hosted tests from the fast-track CI workflow.
Dispatch the staging manual workflow explicitly on the approved branch.

The existing API/provider secrets are reused. Five cases deliberately disabled
in the test source are listed in `.github/hosted-e2e/allowed_skips.json`; the
collection-skip allowlist is empty. Empty provider credentials fail preparation.
Nothing automatically expands the allowlist from observed skips.

Optional secrets `HOSTED_E2E_STAGING_TIMING_PROFILE` and
`HOSTED_E2E_PRODUCTION_TIMING_PROFILE` hold the environment-specific JSON
profiles described below. Without a profile, scheduling falls back to case
counts. Profiles are never committed here or printed. Changed files invalidate
individual estimates rather than blindly reusing stale measurements.

### Public-repository output and encrypted evidence

Treat all Actions output and artifacts as potentially accessible outside the
team, regardless of whether a particular signed-out browser can open them.
The stage wrapper captures pytest stdout and stderr into a local file. Only
fixed stage statuses and aggregate counts go to the visible log. Artifacts
contain AES-256-GCM ciphertext, including detailed test reports, logs and the
prepared dependency list; plaintext evidence is never uploaded. Artifact
retention is three days. Artifact names contain only run/attempt/stage IDs.

HKDF derives the encryption key from the existing hosted-test API credential,
with a distinct domain and authenticated repository, revision, target, run and
attempt context. Every bundle has a random nonce. A different credential or
context, a changed ciphertext, conflicting reports or unexpected paths makes
restoration fail. Rotating the API credential makes older evidence unreadable
unless the old credential is available through the authorized secret store.
This protects artifact contents, not job names, status or runtime metadata.

For restricted investigation, download the encrypted artifacts to an isolated
local directory and provide the original hosted API credential from the team's
secret store through `HOSTED_PLATFORM_TESTS_API_KEY` (never a command argument).
Set `GITHUB_REPOSITORY=roboflow/inference`, `GITHUB_EVENT_NAME=workflow_dispatch`,
`GITHUB_REF` and `HOSTED_E2E_DEFAULT_REF` to the selected branch ref and the default branch ref respectively,
`HOSTED_E2E_APPROVED_REF` and `HOSTED_E2E_APPROVED_SHA` to the verified ref and SHA,
`GITHUB_SHA`, `HOSTED_E2E_TEST_SHA` and `HOSTED_E2E_RUNNER_SHA` to the tested
commit, `HOSTED_E2E_RUN_KEY` to `RUN_ID:ATTEMPT`, and the original
`HOSTED_E2E_TARGET` and `HOSTED_PLATFORM_TESTS_PROJECT`. Then use the reviewed
runner's `restore --source incoming --output evidence` command. It restores
structured evidence only and deliberately omits raw `stage.log` files; raw logs
remain encrypted. GitHub does not provide a way to read back secret values.
Do not publish decrypted reports or attach them to public issues.

These manual runs verify the hosted suite against its configured target. They
**do not establish the deployed source revision or image digest**, matching the
previous manual workflows. Their evidence explicitly says deployment identity
was not verified. They do not deploy, publish or authorize a release.

Use **Re-run all jobs**, not **Re-run failed jobs**. Both artifact names and
authenticated encryption bind to the run attempt. Preparation, warmup and all
four shards must be rerun together. Validate actual load and timing in staging
before dispatching production.

**Rollout status:** wiring and offline tests are implementation evidence. Live
activation is incomplete until reviewed code is on the default branch or an approved fast-track and an
actual staging run shows preparation, all four shards and successful complete
aggregation. Record that run before claiming the solution is operational.

## Separate private deployment-acceptance workflow

`hosted_inference_e2e_sharded.yml` is an opt-in reusable workflow for a private
repository's protected manual workflow. It collects the complete hosted pytest
manifest, verifies serving identity, warms up once, runs four serial pytest
processes, and requires complete successful evidence from all four shards.
It does not deploy, publish or make the release decision. The repository manual workflows above use a separate encrypted execution path.
This stronger deployment-acceptance path remains opt-in and requires its own caller.

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
python -m unittest discover -s .github/scripts -p 'test_hosted_e2e*.py'
```

Use Python 3.12 with pytest 8.4.2, pytest-retry 1.6.3, Click, PyYAML and cryptography 46.0.3. The repository manual path requires
a verified live staging run. The separate private deployment-acceptance path
also requires its private caller and trusted deployment-verification adapter.
