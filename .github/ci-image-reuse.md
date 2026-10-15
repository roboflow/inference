# CPU/GPU CI image reuse

## Implementation plan

**Author:** [PR #3200 author](https://github.com/roboflow/inference/pull/3200)

**Related PR:** [#3200](https://github.com/roboflow/inference/pull/3200)

This plan follows the [contribution template](implementation-plan-template.md).
The team has agreed to the approach, as confirmed by the PR author. Private
discussion details are intentionally omitted. Infrastructure configuration and
live validation remain required before activation.

### 1. What problem are we solving?

Maintainers need both server test modes to validate the images produced for a
main commit without rebuilding identical source in every consumer. Today, a
single main push starts independent CPU/GPU producers plus two CPU and two T4
consumer legs, each of which builds its own image. Passing tests therefore do
not identify the separate producer artifacts, and consumers repeat build work.

### 2. How will the behaviour change?

Before, the same main push fans out to independent builds:

```mermaid
flowchart LR
    M[Main commit] --> P[Independent CPU and GPU producers]
    M --> C[CPU true and false modes each rebuild]
    M --> G[T4 true and false modes each rebuild]
    C --> T[Tests run against consumer-built images]
    G --> T
```

After activation, the same main push connects producers and consumers:

```mermaid
flowchart LR
    M[Main commit] --> C[Build CPU amd64 and arm64 once]
    M --> G[Build GPU amd64 once]
    C --> CT[CPU true and false consume the same amd64 digest]
    G --> GT[T4 true and false consume the same amd64 digest]
    CT --> Q[Require both complete test matrices]
    GT --> Q
```

`ci_image_reuse.yml` coordinates the standard server image builds and tests for
one main commit. CPU builds amd64 and arm64 in one producer invocation; GPU
builds amd64 once. The existing CPU and T4 workflows each run both
`USE_INFERENCE_MODELS` modes against their producer's amd64 digest. T4 retains
the complete existing selection, including runtime checks and all model stages.
The separate models-GPU, package, parallel-server and other workflows continue
independently.

Consumers pull by digest, verify architecture and the source revision label,
then run by local image ID. Failed pulls never trigger a replacement build.
Test containers have job-specific names and dynamically allocated loopback
ports. The CPU PyTorch runtime check is retained in the consumer. CPU arm64 is
built but is **not runtime-qualified** by these x86 tests.

The existing reusable test workflows retain their suites and manual inputs,
which keeps the change compatible with fast-track callers. Offline contract
tests and workflow linting verify the wiring; the live trial described below
must verify transport and timing before activation.

### 3. Uncertainties and decisions needing input

| Question | Investigation | Recommendation | Input needed |
| --- | --- | --- | --- |
| Which approved private transport and identities should be used? | Existing producer configuration does not establish a private CI boundary; the new path accepts only external secret parameters. | Use an approved private namespace and separate scoped OIDC reader/writer identities. | Infrastructure maintainers must designate the configuration owner and verify access and retention before activation. No credential values belong in this plan. |
| Are runners and builder caches sufficiently isolated? | Unique container names and loopback ports prevent accidental naming/port collisions; they cannot isolate jobs sharing a Docker daemon. | Use isolated ephemeral runners and a trusted builder/cache boundary. | Runner and builder owners must confirm isolation before activation. |
| Which required-check names should protect main? | The orchestrator adds caller prefixes to reusable job names and introduces an aggregate image-test check. | Review protection expectations while retaining all independent required suites. | Repository maintainers must approve the required-check mapping before activation. |
| Does reuse improve end-to-end duration? | Offline tests verify routing and failure handling, not transfer costs or GPU queueing. | Measure the complete candidate path and GPU occupancy in a full-coverage trial. | An authorized trial is required; no numerical performance claim is made. |
| What qualifies CPU arm64 for release? | Existing server consumers execute on amd64 only. | Retain the arm64 build and report it separately from tested amd64 images. | Release maintainers must define runtime coverage before treating a multi-platform image as release-qualified; this does not block preparing the reuse path. |

These decisions block activation, not preparation of the disabled implementation.
The team discussion required by
[AGENTS.md](../AGENTS.md#internal-contributions-roboflow-team-only) has been
confirmed by the PR author; code review and the activation prerequisites still
apply.

## Change record — Unreleased

### Review fixes

- Preserve Buildx state until its post-job cleanup by removing only Docker
  credentials, preventing orphaned builder containers on persistent runners.
- Keep image-reuse tests explicitly scheduled in the GitHub Actions validation
  workflow after incorporating the hosted-E2E changes from main.

### Added

- Add opt-in CPU/GPU producer-to-consumer reuse by immutable digest, preserving
  both model modes and the existing server test scope.
- Validate artifact identity, isolate test container names and host ports, and
  require complete producer/test results before the aggregate check passes.
- Parameterize credentials, restrict the new transport to trusted main runs,
  and pin external actions in the affected workflows to verified commit SHAs.

This is a CI infrastructure change. It changes no runtime package, execution
engine behavior, package dependency or release version. The package-specific
`Unreleased` requirements in [AGENTS.md](../AGENTS.md#versioning) and the
[engine rule](../.cursor/rules/execution-engine-version-changelog.mdc) therefore
do not require entries in package changelogs or version bumps for this PR.
This section records the CI change without assigning it to a package release.

## Configuration and activation

The implementation is disabled until the repository variable
`CI_IMAGE_REUSE_ENABLED` is exactly `true`. With it unset or false, the existing
main producer and consumer jobs continue. With it true, those four independent
main jobs skip and the orchestrator owns their builds and tests. Release and
manual/fast-track entry points retain their existing behavior.

Configure these GitHub **secret parameters** outside the repository. Never put
their actual values in source, issues, public summaries or examples:

| Parameter | Purpose |
| --- | --- |
| `CI_REGISTRY_PREFIX` | Approved private Artifact Registry namespace, in `HOST/PROJECT/REPOSITORY` form, without a trailing slash |
| `CI_WIF_PROVIDER` | OIDC workload identity provider resource |
| `CI_REGISTRY_WRITER` | Identity with write access restricted to the CI repository |
| `CI_REGISTRY_READER` | Separate identity with read-only access to that repository |
| `CI_DEPOT_PROJECT` | Approved builder project and cache boundary |

The registry namespace contains `inference-ci-cpu` and `inference-ci-gpu` images.
Transport tags include source SHA, run ID and attempt. They are retention
anchors; consumers use platform digests. Release registries, Docker Hub
credentials, `latest`, version tags and RC tags are not used by this path.

Before activation, the infrastructure owner must confirm private access at both
repository and inherited policy levels, retention long enough for failed-job
retries, and OIDC restrictions to the expected repository identity, main ref,
event types and workflow identities. The writer must not be able to publish to
release repositories. Restrict the builder/cache trust boundary likewise. Use
isolated ephemeral runners; unique container names do not provide isolation
from another job with access to the same Docker daemon. Configure no credentials
or private assets for untrusted PR jobs.

Do not enable the switch with missing or unverified configuration. Once enabled,
configuration errors fail closed; they do not fall back to public registries or
the previous independent builds. Review required-check names before activation:
reusable jobs acquire the orchestrator's caller prefixes. This change does not
modify repository protection settings.

## Credential and output handling

Authentication uses short-lived OIDC-derived access tokens, passed to Docker
through stdin. Reader jobs receive only reader configuration and the existing
test secret parameters; producer jobs receive no test or Docker Hub secrets.
Generated authentication files are disabled. Docker credentials are placed in
an isolated temporary directory. Cleanup removes only `config.json` before
tests and on job completion, preserving Buildx builder records for the action's
post-job cleanup. The remaining temporary directory is left to runner cleanup;
it is not removed by a normal step before action post hooks run.
Existing test credentials are passed through step environment parameters rather
than interpolated into shell code.

Producer context is a `git archive` of the exact candidate, rather than the
working directory. Runner credentials, generated logs and untracked files
cannot enter that context. Subprocess build/pull/inspection output is withheld
from public logs, including on failure. No private build record, image archive,
SBOM or test data is uploaded as a GitHub artifact. Minimal provenance remains
with the image in its configured registry. Detailed transport diagnostics are
not retained by this workflow; investigate through the approved private builder
and registry tools. Do not enable public debug tracing to retrieve them.

Digest outputs and the public commit/run identity are not credentials and are
used to connect the jobs. Public summaries contain only coverage/status text.
Existing test output still follows the server suites' logging behavior; review
that output before introducing additional private test data. Secret masking is
not a substitute for keeping secrets out of application logs.

## Completion and retries

The aggregate `CPU and GPU image tests complete` check requires both producers,
both complete test matrices and all required platform digest outputs. Failure,
missing output, skipped work or cancellation prevents it from passing. Matrix
fail-fast is disabled so one model mode does not cancel the other mode's work.

Use **Re-run failed jobs** to retain successful producer outputs and retest the
same images. Re-running every job creates a new producer attempt and may produce
different images, even for the same source SHA. Retention must preserve previous
attempts' manifests. Do not substitute a tag or silently rebuild a missing image.

This is a server-image test gate, not full release qualification. Other required
checks, ARM64 runtime policy, detailed private qualification evidence and RC
promotion remain separate. Existing preprocessing stages are preserved; the
pre-existing Make target does not forward `USE_PYTORCH_FOR_PREPROCESSING`, so
distinct effective preprocessing configurations still need a separate fix.

Rollback sets `CI_IMAGE_REUSE_ENABLED` to false for subsequent runs, restoring
the previous independent main builds, including CPU's existing mutable `:test`
publication. Do not toggle it while a candidate is running: finish or cancel
that run first so old and new jobs cannot overlap unpredictably.

## Validation

Offline checks (no registry writes, GPU execution or test credentials):

```sh
python .github/scripts/test_ci_image.py
python .github/scripts/check_workflow_concurrency.py
actionlint
```

The existing GitHub Actions validation workflow discovers `test_ci_image.py`
with the other script tests. Tests cover malformed/conflicting inputs, legacy
tags, trusted-source checks, identity mismatches, pull failures, platform
manifests, single producer invocation, private-output suppression, isolated
ports, unchanged test scope and incomplete aggregate outcomes.

Before rollout, run an explicitly authorized full-coverage trial against the
approved private transport. Verify permissions, real digest pulls, runtime
identity, cleanup, retries and required-check names. Measure producer export,
consumer pull, GPU occupancy and complete candidate duration. Offline checks do
not establish performance gains or production registry/runner compatibility.
