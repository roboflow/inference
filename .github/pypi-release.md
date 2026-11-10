# PyPI and inference release runbook

This runbook works with the SDK publisher split alone. Dependency selection and
generalized retry handling are later changes, not prerequisites. SDK and runtime
packages still share `inference/core/version.py`; runtime/CLI wheels still require
exactly that SDK version. Independent SDK version numbering is not implemented.

## One-time cutover before merging

1. In the **inference-sdk** PyPI project's Publishing settings, add a trusted
   publisher with owner `roboflow`, repository `inference`, workflow filename
   `publish.pypi.inference_sdk.yml`, and no environment (the workflow does not
   declare one). Keep the existing five projects' publishers unchanged.
2. Confirm that repository Actions policies permit the new workflow and that
   the build runners have Docker. Keep this PR draft until configuration and
   required validation evidence have been reviewed.
3. Merge both publishers together. Never remove SDK ownership from the old
   workflow in a separate deployment. After cutover, remove the old
   `publish.pypi.yml` trusted-publisher entry from **inference-sdk only**.

PyPI configuration is external to Git and is not changed by merging this PR.
See [PyPI's trusted-publisher configuration](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

## Record the candidate

Use a frozen candidate branch/tag for every dispatch and record its resolved
commit SHA. `gh workflow run --ref` selects a branch/tag; it does not change the
version in source. Check that each run's SHA is the approved candidate.

Retain a release record with package versions, selected upstream versions,
workflow URLs, image digests, test commands/results, and publication receipts.
Artifacts are retained for 90 days; archive needed evidence before expiration.
Artifact names include their build attempt. Rerunning a failed publication job
uses its original build output; rerunning the build retains a separate artifact.
A source, metadata, dependency or toolchain change invalidates affected evidence.

All validation must execute in Docker, including smoke tests, benchmarks and
reproductions. Record image ID/digest, candidate revision, actual command and
result. Missing Docker/dependencies/hardware is NO-GO; there is no host fallback.

## Publisher controls

| Workflow filename | Default manual run | Publishing inputs |
| --- | --- | --- |
| `publish.pypi.inference_sdk.yml` | SDK build only | `publish=true`, `expected_version=<SDK version>` |
| `publish.pypi.yml` | Five-package build only | `publish=true`, `expected_version=<shared version>` |
| `publish.pypi.inference_exp.yml` | Models build only | `publish=true` |
| `publish.pypi.roboflow_workflows.yml` | Workflows build only | `publish=true` |
| `publish.pypi.streamvision.yml` | Streamvision build only | `publish=true` |

For the SDK and five-package workflows, GitHub `release.created` enables
publication automatically and checks the release tag against package metadata.
The five-package workflow retains its filename and name, **Publish Wheels to
PyPi**, so its existing trusted publishers remain valid.

Build jobs have no OIDC permission. They retain the exact wheels later consumed
by separate publication jobs. The five-package upload directory contains only
inference, inference-core, inference-cpu, inference-gpu and inference-cli. It
also builds an SDK reference in a different directory, for the availability
check; that reference is never uploaded by the five-package publisher.

## SDK-only release

1. Prepare the shared source version and freeze the candidate ref. Changes to
   SDK requirements, packaging metadata or README are SDK artifact changes too.
2. Run `Test PyPI release contracts` (`test.pypi.release.yml`) and the SDK checks
   listed below on that ref. Inspect the commands and image IDs in their logs.
3. Dispatch the SDK publisher with `publish=false`. Inspect the retained wheel
   and `sdk-manifest.json`; verify the version and dependency metadata.
4. Dispatch with `publish=true` and `expected_version`. Example (replace values):

   ```sh
   gh workflow run publish.pypi.inference_sdk.yml --repo roboflow/inference \
     --ref release/sdk-1.9.0rc2 -f publish=true -f expected_version=1.9.0rc2
   ```

5. Require the publication job to finish its registry verification. Retain
   `sdk-publication-receipt` along with the build artifact and validation runs.
6. Install the published wheel in a clean Docker container and confirm its
   version/import outside the checkout. Record the image, exact version,
   command and result; the earlier installation CI checks the candidate wheel,
   not the live registry installation.

Do not create a repository-wide GitHub Release for an SDK-only publication: it
would start unrelated package/image releases. Do not manually run the five-
package publisher, Models/Workflows/Streamvision publishers, server image builds,
Modal deployment or hosted tests solely because the SDK is being released.

### SDK validation commands

These existing workflow names now execute their tests in Docker. Their matrices
cover Python 3.10, 3.11, 3.12 and 3.13; no manual flags are required.

| Workflow | Coverage | Local Docker orchestration |
| --- | --- | --- |
| `unit_tests_inference_sdk_x86.yml` | SDK units | `bash .github/scripts/run_sdk_tests.sh unit 3.12` |
| `integration_e2e_tests_inference_sdk_x86.yml` | Mocked integration and real same-checkout server E2E | `bash .github/scripts/run_sdk_tests.sh integration 3.12`; `bash .github/scripts/run_sdk_tests.sh e2e 3.12` |
| `test_package_install_inference_sdk.yml` | Clean SDK-only install and import outside checkout | `bash .github/scripts/run_sdk_tests.sh install 3.12` |

Replace `3.12` to reproduce another matrix entry. E2E builds server dependencies
inside its validation container; it does not publish those dependencies or any
server image. SDK publishing itself installs only packaging tools.

For release-helper tests:

```sh
docker build -t inference-pypi-tests -f .release/pypi/Dockerfile .release/pypi
docker image inspect inference-pypi-tests --format '{{.Id}}'
git rev-parse HEAD
docker run --rm -v "$PWD:/work" -w /work inference-pypi-tests \
  python -m pytest .github/scripts/test_pypi_release.py -v
```

## Standalone package release

Run the changed package's validation and build-only publisher first. Publish
only required upstream versions, verify they are available, then publish the
consumer. Models and SDK can proceed independently; Workflows needs compatible
Models and SDK; Streamvision's workflows extra needs compatible Workflows.
Existing compatible versions can be reused. Do not raise dependency floors or
bump unrelated packages merely to make all version numbers move together.

Before publication, inspect both source declarations and built metadata.
Validate applicable locks for Models, Workflows and Streamvision in Docker.
`make create_models_wheel` does not itself perform a lock check. Workflows builds
both wheel and sdist, despite its Make target's name; preserve both in evidence.

## Full release with only PR1 merged

1. Select versions and required upstream releases. Update source versions,
   requirements pins, pyprojects, locks and version-consistency checks together.
   Freeze and push each candidate after relocking; test the pushed SHA.
2. Build and validate required upstream candidates. Publish SDK and Models as
   needed, then Workflows, then Streamvision, verifying each before its consumer.
   The SDK-only workflow is the sole SDK upload owner. A full release still
   needs a new SDK version when its exact shared-version pin requires one.
3. For a public RC, publish required RC packages in that order, then dispatch
   the five-package publisher with `publish=true` and its expected RC version.
   It waits up to ten minutes for a matching SDK before uploading any consumer.
4. Build the required legacy CPU/GPU/Jetson candidate images, record their
   digests, run runtime checks, deploy staging and run staging acceptance tests.
5. If another candidate is needed, repeat with RC2, RC3, etc. Record new evidence;
   do not search only for `rc1` or reuse results after relevant changes.
6. Prepare stable versions and locks. Build and validate stable artifacts again:
   RC and stable wheels contain different metadata and are not byte-identical.
   Publish and verify required stable SDK/Models/Workflows/Streamvision versions
   before stable consumers. Reuse unchanged compatible upstream releases.
7. Merge the approved stable candidate, check its final SHA and dependency
   selections, then create the GitHub Release `v<VERSION>` at that commit.
   **Do not manually publish the five stable packages before this event.**
   The release automatically starts the SDK and five-package workflows plus
   existing standalone/image workflows. The SDK verifies its prepublished wheel
   and becomes a no-op when contents match. Monitor the automatic jobs rather
   than launching duplicates. The remaining publishers retain their existing
   behavior, including their current retry limitations.
8. Verify stable package installation and image digests, complete production
   deployment/handoffs, and run production acceptance checks. Retain the record.

### Other validation selection and flags

For a full server release, include relevant runtime/platform coverage as well
as validation of changed standalone packages. SDK-only releases do not require
unrelated GPU or Jetson suites. Do not repeat unchanged exact-revision evidence.

| Workflow | Inputs |
| --- | --- |
| `static_code_analysis.yml` | None |
| `unit_tests_inference_cli_x86.yml`, `integration_tests_inference_cli_x86.yml`, `integration_tests_inference_cli_depending_on_inference_x86.yml`, `test_package_install_inference_cli.yml` | None |
| `unit_tests_inference_x86.yml` | `run_stream_benchmark_checks=false`; enable only for required benchmarking |
| `test_package_install_inference.yml` | None |
| `unit_tests_workflows_x86.yml`, `unit_tests_workflows_tensor_native_x86.yml` | None |
| `integration_tests_workflows_x86.yml` | `python-version=all`, `use-inference-models=all` |
| `integration_tests_workflows_tensor_native_x86.yml` | `python-version=all` |
| `unit_tests_streamvision_x86.yml`, `unit_tests_inference_experimental.yml` | None |
| `integration_tests_inference_experimental_cpu.yml` | `extras=""`, `python_version=""` for the full available matrix |
| `integration_tests_inference_experimental_gpu.yml` | `extras=""`, `python_version=3.12`; requires GPU hardware |
| `integration_tests_inference_server_x86.yml` | `server-image=legacy`, `use-inference-models=all`, `image_digest=<amd64 digest>`, `image_tag=""` |

This table identifies coverage and current controls, not certification that
every existing workflow runs in Docker. Non-SDK workflows that execute checks
on the host cannot count as release validation under the Docker policy. Use a
reviewed Docker execution path for their suites and record equivalent coverage;
if one is unavailable, that gate remains NO-GO. PR1 does not rewrite all server,
Models, Workflows or hosted-test infrastructure. This pre-existing prerequisite
is not supplied by PR2 or PR3 either.

### Images and deployment

| Workflow | Manual candidate inputs |
| --- | --- |
| `docker.cpu.yml`, `docker.gpu.yml` | `force_push=true`, `custom_tag` as explained below |
| `docker.jetson.5.1.1.yml`, `docker.jetson.6.2.0.yml`, `docker.jetson.7.2.0.yml` | Same push/tag inputs, only for target platforms; 7.2 `rebuild_media=false` unless needed |
| `modal.custom_python_block.deploy.yml` | `inference_version=<existing CPU image tag>`, `targets=us,eu`, `stage=staging`; later `stage=prod` |
| `hosted_inference_e2e_test_staging.yml`, `hosted_inference_e2e_test_production.yml` | `test_target=serverless-v2`, `skip_warmup=false`; test execution must be in Docker |

`custom_tag` is a **suffix**, despite the current UI description. Source version
`1.9.0rc2` plus an empty suffix gives `1.9.0rc2`; adding `rc2` would give the
unintended `1.9.0rc2-rc2`. Source `1.9.0` plus suffix `rc2` gives `1.9.0-rc2`.
Confirm the final tag before dispatch. For build-only image checks use
`force_push=false`. Prefer an immutable amd64 digest for server integration;
`image_digest` and `image_tag` are mutually exclusive.

Keep the legacy server image default. Do not run
`publish.pypi.inference_server_stack.yml` or `docker.inference_server.yml` as part
of this release path. Production image build behavior is unchanged by PR1.

## Retry and recovery

The SDK publisher checks PyPI before upload. Only an absent version permits
upload. For an existing version, it checks the inventory, yanked state, registry
SHA-256 and every wheel entry's contents and permissions against the candidate.
ZIP timestamps/compression may differ; package code, dependency metadata,
README, license, RECORD and WHEEL metadata may not. Toolchain changes can alter
WHEEL metadata and require a new version. Verification receipts record both the
candidate's hash and the actual published file hash, plus the verification run.
They do not claim to identify a previous upload's original workflow run.

On an SDK upload failure, rerun the failed publication job to reuse its retained
wheel. If the upload actually completed, verification will detect it and skip a
duplicate upload. Missing, yanked or conflicting SDKs block consumer upload.
Registry/authentication failures never mean "missing". Transient errors and
visibility delays have bounded retries; inspect the reported error on timeout.

The five-package publisher is **not yet generally retry-safe**. After a partial
upload, stop and inventory PyPI; do not add `skip-existing` or blindly rerun it.
General per-file recovery is PR3. Never delete/overwrite an existing release to
make it match. A contents conflict requires investigation and usually a new
version. Workflow rollback cannot undo already published files.

If rolling back the ownership split, restore both workflow changes together
and restore the SDK's old trusted publisher first. Never leave two active SDK
upload owners or no owner.
