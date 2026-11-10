# PyPI and inference release runbook

This runbook covers separate SDK/runtime publication, dependency coordination and
verified retries of missing release files. SDK and runtime packages still share
`inference/core/version.py`; runtime/CLI wheels still require exactly that SDK
version. Independent SDK version numbering is not implemented.

## Existing PyPI authorization

Both publication jobs stay in `publish.pypi.yml` in `roboflow/inference`, with
no GitHub environment. All six projects already use this workflow's OIDC
identity. No new trusted publisher, API token, or personal PyPI access is needed
for this split. Keep the existing PyPI publisher entries, including the SDK's.
The SDK upload and five-package upload have separate jobs and inventories.
The three standalone publishers also keep their original workflow filenames and
trusted-publisher identities.

Before merging, review the Docker validation and build-only runs for all four
package modes. Preserve the workflow filename and environment configuration:
changing either would require coordination with a PyPI project owner.

## Record the candidate

Use a frozen candidate branch/tag for every dispatch and record its resolved
commit SHA. `gh workflow run --ref` selects a branch/tag; it does not change the
version in source. Check that each run's SHA is the approved candidate.

Retain a release record with package versions, selected upstream versions,
workflow URLs, image digests, test commands/results, and publication receipts.
SDK/runtime build artifacts are retained for 90 days; archive needed evidence
before expiration. Their names include the build attempt. Rerunning a failed
publication job uses its original build output; rerunning the build retains a
separate artifact. Standalone publisher jobs rebuild on rerun, so freeze both
source and toolchain and inspect the verification result before proceeding.
A source, metadata, dependency or toolchain change invalidates affected evidence.

All local validation must execute in Docker, including smoke tests, benchmarks
and reproductions. Record image ID/digest, candidate revision, actual command and
result. Missing local Docker/dependencies/hardware is NO-GO for that validation;
there is no local host fallback. GitHub Actions workflows retain their configured
runner or container execution. The local Docker requirement does not require
skipping normal PR CI or migrating unrelated CI suites to Docker.

## Publisher controls

| Workflow filename | Default manual run | Publishing inputs |
| --- | --- | --- |
| `publish.pypi.yml`, `packages=sdk` | SDK build only | `publish=true`, `expected_version=<SDK version>` |
| `publish.pypi.yml`, `packages=runtime` | Five-package build only | `publish=true`, `expected_version=<shared version>` |
| `publish.pypi.yml`, `packages=all` (default) | SDK and five-package builds | `publish=true`, `expected_version=<shared version>` |
| `publish.pypi.yml`, `packages=dependencies` | SDK/runtime builds and dependency plan | `publish=true`, `expected_version=<shared version>`; uploads upstream dependencies only |
| `publish.pypi.inference_exp.yml` | Models build only | `publish=true` |
| `publish.pypi.roboflow_workflows.yml` | Workflows build only | `publish=true` |
| `publish.pypi.streamvision.yml` | Streamvision build only | `publish=true` |

In **Publish Wheels to PyPi**, GitHub `release.created` selects both package
groups, enables publication automatically and checks the release tag against
package metadata. Manual runs default to `packages=all`, `publish=false`.
SDK-only mode skips both runtime jobs. Runtime-only mode skips both SDK jobs
and requires the exact SDK to have been published already. In all/release mode,
the runtime publication job waits for successful SDK publication and verification;
an SDK failure or cancellation blocks runtime upload. Each runtime upload also
verifies the actual SDK files in PyPI, including runtime-only dispatches.

Runtime/all/dependencies modes resolve first-party dependencies from the built
wheel metadata. They reuse complete compatible published versions, or dispatch
missing same-checkout Models, Workflows and Streamvision versions in dependency
order before runtime publication. An incomplete known standalone inventory can
be repaired only when its selected version equals the compatible project version
in this checkout. The original child publisher rebuilds it and verifies existing
files before uploading missing files. An older incomplete version cannot be
repaired from a different checkout. Changed source alone does not force a new
upload when a complete compatible published dependency can be reused.
Build-only runs emit the plan without dispatching child publishers, staging,
uploading or running final publication verification. Dependency planning still
reads PyPI and verifies published dependency metadata and files.
The selected versions, file hashes and child run IDs are retained as evidence.
The runtime publisher verifies the selected dependencies again before upload.

`packages=dependencies` performs the same preparation, including SDK publication,
but never uploads the five runtime packages. Use it before image validation and
GitHub Release creation. SDK-only mode remains independent.

Build-only examples (replace the ref/version):

```sh
gh workflow run publish.pypi.yml --repo roboflow/inference \
  --ref release/candidate -f packages=sdk -f publish=false -f expected_version=1.9.0rc2
gh workflow run publish.pypi.yml --repo roboflow/inference \
  --ref release/candidate -f packages=runtime -f publish=false -f expected_version=1.9.0rc2
gh workflow run publish.pypi.yml --repo roboflow/inference \
  --ref release/candidate -f packages=all -f publish=false -f expected_version=1.9.0rc2
```

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
3. Dispatch `publish.pypi.yml` with `packages=sdk`, `publish=false`. Inspect the
   retained wheel and `sdk-manifest.json`; verify version and dependency metadata.
4. Dispatch the same workflow with `packages=sdk`, `publish=true` and
   `expected_version`. Example (replace values):

   ```sh
   gh workflow run publish.pypi.yml --repo roboflow/inference \
     --ref release/sdk-1.9.0rc2 -f packages=sdk \
     -f publish=true -f expected_version=1.9.0rc2
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
  python -m pytest .github/scripts/test_pypi_release.py .github/scripts/test_pypi_dependencies.py .github/scripts/test_pypi_publish.py -v
```

## Standalone package release

Run the changed package's validation and build-only publisher first. Publish
only required upstream versions, verify they are available, then publish the
consumer. Models and SDK can proceed independently; Workflows needs compatible
Models and SDK; Streamvision's workflows extra needs compatible Workflows.
Existing compatible versions can be reused. Do not raise dependency floors or
bump unrelated packages merely to make all version numbers move together.

Standalone publishers are manual workflows; a full release dispatches them from
`publish.pypi.yml` only when its plan selects a missing version or an eligible
incomplete inventory. They no longer start independently on a GitHub Release
event. Their filenames and PyPI trust
identities are unchanged. Each manual publisher checks the first-party
dependencies in its built wheel before uploading; a missing dependency blocks
that publication. Publish its required upstream first, then rerun the consumer.
Standalone publication attempts are serialized without cancelling an active
attempt. The coordinator sets `expected_version` and `expected_revision`;
operators may also set these inputs to bind a standalone run to a reviewed
candidate.

The resolver checks exact pins, version ranges, provided extras and Python
3.10–3.13 compatibility. It includes internal dependencies declared by all extras
and platform markers, failing conservatively if their constraints conflict.
It prefers published non-yanked universal wheels. If no published version fits,
the same-checkout project version must fit; otherwise update versions/pins/locks
explicitly before releasing. Third-party dependency resolution remains the job
of package locks and installation tests.

Before publication, inspect both source declarations and built metadata.
Validate applicable locks for Models, Workflows and Streamvision; local checks
must run in Docker.
`make create_models_wheel` does not itself perform a lock check. Models and
Workflows publishers build both wheel and sdist; preserve both in evidence.

## Full release

1. Select versions and required upstream releases. Update source versions,
   requirements pins, pyprojects, locks and version-consistency checks together.
   Freeze and push each candidate after relocking; test the pushed SHA.
2. Build and validate the candidate packages. Dispatch `publish.pypi.yml` with
   `packages=dependencies`, `publish=false` and the expected source version.
   Review the retained plan. Then dispatch the same mode with `publish=true`.
   It publishes/verifies the SDK, reuses existing upstreams and dispatches missing
   Models, Workflows and Streamvision versions or eligible incomplete inventories
   in dependency order. Require the coordinator to succeed before proceeding.
   The `publish_sdk` job is the sole
   SDK upload owner. A full release still needs the exact shared SDK version.
3. For a public RC, publish required RC packages in that order, then dispatch
   `publish.pypi.yml` with `packages=runtime`, `publish=true` and its expected
   RC version.
   It verifies the SDK and selected upstream dependencies before uploading any
   runtime consumer. Missing standalone versions are coordinated automatically.
4. Build the required legacy CPU/GPU/Jetson candidate images, record their
   digests, run runtime checks, deploy staging and run staging acceptance tests.
5. If another candidate is needed, repeat with RC2, RC3, etc. Record new evidence;
   do not search only for `rc1` or reuse results after relevant changes.
6. Prepare stable versions and locks. Build and validate stable artifacts again:
   RC and stable wheels contain different metadata and are not byte-identical.
   Repeat `packages=dependencies` for the stable version and inspect its receipts.
   RC and stable selections are separate. Reuse unchanged compatible upstreams.
7. Merge the approved stable candidate, check its final SHA and dependency
   selections, then create the GitHub Release `v<VERSION>` at that commit.
   **Do not manually publish the five stable packages before this event.**
   The release automatically starts both package groups in `publish.pypi.yml`
   plus existing image workflows. The SDK verifies its prepublished
   wheel and becomes a no-op when contents match. Runtime upload waits for that
   verification and dependency coordination. Standalone publishers run only if
   required by the plan. Monitor these jobs rather than launching duplicates.
   Use the verified retry procedure below if publication fails.
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

These GitHub Actions workflows provide release validation using their existing
runner or container configuration. Record each run URL, candidate revision and
result. The Docker requirement applies to local execution and reproductions;
direct execution on a CI runner does not invalidate a workflow's evidence. This
change does not require a Docker migration of server, Models, Workflows or
hosted-test infrastructure.

### Images and deployment

| Workflow | Manual candidate inputs |
| --- | --- |
| `docker.cpu.yml`, `docker.gpu.yml` | `force_push=true`, `custom_tag` as explained below |
| `docker.jetson.5.1.1.yml`, `docker.jetson.6.2.0.yml`, `docker.jetson.7.2.0.yml` | Same push/tag inputs, only for target platforms; 7.2 `rebuild_media=false` unless needed |
| `modal.custom_python_block.deploy.yml` | `inference_version=<existing CPU image tag>`, `targets=us,eu`, `stage=staging`; later `stage=prod` |
| `hosted_inference_e2e_test_staging.yml`, `hosted_inference_e2e_test_production.yml` | `test_target=serverless-v2`, `skip_warmup=false` |

`custom_tag` is a **suffix**, despite the current UI description. Source version
`1.9.0rc2` plus an empty suffix gives `1.9.0rc2`; adding `rc2` would give the
unintended `1.9.0rc2-rc2`. Source `1.9.0` plus suffix `rc2` gives `1.9.0-rc2`.
Confirm the final tag before dispatch. For build-only image checks use
`force_push=false`. Prefer an immutable amd64 digest for server integration;
`image_digest` and `image_tag` are mutually exclusive.

Keep the legacy server image default. Do not run
`publish.pypi.inference_server_stack.yml` or `docker.inference_server.yml` as part
of this release path. Production image build behavior is unchanged by the
publisher split.

The Makefile `upload_wheels` target still invokes raw Twine and is outside the
verified workflow contract. Automatic package selection from source changes,
Docker Hub coordination, test/approval orchestration and cross-package RC/post
suffix enforcement are not implemented. Existing PEP 440 RC/post versions remain
supported; choose versions and suffixes explicitly.

## Retry and recovery

The dependency coordinator dispatches publishers using GitHub's returned run ID,
checks that run's source SHA and workflow identity, and waits for success before
checking PyPI. A moved candidate ref, child failure/cancellation, incompatible
metadata, authentication error or timeout blocks runtime upload. Registry
verification is read-only and checks downloaded wheels against registry hashes.
Reusing a released dependency does not claim it was built from this checkout.

Runtime and standalone workflows use `bash .release/pypi/publish.sh prepare`
before upload and `verify` afterward, selecting a `--profile`, the built
`--directory`, a missing-only `--upload-directory` and a JSON `--receipt`.
`--wait-seconds 600` bounds registry visibility retries. Uploads are performed by
the existing workflow publishing action from that staged directory, without
`skip-existing`.

| Profile | Complete inventory for one version |
| --- | --- |
| `runtime` | One wheel each for inference, inference-core, inference-cpu, inference-gpu and inference-cli |
| `inference-models` | One wheel and one sdist |
| `roboflow-workflows` | One wheel and one sdist |
| `streamvision` | One wheel |

Before any upload, preparation checks the exact known inventory and downloads
every existing file. It verifies each registry SHA-256 and compares the complete
archive against the candidate. Entry names, contents, metadata and permissions
must match. ZIP timestamps/compression and tar/gzip timestamps, owner IDs and
entry order may differ. Code, dependency metadata, README, license, RECORD and
WHEEL metadata may not. A toolchain change can alter metadata and require a new
version.

| Observed state | Decision and operator action |
| --- | --- |
| Version absent (version endpoint HTTP 404) | Stage the complete candidate inventory and upload it. |
| Some expected files published, all matching | Stage and upload only the missing files. |
| Complete inventory, all matching | Upload nothing; verify the complete published inventory. |
| Conflicting contents, yanked or unexpected files, empty/malformed existing inventory, registry hash mismatch | Stop before upload; investigate and use a new version for changed contents. |
| Authentication/registry failure or visibility timeout | Stop; inspect the error and retry the same frozen candidate after resolving it. Failures never mean “missing”. |

Preparation verifies all existing files before staging any upload. Final
verification requires the complete inventory, including newly uploaded files,
with bounded visibility retries. Receipts record candidate and published hashes,
source revision, verification image and run; they do not establish the original
upload's provenance.

1. Inspect the failed run, retained artifacts, receipts, dependency plan and
   child run URLs. A timeout, lost dispatch response or coordinator cancellation
   can leave a child running; inspect Actions before starting another attempt.
2. Keep the candidate ref frozen. For runtime publication, rerun the failed
   `publish_runtime` job to reuse its retained five wheels. For a standalone
   publisher, rerun its original workflow with the same candidate inputs; it
   rebuilds and verifies every existing file before uploading any missing file.
   Freezing the ref alone does not freeze toolchain output. A rebuilt content
   conflict requires investigation and a new version, rather than bypassing
   verification.
3. If dependency coordination failed, rerun it after inspecting any child run.
   It replans against current PyPI, reuses complete compatible dependencies and
   dispatches eligible same-checkout repairs in dependency order. Require
   successful child and registry verification before runtime publication.
4. Require final complete-inventory verification and retain its receipts with
   the candidate SHA, image ID and workflow URL. Never delete or overwrite
   published files to make a retry pass. Workflow rollback cannot undo uploads.

The SDK keeps its existing `release.sh prepare-sdk` / `verify-sdk` path. Only an
absent version permits upload. An existing SDK must pass inventory, yanked-state,
registry SHA-256 and complete wheel contents/permissions checks against the
candidate; the same ZIP timestamp/compression tolerance applies.

On an SDK upload failure, rerun the failed `publish_sdk` job to reuse its retained
wheel; GitHub also reruns its dependent `publish_runtime` job in all/release mode.
For SDK-only mode, runtime jobs stay skipped. If the upload actually completed,
verification will detect it and skip a duplicate upload. Missing, yanked or
conflicting SDKs block consumer upload.

If rolling back the split, restore the previous `publish.pypi.yml` behavior as
one change. PyPI trust remains unchanged. Never leave two active SDK upload jobs
or no SDK upload job.
