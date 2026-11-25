# Release preparation and retries

Choose package versions, dependency constraints and runtime pins on a fixed
source revision before starting a release. A coordinated release regenerates
the companion locks automatically; unpublished first-party versions do not
need to resolve before their upstream publisher runs.

## Initial run

All package and image release workflows use only `release: published`, including
publishing a draft or prerelease. Draft creation alone starts no release work.
Do not subscribe to both `created` and `published`: duplicate coordinators are
ambiguous. See [GitHub release events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#release).

Use that release event, or dispatch `publish.pypi.yml` once with
`packages=all`, `publish=true` and the matching `expected_version`. These are
publishing operations and require the maintainer's release authorization.

The coordinator verifies SDK publication before coordinating Models,
Workflows and Streamvision in dependency order. Each new companion publisher
downloads the retained dependency plan, runs the pinned `uv lock` with its
selected upstream versions, checks the lock, builds and retains its candidate,
then publishes through its existing workflow/OIDC identity. A stale lock or an
RC-to-final transition requires no separate manual lock edit. A preparation
failure stops that publisher and its dependents. Resolver failures get at most
three attempts with 10/20-second backoffs inside one 600-second deadline, to
allow a newly published upstream to reach the simple index. Exhaustion restores
the original lock; retry the original coordinator once the index is ready.

Lock regeneration is intentional, including stale committed locks and standalone
build-only runs. Coordinated first-party versions are fixed by the retained plan;
standalone runs select compatible available first-party versions. uv retains
unrelated pins where possible, but changed constraints can require third-party
changes. Review uv's resolver output and the original/prepared lock hashes in the
retained receipt; preparation does not promise zero third-party drift.

The dependency-preparation job checks all companion source versions and dependency
edges against the plan before coordinating publishers or allowing runtime upload.
Runtime-only publication also produces image inputs when its three prerequisites
succeed, even though the SDK jobs were intentionally skipped.

After runtime publication succeeds, `prepare_image_inputs` collects the exact
locks from the companion candidates. For already-published companions, it
prepares their source locks against the same dependency plan without rebuilding
or publishing their distributions. The immutable `candidate-image-inputs`
artifact contains the source revision, release tag, package manifests, locks,
preparation receipts, dependency plan and originating publisher identities.
The release tag and repository files are not rewritten; the bundle records the
generated inputs alongside the fixed source commit.

Release image workflows wait for the unique coordinator with the same release
tag and source SHA. They require its successful completion, download its exact
artifact, verify all metadata/hashes, and restore the locks before building from
the local checkout. Image labels record the source SHA and input-manifest digest.
The existing RC/stable tag decisions are unchanged. All 13 image workflows wait
for package success intentionally: this is a release-wide readiness gate, even
when a Dockerfile does not use a companion lock.

| Release image workflows | Lock consumed by the image build |
| --- | --- |
| `docker.gpu`, `docker.gpu.cu13`, `docker.trt` | `inference_models/uv.lock` |
| `docker.inference-exp` | `inference_models/uv.lock` via its Models build context |
| `docker.cpu`, `docker.cpu.parallel`, `docker.cpu.slim`, `docker.gpu.parallel`, `docker.gpu.slim`, `docker.gpu.udp`, `docker.jetson.5.1.1`, `docker.jetson.6.2.0`, `docker.jetson.7.2.0` | None; shared release readiness and provenance only |

No current release Dockerfile consumes `workflows/uv.lock` or
`stream_vision/uv.lock`; those are retained publisher inputs. A common source SHA
and input bundle do not make all image dependency graphs identical.

The gate and restore steps build the existing release-tooling Docker image. This
adds runner time and Docker Hub/Debian/PyPI availability requirements. Docker layer
caching can reduce repeated setup; a separately published tooling image is outside
this change.

`publish=false` on the coordinator builds SDK/runtime candidates and plans
dependencies; it does **not** publish upstreams or produce a complete release
image bundle. Individual companion build-only runs regenerate their locks but
require their dependencies to be available. PR contract tests exercise an
unpublished RC chain with an isolated fixture index and no publication.

## Retry

1. Rerun failed jobs in the **original coordinator run**. It reconnects to
   original companion runs and retains the original dependency selection.
2. A retained companion restores its original locks and distribution bytes;
   it neither resolves nor rebuilds. Failure before successful candidate
   retention may retry preparation. A retained image-input bundle also skips
   assembly and resolution.
3. Rerun failed jobs in the **original image run** after the coordinator is
   successful. Its retained coordinator selection prevents switching to another
   release run. A readiness timeout can be retried after the coordinator finishes.
4. For a new manual image run using prepared release inputs, select the same
   source revision and supply `release_run_id=<original coordinator run ID>`.
   Non-release builds without that input skip the release gate. Build contexts
   now explicitly use the local checkout, including on push/manual paths, so
   restored inputs are visible; `.dockerignore` excludes release staging data.
   Non-release images have an empty prepared-input digest label.

The image gate has a 360-minute job limit: selection can wait 5 minutes and the
coordinator wait 350 minutes, with setup using the remaining time. The coordinator
critical path has up to 445 minutes of job budgets plus queueing. A slow successful
release can therefore time out image gates. Let the original coordinator finish,
then rerun the original failed image jobs; their retained selection still identifies
the same coordinator. Do not extend polling beyond the hosted-runner job limit.

Do not delete and recreate a release at the same tag/SHA as a retry. Multiple
matching coordinator runs are deliberately rejected rather than guessing which
package bytes are authoritative. Reconcile the original runs, then use an explicit
`release_run_id` for manual image recovery if necessary. Manual runs retain manual
tag rules and do not promote release-only stable aliases.

Missing, expired, ambiguous or changed artifacts stop the run. Restore/reconcile
the original evidence; do not regenerate a retained candidate or start a second
publisher blindly. Changed source or dependency choices need a new candidate
version. Artifacts are retained for 90 days.

Package retries reuse exact distribution bytes. Image retries reuse exact source
and lock inputs but may rebuild images; this does not promise identical image
digests or add GPU acceptance coverage.
