# Release preparation and retries

Choose package versions, dependency constraints and runtime pins on a fixed
source revision before starting a release. A coordinated release regenerates
the companion locks automatically; unpublished first-party versions do not
need to resolve before their upstream publisher runs.

## Initial run

Use the existing release event, or dispatch `publish.pypi.yml` once with
`packages=all`, `publish=true` and the matching `expected_version`. These are
publishing operations and require the maintainer's release authorization.

The coordinator verifies SDK publication before coordinating Models,
Workflows and Streamvision in dependency order. Each new companion publisher
downloads the retained dependency plan, runs the pinned `uv lock` with its
selected upstream versions, checks the lock, builds and retains its candidate,
then publishes through its existing workflow/OIDC identity. A stale lock or an
RC-to-final transition requires no separate manual lock edit. A preparation
failure stops that publisher and its dependents.

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
The existing RC/stable tag decisions are unchanged.

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
   Ordinary non-release builds without that input retain their existing behavior.

Missing, expired, ambiguous or changed artifacts stop the run. Restore/reconcile
the original evidence; do not regenerate a retained candidate or start a second
publisher blindly. Changed source or dependency choices need a new candidate
version. Artifacts are retained for 90 days.

Package retries reuse exact distribution bytes. Image retries reuse exact source
and lock inputs but may rebuild images; this does not promise identical image
digests or add GPU acceptance coverage.
