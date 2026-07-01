---
name: review-packaging-ci
description: Review guidance for PRs touching .github/ (workflows, actions, prompts, CODEOWNERS), docker/ (dockerfiles, publish, config), requirements/, .release/pypi/, and Makefile — enforces the packaging & CI standards, release/version invariants, dependency-pin discipline, and Docker/GHA companion requirements the inference maintainers uphold.
---

# Reviewing packaging-ci changes

## Scope
Trigger this skill when a PR changes any of:
- `.github/workflows/**`, `.github/actions/**`, `.github/prompts/**`, `.github/CODEOWNERS` / `CODEOWNERS`
- `docker/dockerfiles/**`, `docker/publish/**`, `docker/config/**`, `docker/entrypoint/**`
- `requirements/*.txt`
- `.release/pypi/*.setup.py`
- `Makefile`
- `inference/core/version.py`, `inference_models/pyproject.toml` / `inference_models/uv.lock` (version + pin sync — see Companions)

OUT of scope (defer to other skills): library/business-logic changes under `inference/`, `inference_models/inference_exp/` source, `inference_sdk/`, `inference_cli/`; Workflows blocks/kinds; test *content* (this skill covers only which CI job runs the tests and on what runner/deps).

## What this surface is
Packaging-CI is the release & build machinery: GitHub Actions workflows that build/push Docker images (Docker Hub + GCP Artifact Registry) and publish PyPI wheels; the Dockerfiles that assemble every deployment target (CPU, GPU, VINO, TRT, Lambda, 4 Jetson variants, Colab, Windows/OSX bundles); the `requirements/*.txt` pin sets consumed by both `setup.py`-derived wheels (`.release/pypi/*.setup.py`) and the Dockerfiles; and the `Makefile` glue (`create_wheels`, `create_inference_cli_whl`). Contracts a reviewer must protect:
- **Version is single-sourced.** Image tags and wheel versions derive from `inference/core/version.py` (`__version__`), read in CI via `DISABLE_VERSION_CHECK=true python ./inference/core/version.py` (docker.cpu.yml). `inference-models` is versioned separately in `inference_models/pyproject.toml`.
- **`inference-models` pin must stay in lockstep across 4 files.** `requirements.cpu.txt`, `requirements.gpu.txt`, `requirements.vino.txt`, `requirements.jetson.txt` each pin `inference-models~=X.Y.Z` and carry the literal comment `# keep in sync between ...` (#2341). A bump in one without the others is a defect.
- **Requirements feed wheels too.** `.release/pypi/*.setup.py` and `Makefile:create_wheels` read the same `requirements/*.txt`; a bad pin breaks both Docker and PyPI paths.
- **Tag/registry semantics.** `latest` is only added for real releases, never RCs (#1983), and version normalization strips a leading `v` (#1470) — both live in `.github/actions/determine-tags/action.yml`.

## Standards enforced here
- **Every workflow declares least-privilege `permissions:`.** Top-level `permissions: contents: read`, with job-level escalation only where needed (`id-token: write` for GCP OIDC in docker.*.yml; `issues/pull-requests: write` in claude-pr-review.yml). Added en masse in #1242 to fix a code-scanning alert, and re-required per-workflow in #1484. A new workflow without `permissions:` is a block.
- **`inference-models` version pins stay synced** across the 4 requirements files, with the sync comment preserved (#2341, #2144). Verify all four move together.
- **Dependency pins are bounded, and additions are surgical.** Maintainers add upper bounds proactively (`aiohttp` #2071, `supervision<0.26.0` #1424/#1425, `typing-extensions` loosen #2072) and pin transitive deps that broke silently even when not directly listed (`tokenizers` #839, `timm` #1879, `aiortc` #989). A bare `>=` with no upper bound on a fast-moving lib warrants a comment asking for justification.
- **Jetson/onnxruntime numpy discipline.** Jetson Dockerfiles enforce `numpy<=1.26.4` for onnxruntime compatibility and this must run AFTER the inference-cli wheel install, or the wheel drags numpy 2 back in (#1441, #1457). The `# BE CAREFUL, WE ENFORCE numpy 1.x` comment blocks must not be silently deleted or reordered.
- **Requirement-file paths in Dockerfiles must be correct.** COPY/`-r` paths to `requirements/*.txt` are a recurring break source (#1595 "wrong path to requirements file"). Cross-check every `-r requirements.*.txt` against the files that are COPY-ed in.
- **GHA `workflow_dispatch` boolean inputs are strings.** Compare with `== 'true'`, never truthiness: `push: ${{ github.event_name == 'release' || (github.event.inputs.force_push == 'true') }}` (#522 fixed broken type coercion). Flag any `github.event.inputs.<bool>` used bare in an `if`/`push`.
- **Pinned runners/base images intentionally.** Python 3.9 Dockerfiles pinned to `bookworm` after Debian trixie broke them (#1491); Jetson images pin `setuptools<=75.5.0`, `pycuda`, and (Jetson 6/7) `transformers>=4.57.3,<5.9.0` (#2385, #2339). Base-image or tag changes need a stated reason.
- **CI jobs run on runners that actually have the system libs the target needs.** `inference-gpu` install tests require `cuda.h`/GPU libs → must run on a GPU machine (#2147); experimental e2e needs `libgles2`/`libegl1`/`libvips-dev` apt-installed (#2294). A new/moved test job must name its runner and preinstall OS deps.
- **Timeouts on long jobs.** Docker builds set `timeout-minutes` (120 in docker.cpu.yml); global limits added in #596/#635. New build/test jobs should carry a timeout.
- **Action versions and formatting.** Uses pinned major action versions (`actions/checkout@v6`, `depot/build-push-action@v1`, `docker/login-action@v3/@v4`); keep quoting consistent (#2326 normalized quoting/`token_format`). Cosmetic-only churn is acceptable but shouldn't hide behavior changes.

## Required companions
Block the PR if a change lacks its companion:
- **Release PR** (title like "Release X.Y.Z", #2341/#2175/#2102): must bump `inference/core/version.py` `__version__` AND `inference_models/pyproject.toml` `version` AND `inference_models/uv.lock` AND all 4 `inference-models~=` pins in `requirements/*.txt`. RC→final drops the `rc` suffix in every one of those places.
- **`inference-models` version bump anywhere** → the other 3 requirements files + `pyproject.toml` + `uv.lock` (#2341). Never one-of-four.
- **`inference_models/pyproject.toml` dependency add/bump** → regenerated `inference_models/uv.lock` in the same PR (#2301 added `Pygments` to both), and a `inference_models/docs/changelog.md` entry under the new version header for user-facing changes.
- **New Docker image target** → its `.github/workflows/docker.<name>.yml` + Dockerfile under `docker/dockerfiles/` + (if published) publish script; and `permissions:` block. Removing a target removes all of these (#2149 deleted stream-manager workflows).
- **New PyPI wheel / requirements group** → wire it into `.release/pypi/*.setup.py` and/or `Makefile:create_wheels` and the relevant `test_package_install_*.yml` install test.
- **New dependency needed at runtime** → add to the correct `requirements/*.txt` AND every Dockerfile whose target uses it (system libs via `apt/dnf`, e.g. `libvips` #1209, `rustc` #975). Don't add to requirements only.
- **New GHA workflow** → `permissions:` (top-level `contents: read`), `timeout-minutes`, and a `concurrency:` group for PR-triggered jobs.

## Common pitfalls & past regressions
- **#1441 / #1457** — Jetson numpy re-pin placed BEFORE cli-wheel install; wheel pulled numpy 2 back. Order matters: force numpy 1.x AFTER wheel installs.
- **#522** — `github.event.inputs.force_push` used truthily; string `"false"` is truthy → images pushed unintentionally. Always `== 'true'`.
- **#1470** — version-tag regex `[0-9]$` didn't match 2-digit patch (e.g. `.10`) and failed to strip leading `v`. Regex is `^v?[0-9]+\.[0-9]+\.[0-9]+$`.
- **#1983** — `latest` tag was pushed for RC releases. Gate: `&& [[ "$NORMALIZED_RELEASE" != *rc* ]]`.
- **#1595 / #1166 / #810 / #807** — broken Docker builds from wrong `requirements/*.txt` COPY paths or unpinned libs; a "trivial" Dockerfile edit can break the whole image build.
- **#839 / #1879 / #989 / #363 / #374** — a transitive/indirect dep's new release silently broke installs (`tokenizers`, `timm`, `aiortc`, `transformers`, `fastapi`); fix is a defensive pin even for deps not directly listed.
- **#1132** (revert of #1123) — numpy 2 allowed then reverted across dockerfiles + requirements; numpy major bumps are high-risk and touch many files.
- **#2301** — Lambda build broke on stale `certifi==2022.12.07` and a missing `Pygments` pin; wheel/lock and dockerfile pins must agree.
- **#2223** — macOS PyInstaller hook redefined `hiddenimports=[]`/`datas=[]`, wiping collected entries. Bundle-hook edits need extra scrutiny.
- **#2188** — Windows installer split CPU/GPU; wrong wheel glob (`inference_gpu-*` vs `inference-*`) selects the wrong package.
- **#2147 / #780** — install-test jobs failing because the runner lacked system libs (`cuda.h`) or carried leftover debug steps; match job to runner capabilities.

## Review checklist
1. If version changed anywhere, confirm ALL of: `inference/core/version.py`, `inference_models/pyproject.toml`, `inference_models/uv.lock`, and the 4 `inference-models~=` requirement pins move together; RC suffix consistent.
2. For every touched `requirements/*.txt`: is the pin bounded (upper limit) and, for `inference-models`, is the `# keep in sync` comment intact across all 4 files?
3. If `inference_models/pyproject.toml` deps changed → is `uv.lock` regenerated and is there a `changelog.md` entry?
4. New/edited workflow: has top-level `permissions:` (least privilege), `timeout-minutes`, `concurrency:` for PR triggers, and pinned action versions?
5. Any `github.event.inputs.<bool>` compared with `== 'true'` (not bare)?
6. Dockerfile edits: are all `-r requirements/*.txt` paths COPY-ed and correct? Is Jetson numpy-1.x enforcement present and AFTER wheel install? Base-image/tag pins justified?
7. New runtime dep: added to correct requirements file AND every Dockerfile target that needs it (plus OS/system libs)?
8. New/moved test or install job: does its runner have the required system libs (GPU/`cuda.h`, `libvips`, `libgles2`/`libegl1`)?
9. New image target: workflow + Dockerfile + publish script + install test all present? Removed target: all removed?
10. Tagging logic touched (`determine-tags/action.yml`): `latest` still gated off RCs, leading-`v` stripped, regex handles multi-digit patch?
11. Transitive-dep break suspected (a new upstream release broke the build)? Confirm the pin is defensive and commented.
12. CODEOWNERS / claude-pr-review prompt changes: reviewed by an existing codeowner (#455, #2228, #2135).

## Key files & entry points
- `inference/core/version.py` — canonical server version.
- `inference_models/pyproject.toml`, `inference_models/uv.lock`, `inference_models/docs/changelog.md` — inference-models version + changelog.
- `requirements/requirements.{cpu,gpu,vino,jetson}.txt` — the 4 in-sync `inference-models` pin files; `requirements/_requirements.txt` — shared core pins.
- `.github/actions/determine-tags/action.yml` — tag/normalization logic.
- `.github/workflows/docker.*.yml` — image build/push (canonical: `docker.cpu.yml`); `publish.pypi*.yml` — wheel publishing.
- `docker/dockerfiles/Dockerfile.onnx.*` — per-target build recipes (`.jetson.*` are the fragile ones).
- `.release/pypi/*.setup.py`, `Makefile` (`create_wheels`, `create_inference_cli_whl`) — wheel assembly.
- `CODEOWNERS`, `.github/prompts/claude-pr-review.md` — ownership + review-agent config.

## Reference PRs
- [#2341](https://github.com/roboflow/inference/pull/2341) — Release 1.2.8: version + 4-file pin sync (the release template).
- [#2326](https://github.com/roboflow/inference/pull/2326) — build once, push to Docker Hub + GCP; canonical docker workflow shape.
- [#1242](https://github.com/roboflow/inference/pull/1242) / [#1484](https://github.com/roboflow/inference/pull/1484) — least-privilege `permissions:` across workflows.
- [#1983](https://github.com/roboflow/inference/pull/1983) — don't push `latest` for RC releases (tag semantics).
- [#1470](https://github.com/roboflow/inference/pull/1470) — version-tag regex / leading-`v` strip fix.
- [#522](https://github.com/roboflow/inference/pull/522) — GHA boolean input `== 'true'` coercion bug.
- [#1441](https://github.com/roboflow/inference/pull/1441) — Jetson: install cli before forcing numpy 1.x.
- [#839](https://github.com/roboflow/inference/pull/839) / [#1879](https://github.com/roboflow/inference/pull/1879) — defensive pins for transitive-dep breaks.
- [#2301](https://github.com/roboflow/inference/pull/2301) — Lambda build fix: certifi/Pygments pin + uv.lock sync.
- [#2385](https://github.com/roboflow/inference/pull/2385) — pin `transformers<5.9` in Jetson 6/7 Dockerfiles.
- [#2147](https://github.com/roboflow/inference/pull/2147) — run gpu-install test on runner with required system libs.
- [#2149](https://github.com/roboflow/inference/pull/2149) — CI/docker/release cleanup after inference-models switch (target removal).

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-test-hygiene`
