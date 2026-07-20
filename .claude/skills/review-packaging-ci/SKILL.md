---
name: review-packaging-ci
description: Review guidance for PRs touching .github/ (workflows, actions, prompts/claude-pr-review.md, CODEOWNERS), docker/ (dockerfiles, publish, config, entrypoint), requirements/*.txt, .release/pypi/*.setup.py, Makefile, inference/core/version.py, and inference_models/{pyproject.toml,uv.lock}. Diff signals: new/edited GitHub Actions workflow or action.yml, Dockerfile edits, requirements-pin bumps, version bumps, tag/publish logic.
---

# Reviewing packaging-ci changes

## Scope
Trigger when a PR changes any of:
- `.github/workflows/**`, `.github/actions/**`, `.github/prompts/claude-pr-review.md`, `CODEOWNERS`
- `docker/dockerfiles/**`, `docker/publish/**`, `docker/config/**`, `docker/entrypoint/**`
- `requirements/*.txt`
- `.release/pypi/*.setup.py`
- `Makefile`
- `inference/core/version.py`, `inference_models/pyproject.toml`, `inference_models/uv.lock` (version + pin sync)

OUT of scope (defer): library/business logic under `inference/`, `inference_models/inference_models/` source, `inference_sdk/`, `inference_cli/`; Workflows blocks/kinds; test *content* (this skill covers only which CI job runs tests and on what runner/deps).

## Review checklist

**BLOCK**
- Release PR (`inference/core/version.py` `__version__` changed) → confirm ALL of `inference_models/pyproject.toml` (`version`), `inference_models/uv.lock`, and the 4 `inference-models~=` requirement pins move together; RC suffix consistent everywhere (#2341). The reverse does NOT hold: an `inference_models` version bump alone does not require an `inference/core/version.py` bump (the package versions independently).
- `inference-models` pin bumped in one requirements file but not all 4 (`requirements.{cpu,gpu,vino,jetson}.txt`), or the `# keep in sync` comment deleted (#2341, #2144).
- `inference_models/pyproject.toml` deps changed with no regenerated `inference_models/uv.lock` in the same PR (#2301).
- New/edited workflow missing top-level `permissions:` (least privilege) (#1242, #1484).
- Third-party GitHub Action referenced by a mutable tag/branch instead of a full commit SHA (supply-chain: a compromised upstream tag runs in CI with repo permissions).
- `run:` step interpolates untrusted `${{ github.event.* }}` (PR title/body, issue/comment text) directly into a shell command (script injection) — pass via `env:` and reference `"$VAR"` instead.
- `github.event.inputs.<bool>` used bare in an `if`/`push` instead of `== 'true'` — string `"false"` is truthy (#522).
- Dockerfile `-r requirements/*.txt` / `COPY` path that is not actually COPY-ed in, or wrong (#1595, #1166, #810).
- New runtime dep added to `requirements/*.txt` only, not to every Dockerfile target (+ OS/system libs) that needs it (#1209, #975).
- New/moved test or install job whose runner lacks the required system libs (GPU `cuda.h`, `libgles2`/`libegl1`/`libvips-dev`) (#2147, #2294).

**FLAG**
- New image target missing any of: `.github/workflows/docker.<name>.yml`, Dockerfile under `docker/dockerfiles/`, publish script (if published), `permissions:` block; or a removed target that left orphans (#2149).
- New PyPI wheel / requirements group not wired into `.release/pypi/*.setup.py` and/or `Makefile:create_wheels` and the matching `test_package_install_*.yml`.
- Bare `>=` with no upper bound on a fast-moving lib (ask for justification) (#2071, #1424).
- Suspected transitive-dep break with no defensive, commented pin (#839, #1879, #989).
- Base-image / runner / pinned-action-version change with no stated reason (#1491, #2385).
- New build/test job without `timeout-minutes` (#596, #635).
- New PR-triggered workflow without a `concurrency:` group.
- Secret referenced in a workflow that could be exfiltrated on a fork/`pull_request_target` trigger, or a new secret without an owner.

**NIT**
- Quoting / `token_format` / cosmetic YAML churn (acceptable, but must not hide a behavior change) (#2326).

### Not blocking
- Numpy is pinned `numpy>=2.0.0,<2.4.0` repo-wide (`_requirements.txt` + Jetson Dockerfiles build torch/opencv from source for numpy 2.x). Do NOT ask for a numpy-1.x pin — that constraint is gone. Only flag if a change reintroduces an unbounded numpy or a conflicting per-target pin.
- Adding an upper bound to a dependency proactively is a normal maintainer move, not a regression — don't demand a linked failure for every bound.
- Cosmetic reordering / comment rewording that preserves behavior.
- A version bump PR that touches many files is expected; the release template legitimately spans version.py + pyproject + uv.lock + 4 requirements files.
- Do NOT demand an `inference/core/version.py` bump on non-release PRs — inference releases are versioned separately; the lock-step checks above apply only when a version actually changes in the diff.

## Standards

- **Version is single-sourced.** Image tags and wheel versions derive from `inference/core/version.py` (`__version__`), read in CI via `DISABLE_VERSION_CHECK=true python ./inference/core/version.py`. `inference-models` is versioned separately in `inference_models/pyproject.toml` (`version`).
- **`inference-models` pin stays in lockstep across 4 files.** `requirements.{cpu,gpu,vino,jetson}.txt` each pin `inference-models~=X.Y.Z` and carry the `# keep in sync between ...` comment. Verify all four move together and match `inference_models/pyproject.toml` (#2341).
- **Requirements feed wheels too.** `.release/pypi/inference*.setup.py` and `Makefile:create_wheels` read the same `requirements/*.txt`; a bad pin breaks both Docker and PyPI paths.
- **Least-privilege `permissions:`.** Every workflow declares top-level `permissions: contents: read`, escalating per-job only where needed (`id-token: write` for GCP OIDC in `docker.*.yml`; `issues: write` + `pull-requests: write` in `claude-pr-review.yml`) (#1242, #1484).
- **CI supply-chain & injection hygiene.** This review bot itself runs on PRs. Third-party actions must be SHA-pinned; `run:` steps must not interpolate untrusted `${{ github.event.* }}` (pass through `env:`); new workflows must not expose secrets to fork/`pull_request_target` runs.
- **Dependency pins are bounded; additions are surgical.** Maintainers add upper bounds proactively and pin transitive deps that broke silently even when not directly listed (`tokenizers` #839, `timm` #1879, `aiortc` #989). Numpy is `numpy>=2.0.0,<2.4.0`.
- **Requirement-file paths in Dockerfiles must be correct.** Cross-check every `-r requirements.*.txt` / `COPY` against the files actually COPY-ed in — a recurring break source (#1595).
- **GHA `workflow_dispatch` boolean inputs are strings.** Compare with `== 'true'`, e.g. `push: ${{ github.event_name == 'release' || (github.event.inputs.force_push == 'true') }}` (#522).
- **Pinned runners/base images are intentional.** Python 3.9 Dockerfiles pin `bookworm` (Debian trixie broke them, #1491); Jetson images pin `setuptools`, `pycuda`, and (Jetson 6/7) `transformers>=4.57.3,<5.9.0` (#2385, #2339). Base-image/tag changes need a stated reason.
- **CI jobs run on runners that have the target's system libs.** `inference-gpu` install tests need `cuda.h`/GPU libs → GPU machine (#2147); experimental e2e needs `libgles2`/`libegl1`/`libvips-dev` apt-installed (#2294).
- **Timeouts on long jobs.** Docker builds set `timeout-minutes` (120 in `docker.cpu.yml`) (#596, #635).
- **Tag/registry semantics** live in `.github/actions/determine-tags/action.yml`: `latest` is only added for a real release that is the newest and is not an RC (gate `&& [[ "$NORMALIZED_RELEASE" != *rc* ]]`, #1983); the normalize step strips a leading `v` and matches multi-digit patches via regex `^v?[0-9]+\.[0-9]+\.[0-9]+$` (#1470).

## Required companions
Block the PR if a change lacks its companion:
- **Release PR** ("Release X.Y.Z") → bump `inference/core/version.py` `__version__` AND `inference_models/pyproject.toml` `version` AND `inference_models/uv.lock` AND all 4 `inference-models~=` pins. RC→final drops the `rc` suffix in every place (#2341).
- **`inference-models` bump anywhere** → the other 3 requirements files + `pyproject.toml` + `uv.lock`. Never one-of-four (#2341).
- **`inference_models/pyproject.toml` dep add/bump** → regenerated `inference_models/uv.lock` in the same PR (#2301) + an `inference_models/docs/changelog.md` entry under the new `## \`X.Y.Z\`` header for user-facing changes.
- **New Docker image target** → its `.github/workflows/docker.<name>.yml` + Dockerfile + (if published) publish script + `permissions:`. Removing a target removes all of these (#2149).
- **New PyPI wheel / requirements group** → wire into `.release/pypi/*.setup.py` and/or `Makefile:create_wheels` and the relevant `test_package_install_*.yml`.
- **New runtime dep** → correct `requirements/*.txt` AND every Dockerfile target that uses it (system libs via `apt/dnf`, e.g. `libvips` #1209, `rustc` #975).
- **New GHA workflow** → top-level `permissions: contents: read`, `timeout-minutes`, and a `concurrency:` group for PR-triggered jobs.

## Key files & entry points
- `inference/core/version.py` — canonical server version (`__version__`).
- `inference_models/pyproject.toml`, `inference_models/uv.lock`, `inference_models/docs/changelog.md` — inference-models version + changelog.
- `requirements/requirements.{cpu,gpu,vino,jetson}.txt` — the 4 in-sync `inference-models` pin files; `requirements/_requirements.txt` — shared core pins (numpy lives here).
- `.github/actions/determine-tags/action.yml` — tag/normalization logic.
- `.github/workflows/docker.*.yml` (canonical: `docker.cpu.yml`), `publish.pypi.yml` — image build/push + wheel publishing; `claude-pr-review.yml` — review-agent workflow.
- `docker/dockerfiles/Dockerfile.onnx.jetson.{5.1.1,6.0.0,6.2.0,7.1.0}` — Jetson recipes (the fragile ones).
- `.release/pypi/inference*.setup.py`, `Makefile` (`create_wheels`, `create_inference_cli_whl`) — wheel assembly.
- `CODEOWNERS`, `.github/prompts/claude-pr-review.md` — ownership + review-agent config (codeowner review required, #455, #2228, #2135).

## Reference PRs
- [#2341](https://github.com/roboflow/inference/pull/2341) — version + 4-file pin sync (the release template).
- [#2326](https://github.com/roboflow/inference/pull/2326) — build once, push Docker Hub + GCP; canonical docker workflow shape.
- [#1242](https://github.com/roboflow/inference/pull/1242) / [#1484](https://github.com/roboflow/inference/pull/1484) — least-privilege `permissions:`.
- [#1983](https://github.com/roboflow/inference/pull/1983) — don't push `latest` for RC releases.
- [#1470](https://github.com/roboflow/inference/pull/1470) — version-tag regex / leading-`v` strip.
- [#522](https://github.com/roboflow/inference/pull/522) — GHA boolean input `== 'true'` coercion bug.
- [#839](https://github.com/roboflow/inference/pull/839) / [#1879](https://github.com/roboflow/inference/pull/1879) — defensive pins for transitive-dep breaks.
- [#2301](https://github.com/roboflow/inference/pull/2301) — Lambda build fix: certifi/Pygments pin + `uv.lock` sync.
- [#2385](https://github.com/roboflow/inference/pull/2385) — pin `transformers<5.9` in Jetson 6/7 Dockerfiles.
- [#2147](https://github.com/roboflow/inference/pull/2147) — run gpu-install test on runner with required system libs.
- [#2149](https://github.com/roboflow/inference/pull/2149) — CI/docker/release cleanup after inference-models switch (target removal).
- [#1595](https://github.com/roboflow/inference/pull/1595) — wrong `requirements/*.txt` COPY path broke the build.

## Related topic skills
Load the matching topic skill when the PR also shows these cross-cutting concerns:
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-test-hygiene`
