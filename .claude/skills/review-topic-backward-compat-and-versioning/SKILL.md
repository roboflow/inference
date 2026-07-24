---
name: review-topic-backward-compat-and-versioning
description: Load when a PR changes a user/dev-visible contract or is release-bound. Triggers on diffs to `inference/core/version.py`, `EXECUTION_ENGINE_V1_VERSION`, `inference/core/interfaces/http/http_api.py` or `inference/core/entities/**`, a Workflow block manifest under `inference/core/workflows/core_steps/**` (`type` Literal, `kinds`, `get_execution_engine_compatibility`), a public `inference_models` API, `inference_models/pyproject.toml`, a persisted/cache format, or `requirements/*.txt` / `uv.lock`.
---

# Review topic: Backward compatibility, public contracts & version companions

## When this applies
Trigger on the CONTENT of the diff, not one directory. Load when a PR:
- Adds/removes/renames an HTTP route or changes a request/response entity under `inference/core/entities/**` or `inference/core/interfaces/http/http_api.py`.
- Changes a Workflow **block** manifest under `inference/core/workflows/core_steps/**` — its `type` Literal, I/O fields, `kinds`, defaults, or `get_execution_engine_compatibility`.
- Changes Workflow **compile/run behavior** under `inference/core/workflows/execution_engine/**`.
- Changes a **public** `inference_models` API (AutoModel, auto_loaders, weights_providers, response/entity shapes, backend selection).
- Changes a **persisted or cache format** (model cache, weights layout, compiled-workflow serialization) older code or on-disk artifacts must still read.
- Is **release-bound**: touches `requirements/*.txt`, `inference_models/pyproject.toml`, `uv.lock`, dockerfiles, or `.release/pypi/*`.
- Deprecates or removes a model/block/CLI command users depend on.

## Review checklist
Severity tags: **BLOCK** = fix before merge; **FLAG** = raise it; **NIT** = optional.

- **BLOCK** — Existing-caller break with no version gate: an existing `@vN` block's I/O/`kinds`/`type` mutated in place, a persisted/cache format changed so older builds can't read it, or a live HTTP/SDK contract broken. (see Standards §1, §2)
- **BLOCK** — EE compile/run behavior changed but no user-facing entry was added under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`. (§3)
- **BLOCK** — `inference-models` pin bumped in some `requirements/*.txt` but not all four, producing divergent installs. (§6)
- **BLOCK** — A functional `inference_models` change has no entry under `## Unreleased` in `inference_models/docs/changelog.md`. (§5)
- **FLAG** — Correct additive block version but `get_execution_engine_compatibility` range not set, or the floor left at the copy-pasted default while the block relies on a capability introduced later (verify against `docs/workflows/execution_engine_changelog.md`). Maintainers, not contributors, choose release bump granularity. (§2, §3)
- **FLAG** — Deprecation by deletion: model/block/command ripped from code without the coordinated dockerfile/CI/docs/packaging sweep (#2334). (§7)
- **FLAG** — HTTP error-on-failure/status semantics changed without confirming it's deliberate + tested (#2528/#2529). (§8)

### Not blocking
Pure internal refactor / types / comments / formatting under a public surface with **no** observable contract or behavior change requires NO companion. Say so explicitly so the contributor isn't asked to over-version. Types-only/comment changes under `execution_engine/**` do NOT require an EE bump.

Never require an `inference/core/version.py`, `inference_models/pyproject.toml`, or `EXECUTION_ENGINE_V1_VERSION` bump from a contributor. Maintainers handle version changes in release PRs. This does not relax the contributor-owned `Unreleased` changelog rules (§3, §5), and it has one EE exception: a Workflow block gating on unreleased EE capabilities forces the version to be placed and bumped early (§3).

## Standards
One canonical statement per rule. The checklist references these by number.

1. **Existing callers keep working.** A live workflow JSON, a pinned SDK call, an HTTP client, or an on-disk cache written by an older build must not break silently after upgrade. Removing/renaming a field on an existing surface, or changing a persisted format older code must read, is a break.

2. **Block schemas are additive, versioned side-by-side.** If a block's I/O, `kinds`, or `type` changes, it lands as a NEW `vN+1.py` sibling with a new `type` Literal (e.g. `roboflow_core/roboflow_instance_segmentation_model@v4`), not an in-place edit of an existing `@vN`. Each version sets `get_execution_engine_compatibility` returning a version range (e.g. `">=1.3.0,<2.0.0"`) whose floor is derived from the EE changelog — the version that introduced the newest capability the block uses — not copy-pasted. A block relying on a capability still under `## Unreleased` has no released floor to declare; that forces the EE version to be placed and bumped early (§3's exception) before the block ships. Canonical set: `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/{v1,v2,v3,v4}.py`.

3. **EE behavior change → Unreleased changelog entry.** A compile/run change adds a user-facing entry under `## Unreleased` in `docs/workflows/execution_engine_changelog.md`. The contributor does not change `EXECUTION_ENGINE_V1_VERSION`. At release time maintainers choose the bump, update the constant and mirrored assertions, move the entries under `## Execution Engine \`vX.Y.Z\` | inference \`vA.B.C\``, and leave a fresh `## Unreleased` section. Contract codified in `.cursor/rules/execution-engine-version-changelog.mdc` — treat as the spec. `get_execution_engine_compatibility` gates against the released constant. Exception: when a Workflow block must gate on a capability still under `## Unreleased`, the version is placed and bumped early (final heading + constant + mirrored assertions, maintainer-coordinated) so the block can declare its floor.

4. **Release-time mirrored assertions.** When maintainers bump the EE/inference version, they update the hardcoded version tests: `tests/inference/integration_tests/test_workflow_endpoints.py::test_get_versions_of_execution_engine` and `tests/inference/hosted_platform_tests/test_workflows.py::test_get_versions_of_execution_engine`. Grep `EXECUTION_ENGINE_V1_VERSION`, `execution_engine_version`.

5. **inference_models public API/entity change → Unreleased changelog entry.** Add an entry under `## Unreleased` in `inference_models/docs/changelog.md` (`Added`/`Changed`/`Fixed`/`Removed`). The contributor does not change `inference_models/pyproject.toml`; maintainers choose and apply that package version during release.

6. **Release-bound pins stay in lock-step.** In a maintainer release PR, any `inference-models~=X` change is identical across `requirements/requirements.{cpu,gpu,jetson,vino}.txt` (they carry a "keep in sync" comment), with `uv.lock` regenerated and the `## Unreleased` changelog section amended to its final version heading. Do not demand these changes from a feature/fix contributor.

7. **Deprecation done right.** Removing a model/block/command raises the agreed error (`FeatureDeprecatedError` from `inference/core/exceptions.py` for deprecation-scoped surfaces), keeps the symbol resolvable, and sweeps dockerfiles/CI/docs/packaging together — not a bare code delete.

8. **HTTP behavior change is intentional and gated.** Changing error-on-failure / status semantics (e.g. serverless auth failure) is a client-visible contract; confirm it's deliberate, tested, and not an accidental regression.

## Release notices

When the PR changes inference-models functionality, include the non-blocking
maintainer notice **inference-models requires a version bump for release**. When
it changes Execution Engine compile/run behavior, include **Execution Engine
requires a version bump for release**. Include both notices for cross-system
changes. Missing `Unreleased` entries are contributor-actionable findings;
version bumps are not.

## Key files & Reference PRs
- `inference/core/version.py` — single source of truth for the inference package version; release PRs touch it (never demand a bump on feature/bugfix PRs).
- `inference/core/workflows/execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION` constant that `get_execution_engine_compatibility` gates against.
- `docs/workflows/execution_engine_changelog.md` — canonical EE changelog format.
- `.cursor/rules/execution-engine-version-changelog.mdc` — the repo's written contract for when/how to bump EE + changelog + version tests.
- `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/{v1,v2,v3,v4}.py` — canonical additive block versioning; distinct `type` Literal + version-range compat per file.
- `inference_models/pyproject.toml` + `inference_models/docs/changelog.md` — the standalone package's version + changelog companion pair.
- `requirements/requirements.{cpu,gpu,jetson,vino}.txt` — the four `inference-models~=0.29.7` pins carrying the "keep in sync" comment.

Reference PRs that shipped a contract change WITH its companions: **#2383** (EE version + changelog + tests), **#2475** (EE perf change keeping parity), **#2384** (block schema shipped additively with defaults), **#2510 / #2341 / #2175 / #2102** (release: version + changelog + all four pins + lock), **#2334** (mediapipe gaze deprecation across dockerfiles/CI/docs/packaging), **#2395** (removing deprecated Gemini block versions), **#2528/#2529** (serverless auth-failure status change then revert), **#2115 / #2267** (stale mirrored version assertions fixed).
