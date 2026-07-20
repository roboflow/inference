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
- **BLOCK** — EE compile/run behavior changed but `EXECUTION_ENGINE_V1_VERSION` not bumped (flagship miss; broke absent bump — correct in #2383). (§3)
- **BLOCK** — `inference-models` pin bumped in some `requirements/*.txt` but not all four, producing divergent installs. (§6)
- **FLAG** — Version bumped but changelog entry missing (EE changelog or `inference_models/docs/changelog.md`), or mirrored version-assertion tests left stale (#2115, #2267). (§3, §4, §5)
- **FLAG** — Correct additive block version but `get_execution_engine_compatibility` range not set, or wrong bump granularity (patch where minor/major is warranted). (§2, §3)
- **FLAG** — Deprecation by deletion: model/block/command ripped from code without the coordinated dockerfile/CI/docs/packaging sweep (#2334). (§7)
- **FLAG** — HTTP error-on-failure/status semantics changed without confirming it's deliberate + tested (#2528/#2529). (§8)

### Not blocking
Pure internal refactor / types / comments / formatting under a public surface with **no** observable contract or behavior change requires NO companion. Say so explicitly so the contributor isn't asked to over-version. Types-only/comment changes under `execution_engine/**` do NOT require an EE bump.

Never require an `inference/core/version.py` bump as a companion on any PR — inference releases are versioned separately (release chore PRs handle the bump). This does NOT relax the `inference_models/pyproject.toml` version+changelog rule (§5) or the EE version rule (§3).

## Standards
One canonical statement per rule. The checklist references these by number.

1. **Existing callers keep working.** A live workflow JSON, a pinned SDK call, an HTTP client, or an on-disk cache written by an older build must not break silently after upgrade. Removing/renaming a field on an existing surface, or changing a persisted format older code must read, is a break.

2. **Block schemas are additive, versioned side-by-side.** If a block's I/O, `kinds`, or `type` changes, it lands as a NEW `vN+1.py` sibling with a new `type` Literal (e.g. `roboflow_core/roboflow_instance_segmentation_model@v4`), not an in-place edit of an existing `@vN`. Each version sets `get_execution_engine_compatibility` returning a version range (e.g. `">=1.3.0,<2.0.0"`). Canonical set: `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/{v1,v2,v3,v4}.py`.

3. **EE behavior change → EE version + changelog.** A compile/run change bumps `EXECUTION_ENGINE_V1_VERSION` in `inference/core/workflows/execution_engine/v1/core.py` (currently `Version("1.12.0")`) AND adds a top entry to `docs/workflows/execution_engine_changelog.md` (heading `## Execution Engine \`vX.Y.Z\` | inference \`vA.B.C\``). Patch for fixes, minor for new capability, major for breaks. Contract codified in `.cursor/rules/execution-engine-version-changelog.mdc` — treat as the spec. `get_execution_engine_compatibility` gates against this constant.

4. **Mirrored version assertions updated.** An EE/inference bump updates the hardcoded version tests: `tests/inference/integration_tests/test_workflow_endpoints.py::test_get_versions_of_execution_engine` and `tests/inference/hosted_platform_tests/test_workflows.py::test_get_versions_of_execution_engine`. Grep `EXECUTION_ENGINE_V1_VERSION`, `execution_engine_version`.

5. **inference_models public API/entity change → version + changelog.** Bump `version` in `inference_models/pyproject.toml` (currently `0.29.7`) and add an entry to `inference_models/docs/changelog.md` under the matching `## \`X.Y.Z\`` heading (`Added`/`Fixed`/`Removed`).

6. **Release-bound pins stay in lock-step.** Any `inference-models~=X` change is identical across `requirements/requirements.{cpu,gpu,jetson,vino}.txt` (they carry a "keep in sync" comment), with `uv.lock` regenerated and changelog updated when `inference_models/pyproject.toml` changes. Do not demand an `inference/core/version.py` bump — that happens on release PRs only.

7. **Deprecation done right.** Removing a model/block/command raises the agreed error (`FeatureDeprecatedError` from `inference/core/exceptions.py` for deprecation-scoped surfaces), keeps the symbol resolvable, and sweeps dockerfiles/CI/docs/packaging together — not a bare code delete.

8. **HTTP behavior change is intentional and gated.** Changing error-on-failure / status semantics (e.g. serverless auth failure) is a client-visible contract; confirm it's deliberate, tested, and not an accidental regression.

## Key files & Reference PRs
- `inference/core/version.py` — single source of truth for the inference package version; release PRs touch it (never demand a bump on feature/bugfix PRs).
- `inference/core/workflows/execution_engine/v1/core.py` — `EXECUTION_ENGINE_V1_VERSION` constant that `get_execution_engine_compatibility` gates against.
- `docs/workflows/execution_engine_changelog.md` — canonical EE changelog format.
- `.cursor/rules/execution-engine-version-changelog.mdc` — the repo's written contract for when/how to bump EE + changelog + version tests.
- `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/{v1,v2,v3,v4}.py` — canonical additive block versioning; distinct `type` Literal + version-range compat per file.
- `inference_models/pyproject.toml` + `inference_models/docs/changelog.md` — the standalone package's version + changelog companion pair.
- `requirements/requirements.{cpu,gpu,jetson,vino}.txt` — the four `inference-models~=0.29.7` pins carrying the "keep in sync" comment.

Reference PRs that shipped a contract change WITH its companions: **#2383** (EE version + changelog + tests), **#2475** (EE perf change keeping parity), **#2384** (block schema shipped additively with defaults), **#2510 / #2341 / #2175 / #2102** (release: version + changelog + all four pins + lock), **#2334** (mediapipe gaze deprecation across dockerfiles/CI/docs/packaging), **#2395** (removing deprecated Gemini block versions), **#2528/#2529** (serverless auth-failure status change then revert), **#2115 / #2267** (stale mirrored version assertions fixed).
