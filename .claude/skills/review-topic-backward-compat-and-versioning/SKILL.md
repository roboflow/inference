---
name: review-topic-backward-compat-and-versioning
description: Load when a PR changes a user- or dev-visible contract (HTTP route/entity, SDK method, CLI command, workflow block schema/kinds, compiled-workflow format, inference_models public API, or a persisted/cache format) OR is release-bound (dependency/packaging/pin changes). Verifies the change ships its REQUIRED version/changelog/docs/pin companions and stays backward compatible.
---

# Review topic: Backward compatibility, public contracts & version companions

## When this applies
Trigger on the CONTENT of the diff, not one directory. Load this skill when a PR:
- Adds/removes/renames an HTTP route or changes a request/response entity under `inference/core/entities/**` or `inference/core/interfaces/http/http_api.py` (route shape, status codes, error-on-failure behavior).
- Changes any Workflow **block** manifest under `inference/core/workflows/core_steps/**` — its `type` Literal, input/output fields, `kinds`, defaults, or `get_execution_engine_compatibility()`.
- Changes Workflow **compile/run behavior** under `inference/core/workflows/execution_engine/**` (compiler, graph construction, input assembly, output construction, scheduling, error handling).
- Changes a **public** `inference_models` API (AutoModel, auto_loaders, weights_providers, model response/entity shapes, backend selection).
- Changes a **persisted or cache format** (model cache, weights layout, compiled-workflow serialization) that older code or on-disk artifacts must still read.
- Is **release-bound**: touches `requirements/*.txt`, `inference_models/pyproject.toml`, `uv.lock`, dockerfiles, or `.release/pypi/*` — i.e. deps/packaging/pins.
- Deprecates or removes a model/block/CLI command users depend on.

## What to protect
- **Existing callers keep working.** A live workflow JSON, a pinned SDK call, an HTTP client, or an on-disk cache written by an older build must not break silently after upgrade. Silent breakage = field of unreproducible production failures.
- **Version is the compatibility signal.** `inference/core/version.py`, `EXECUTION_ENGINE_V1_VERSION`, and `inference_models` pkg version are how the platform, hosted tests, and users negotiate compatibility. A behavior change without the matching bump makes the version a lie and breaks `get_execution_engine_compatibility()` gating.
- **Additive over mutating for schemas.** Block schemas are versioned side-by-side (`v1.py`/`v2.py`/`v3.py`), each a distinct `type` Literal. Editing an existing `@vN` block's contract in place silently changes every deployed workflow that references it.
- **Pins stay in lock-step.** `inference-models` is pinned across four requirements files that MUST match; a bump in one without the others produces divergent installs.

## What to check
1. **Is a contract actually changing?** Distinguish an internal refactor (no companion needed) from a user/dev-visible or release-bound change (companions REQUIRED). Types-only/comment/formatting changes under execution_engine do NOT require a bump.
2. **Block schema change → additive?** If a block's I/O, `kinds`, or `type` changed, confirm it landed as a NEW `vN+1.py` sibling with a new `type` Literal (e.g. `...@v3`) rather than mutating an existing `@vN`. See the instance-segmentation `v1/v2/v3` set. Removing/renaming a field on an existing version is a break.
3. **EE behavior change → EE version + changelog?** If compile/run behavior changed, `EXECUTION_ENGINE_V1_VERSION` in `inference/core/workflows/execution_engine/v1/core.py` MUST bump AND a new top entry added to `docs/workflows/execution_engine_changelog.md` (heading `## Execution Engine \`vX.Y.Z\` | inference \`vA.B.C\``). Patch for fixes, minor for new capability, major for breaks. (Contract codified in `.cursor/rules/execution-engine-version-changelog.mdc`.)
4. **Hardcoded version assertions updated?** An EE/inference bump must update mirrored version tests — `tests/inference/integration_tests/test_workflow_endpoints.py::test_get_versions_of_execution_engine` and `tests/inference/hosted_platform_tests/test_workflows.py`. Grep `EXECUTION_ENGINE_V1_VERSION`, `execution_engine/versions`, `["1.`.
5. **inference_models public API / entity change → version + changelog?** Bump `inference_models/pyproject.toml` `version` and add an entry to `inference_models/docs/changelog.md` (Added/Fixed/Removed).
6. **Release-bound → pins in lock-step?** Any `inference-models~=X` change must be identical across `requirements/requirements.cpu.txt`, `.gpu.txt`, `.jetson.txt`, `.vino.txt` (they carry a "keep in sync" comment) AND `inference/core/version.py` bumped for a release, changelog updated, `uv.lock` regenerated.
7. **Deprecation done right?** Removing a model/block/command should raise the agreed error (e.g. `FeatureDeprecatedError` for the deprecation-scoped surfaces), keep the symbol resolvable, and update dockerfiles/CI/docs/packaging together — not just delete code.
8. **HTTP behavior change is intentional and gated.** Changing error-on-failure / status semantics (e.g. serverless auth failure) is a client-visible contract; confirm it's deliberate, tested, and not an accidental regression.

## Common failure modes
- **Behavior change, no version bump** — EE compile/run or block runtime changed but `EXECUTION_ENGINE_V1_VERSION` / version untouched. This is the flagship HIGH-severity miss. Correct pattern in PR #2383 (EE → v1.10.1 with changelog + version tests) and PR #2475 (EE perf change keeping parity).
- **Mutating an existing block schema instead of adding a version** — should be a new `vN+1` sibling. PR #2384 enforced dense masks by touching `instance_segmentation` `v1/v2/v3` together AND bumping `inference/core/version.py`.
- **Pins drift across requirements files** — bumping `inference-models` in one file, forgetting the other three. Correct lock-step: PR #2510 (release inference_models 1.3.2 across all four requirements + pyproject + uv.lock), PR #2341, PR #2175, PR #2102.
- **inference_models change without changelog entry** — public shape/behavior changed but `docs/changelog.md` not updated (see the KeyPoints `covariance` entry style in the changelog top).
- **Deprecation by deletion** — ripping a model out of code without the coordinated dockerfile/CI/docs/packaging sweep. Correct: PR #2334 (mediapipe gaze deprecation across dockerfiles, CI, docs, packaging), PR #2395 (removing deprecated Gemini versions in blocks).
- **Silent HTTP contract regression** — PR #2528/#2529 (serverless auth-failure behavior fix then revert) shows how a status/behavior change ripples to clients and needs deliberate handling + tests.
- **Version bump without mirrored test update** — hosted/integration version assertions left stale (PR #2115 to 1.8.0, PR #2267 to 1.9.0 fix exactly this class).

## Example implementations (point here)
- `inference/core/version.py` — single source of truth for the inference package version; every release-bound PR touches it.
- `inference/core/workflows/execution_engine/v1/core.py` (`EXECUTION_ENGINE_V1_VERSION = Version("1.12.0")`) — the EE compatibility constant that `get_execution_engine_compatibility()` gates against.
- `docs/workflows/execution_engine_changelog.md` — canonical EE changelog format (`## Execution Engine vX.Y.Z | inference vA.B.C` + "What changed" bullets).
- `.cursor/rules/execution-engine-version-changelog.mdc` — the repo's own written contract for when/how to bump EE + changelog + version tests; treat as the spec.
- `inference/core/workflows/core_steps/models/roboflow/instance_segmentation/{v1,v2,v3}.py` — canonical additive block versioning: each file a distinct `type: Literal["roboflow_core/roboflow_instance_segmentation_model@vN"]`, plus per-block `get_execution_engine_compatibility()` returning a version range (e.g. `">=1.3.0,<2.0.0"`).
- `inference_models/pyproject.toml` (`version`) + `inference_models/docs/changelog.md` — the standalone package's version + changelog companion pair.
- `requirements/requirements.{cpu,gpu,jetson,vino}.txt` — the four `inference-models~=0.29.7` pins carrying the "keep in sync" comment; the lock-step invariant lives here.
- Reference PRs that shipped a contract change WITH its companions: **#2383** (EE version + changelog + tests), **#2384** (block schema + version), **#2510 / #2341 / #2175 / #2102** (release: version + changelog + all four pins + lock), **#2334** (deprecation sweep across packaging/CI/docs).

## Severity guidance
- **Critical** — a schema/format/API change that breaks existing deployed workflows, persisted caches, or live HTTP/SDK clients with no migration path or version gate (e.g. mutating an existing `@vN` block's I/O, changing a persisted format older builds must read).
- **High** — a clearly release-bound or user/dev-visible behavior change with its REQUIRED version companion omitted: EE compile/run change without `EXECUTION_ENGINE_V1_VERSION` bump; `inference-models` bumped in some requirements files but not all; public entity change without changelog. Also: deprecation-by-deletion with no coordinated packaging/docs sweep.
- **Medium** — companion present but incomplete: version bumped but changelog entry missing or stale mirrored version tests; correct additive block version but `get_execution_engine_compatibility()` range not set; wrong bump granularity (patch where a minor/major is warranted).
- **Not blocking** — pure internal refactor / types / comments / formatting under a public surface with no observable contract or behavior change; no companion required (say so explicitly so the contributor isn't asked to over-version).
