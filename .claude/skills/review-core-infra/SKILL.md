---
name: review-core-infra
description: Review guidance for PRs changing inference/core/ foundation files (env.py, version.py, entities/, utils/, roboflow_api.py, exceptions.py, constants.py, logger.py, nms.py) — enforces core-infra standards, backward-compat contracts, security-hardening on URL/image input, and required companions (version bumps, env docs, tests).
---

# Reviewing core-infra changes

## Scope
Trigger this skill when a PR touches the process-wide foundation under `inference/core/`:
- `inference/core/env.py` — env-var declarations / defaults
- `inference/core/version.py` — the single `__version__` const
- `inference/core/entities/**` — request/response Pydantic models + slotted dataclass (`*DC`) siblings
- `inference/core/utils/**` — especially `image_utils.py`, `requests.py`, `environment.py`, `url_utils.py`, `hash.py`
- `inference/core/roboflow_api.py` — Roboflow platform HTTP client + workflow-spec cache
- `inference/core/exceptions.py`, `inference/core/constants.py`, `inference/core/logger.py`, `inference/core/nms.py`

OUT of scope (other skills own these): Workflows blocks/Execution Engine (`inference/core/workflows/**`), model implementations (`inference/models/**`), the `inference_models`/`inference-exp` package, HTTP route handlers in `inference/core/interfaces/http/**` (touched here only when an env flag rewires middleware, e.g. #1205/#2417).

## What this surface is
The shared substrate every server, CLI, and model path imports at load time. The contracts a reviewer must protect:
- **`env.py` is import-time, side-effecting config.** Every setting is `os.getenv(...)` wrapped in a typed coercion (`str2bool`, `int(...)`, `set(... .split(","))`). Defaults define production behavior on every deployment and Docker image. Changing a default is a behavior change for all users (see revert #2136).
- **`version.py` holds exactly one truth:** `__version__`. It is executed as a script by every Docker build workflow (`python ./inference/core/version.py` → image tag; `.github/workflows/docker.*.yml`). Do not add imports/side effects to it.
- **Entities are wire/serialization contracts.** Pydantic models (`entities/requests`, `entities/responses`) and their slotted `*DC` dataclass fast-path siblings must stay bit-equivalent in dict form — `_is_pred_dc_to_dict` is documented as "Bit-equivalent to `...model_dump(by_alias=True, exclude_none=True)`" (#2484). Field name vs. serialized alias (`class_name` → `"class"`) matters.
- **`roboflow_api.py` is the platform boundary.** Every outbound call must go through the shared header builder and the API-key-safe status check; ret/cache behavior is env-gated and must degrade gracefully.
- **`image_utils.load_image_from_url` is the SSRF attack surface.** It is defense-in-depth: scheme allow-list → FQDN/allowlist/blacklist → DNS-resolves-to-public → no-redirect. Reviewers treat any change here as security-sensitive.

## Standards enforced here
- **New env var = `os.getenv` + typed coercion + sensible default, added next to related vars.** Bools via `str2bool`, ints via `int(...)`, CSV via `set(x.split(","))`. Evidence: `ALLOW_URL_INPUT_TO_PRIVATE_NETWORKS` (#2501), retry knobs `TRANSIENT_ROBOFLOW_API_ERRORS*` (#1004), `MD5_VERIFICATION_ENABLED` (#1492), `QWEN_2_5_ENABLED` (#1140), `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` (#957).
- **New env var MUST be documented** in `docs/server_configuration/environmental_variables.md` (table) or `accepted_input_formats.md` (input/security flags). Evidence: #1004 and #957 add the row in the same PR; #957 documents `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`.
- **Model-enablement flags default `True` and gate the import site**, so a missing optional dependency degrades to a warning, not a crash — flag wraps the `try: from inference.models import ...` (#1140 QWEN, #1389 Jetson VLM flags).
- **Backward compatibility for defaults is load-bearing.** Flipping a default that changes which backend/feature is active is a release-blocking decision — #2136 reverted `USE_INFERENCE_MODELS` default `True`→`False` *and* pinned the CI job with an explicit `USE_INFERENCE_MODELS=False`. A PR that silently flips a default must justify it or is blocked.
- **All Roboflow API HTTP calls go through `build_roboflow_api_headers(...)`** (injects `ROBOFLOW_API_EXTRA_HEADERS`, service secrets) — never a bare `headers={...}` (#932 retrofitted every call site; #2219 adds `X-Roboflow-Internal-Service-Secret` via the same builder).
- **Never call `response.raise_for_status()` directly** — use `api_key_safe_raise_for_status(response=...)` (or the aiohttp variant) so API keys in URLs are redacted from exceptions/logs (#140 retrofitted every call; #1205/#140 are the redaction lineage). Reviewers grep for raw `.raise_for_status(` in diffs.
- **Cache/registry access degrades gracefully.** Ephemeral (Redis/Dragonfly) cache failures must fall back to the API, not fail the request — wrap in `_try_*` helpers catching `CacheUnavailableError`/`RedisConnectionError`/`RedisTimeoutError` (#2387).
- **New exceptions subclass the right base** (`RoboflowAPIRequestError` for API failures, e.g. `RoboflowAPITimeoutError` #1004; standalone `Exception` only for genuinely cross-cutting ones like `CacheUnavailableError` #2387). Keep the docstring-with-Attributes style used throughout `exceptions.py`.
- **Entity additions are additive & defaulted.** New fields carry a default so existing callers/serialization don't break, and are mirrored in BOTH the Pydantic model and its `*DC` sibling + the `_is_*_dc_to_dict` mapper (#2484 `mask_format: Literal["polygon"] = "polygon"`, added to dataclass AND dict mapper AND tests).
- **Structured logging must stay JSON-serializable** — no raw tracebacks or non-serializable objects into the log event dict; use the custom `structlog_exception_formatter` shape (#1225, #1340).
- **SSRF hardening keeps ALL layers.** URL input goes through scheme check → FQDN/allowlist/blacklist → `_ensure_url_resolves_to_allowed_address` (public-IP DNS check) → `allow_redirects=False` with explicit redirect rejection (#2500 backslash/userinfo bypass, #2501 DNS-rebind + redirect-to-internal, #957 local-fs gate). Removing or reordering a check is a block.

## Required companions
- **Version bump:** a release/feature/fix that ships to users bumps `inference/core/version.py` `__version__` (chore PRs #2505, #2396, #1838 do only this; feature/bugfix PRs #793, #966, #1140, #1340, #1389 bump it alongside the change). A pure-refactor/docs PR need not. Note: `inference_models`/`inference-exp` has its own version+changelog — NOT bumped here.
- **Env docs:** any new/renamed env var → row in `docs/server_configuration/environmental_variables.md`; input/security flags also in `docs/server_configuration/accepted_input_formats.md` (#957, #1004). Security posture changes → `docs/install/security.md` (#2417).
- **Tests:** changes to `image_utils.py`, `roboflow_api.py`, `entities/**`, `nms.py`, `logger.py`, `environment.py` require unit tests under `tests/inference/unit_tests/core/**` (matching subdir). Security fixes MUST add a regression test asserting the bypass is rejected AND that `requests.get` is NOT called for a rejected URL (#2500, #2501). Entity changes add/extend `tests/inference/unit_tests/core/entities/**` (#2484).
- **CI env parity:** if a default flips or a flag changes backend selection, the relevant CI workflow must pin the flag so the test matrix stays deterministic (#2136 pinned `USE_INFERENCE_MODELS=False` in `integration_tests_inference_models.yml`).
- **Docker image ENV parity:** new capability/security flags that must differ per image are set in the relevant `docker/dockerfiles/Dockerfile.*` (#957 lambda, #1389 Jetson). Check the flag is disabled where it should be (lambda/serverless lock-down).

## Common pitfalls & past regressions
- **#2501 — FQDN validated on the first URL, then redirect followed to an internal target (SSRF via redirect + DNS rebind).** Check: any URL fetch must resolve to a public address AND set `allow_redirects=False` with explicit redirect rejection.
- **#2500 — allowlist bypass via `https://host\@allowed.com` (backslash in userinfo / netloc).** Check: URL is re-parsed from a prepared request and the netloc is rejected if it contains `\`; hostname (not raw netloc) feeds the FQDN extractor.
- **#2387 — ephemeral cache (Redis/Dragonfly) outage propagated and failed workflow-spec fetch.** Check: cache reads/writes are wrapped and fall back to the API on connection/timeout errors.
- **#966 — internal workflow `id` was written into the spec AFTER it was cached, so cached copies lacked the id.** Check: mutations to a value must happen BEFORE it is cached, not after.
- **#1343 — model-id typo (`perception-encoder` vs `perception_encoder`) shipped a wrong default in `env.py`.** Check: model-id string defaults exactly match the platform's expected slug (underscores vs hyphens).
- **#535 — `.squeeze()` on the NMS confidence mask collapsed a single-detection batch to a 0-d array and broke filtering.** Check: numpy mask/indexing ops don't `squeeze` away the batch/element dimension for the size-1 case.
- **#1205 — env var typo `CORRELACTION_ID_HEADER` and logging gated on the wrong flag** (`DEDICATED_DEPLOYMENT_ID or GCP_SERVERLESS` instead of a single `API_LOGGING_ENABLED`). Check: env-var names spelled consistently between `env.py` and every importer; one flag, one behavior.
- **#1340 — log formatter emitted non-JSON / bracketed prefix, breaking JSON log parsing.** Check: structured-log output stays pure JSON, no stray formatting.
- **#154 — user-facing upgrade message told users to `pip install roboflow-inference` (wrong package name, `inference`).** Check: user-facing strings name the correct package/command.
- **#2136 (revert) — making `inference_models` the default backend via env flag regressed prod; reverted with a version DOWN-bump.** Check: default-backend flips are gated, CI-pinned, and reversible.

## Review checklist
1. **Version:** if user-facing behavior changed or it's a release, is `inference/core/version.py` bumped exactly one const, no new imports/side effects? Is the bump monotonic (except deliberate reverts, #2136)?
2. **New env var:** typed coercion + default present in `env.py`, placed near related vars, AND documented in `environmental_variables.md` / `accepted_input_formats.md`?
3. **Default change:** does any default flip change production behavior/backend? If so, is it justified, CI-pinned, and Docker-ENV-consistent? Otherwise block.
4. **Roboflow API calls:** every new/edited request uses `build_roboflow_api_headers(...)` and `api_key_safe_raise_for_status(...)` — no bare `headers={}` and no raw `.raise_for_status()`.
5. **Cache/registry:** new cache/registry reads degrade gracefully (fall back, don't fail) on backend outage.
6. **URL/image input (`image_utils`):** all SSRF layers intact (scheme → FQDN/allow/blacklist → public-IP DNS → no-redirect); local-fs gated by `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`. Any relaxation added a matching `ALLOW_*` opt-in flag defaulting to the safe value.
7. **Entities:** new fields are defaulted and mirrored across Pydantic model + `*DC` dataclass + `_is_*_dc_to_dict` mapper; serialized aliases preserved.
8. **Exceptions:** correct base class, docstring style consistent.
9. **Logging:** output stays JSON-serializable; no raw tracebacks into event dict.
10. **Tests:** matching `tests/inference/unit_tests/core/**` added/updated; security fixes assert rejection + `requests.get` not called; mutation-before-cache ordering covered.
11. **Ordering bugs:** any "mutate then persist/cache" sequence — confirm mutation precedes the cache write (#966).

## Key files & entry points
- `inference/core/env.py` — env declarations & defaults (grep here first for any flag)
- `inference/core/version.py` — the version const (build tag source)
- `inference/core/roboflow_api.py` — `build_roboflow_api_headers`, `get_workflow_specification`, `_try_*` cache helpers, `_get_from_url`
- `inference/core/utils/image_utils.py` — `load_image_from_url`, `_ensure_url_resolves_to_allowed_address`, `load_image_with_known_type`
- `inference/core/utils/requests.py` — `api_key_safe_raise_for_status[_aiohttp]`
- `inference/core/utils/environment.py` — `str2bool`, `safe_env_to_type`
- `inference/core/entities/responses/inference.py` & `requests/inference.py` — Pydantic + `*DC` siblings, `_is_*_dc_to_dict`
- `inference/core/exceptions.py`, `inference/core/logger.py`, `inference/core/nms.py`
- Docs: `docs/server_configuration/environmental_variables.md`, `accepted_input_formats.md`, `docs/install/security.md`
- Tests: `tests/inference/unit_tests/core/**` (env, roboflow_api, entities, utils/image_utils)

## Reference PRs
- [#2501](https://github.com/roboflow/inference/pull/2501) — SSRF via redirect-after-FQDN + DNS rebind; DNS public-IP check + `allow_redirects=False` (security)
- [#2500](https://github.com/roboflow/inference/pull/2500) — allowlist bypass via backslash in URL userinfo (security)
- [#2417](https://github.com/roboflow/inference/pull/2417) — local-deployment security env flags + `docs/install/security.md` (security/docs)
- [#957](https://github.com/roboflow/inference/pull/957) — `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` gate + docker/docs (security)
- [#2387](https://github.com/roboflow/inference/pull/2387) — graceful fallback on ephemeral cache failure; `CacheUnavailableError` + `_try_*` (bugfix)
- [#2484](https://github.com/roboflow/inference/pull/2484) — additive entity field mirrored across model/`*DC`/dict-mapper/tests (feature)
- [#2219](https://github.com/roboflow/inference/pull/2219) — internal-service-secret header via `build_roboflow_api_headers` (feature)
- [#1004](https://github.com/roboflow/inference/pull/1004) — RF API retry knobs (env + exceptions + docs) (feature)
- [#932](https://github.com/roboflow/inference/pull/932) — env-injectable headers; centralized `build_roboflow_api_headers` (feature)
- [#966](https://github.com/roboflow/inference/pull/966) — mutate-before-cache ordering fix for workflow id (bugfix)
- [#535](https://github.com/roboflow/inference/pull/535) — NMS `.squeeze()` broke single-detection batches (bugfix)
- [#2136](https://github.com/roboflow/inference/pull/2136) — revert default-backend flip; version down-bump + CI pin (revert)
- [#1205](https://github.com/roboflow/inference/pull/1205) — correlation-id env typo + single `API_LOGGING_ENABLED` flag (feature/bugfix)

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-auth-and-tenant-security`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
