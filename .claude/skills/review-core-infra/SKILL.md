---
name: review-core-infra
description: PRs touching inference/core/ foundation — inference/core/env.py, inference/core/version.py, inference/core/entities/**, inference/core/utils/** (image_utils.py, requests.py, environment.py), inference/core/roboflow_api.py, exceptions.py, logger.py, nms.py. Diff signals: new os.getenv(...), __version__, build_roboflow_api_headers, api_key_safe_raise_for_status, load_image_from_url, WHITELISTED/BLACKLISTED_DESTINATIONS_FOR_URL_INPUT, *DC / _is_*_dc_to_dict.
---

# Reviewing core-infra changes

## Scope
The process-wide foundation under `inference/core/` that every server, CLI, and model path imports at load time:
- `inference/core/env.py` — env-var declarations / defaults
- `inference/core/version.py` — the single `__version__` const
- `inference/core/entities/**` — request/response Pydantic models + slotted dataclass (`*DC`) siblings
- `inference/core/utils/**` — especially `image_utils.py`, `requests.py`, `environment.py`, `hash.py`
- `inference/core/roboflow_api.py` — Roboflow platform HTTP client + workflow-spec cache
- `inference/core/exceptions.py`, `inference/core/logger.py`, `inference/core/nms.py`

OUT of scope (other skills own these): Workflows blocks/Execution Engine (`inference/core/workflows/**`), model implementations (`inference/models/**`), the `inference_models` package, HTTP route handlers in `inference/core/interfaces/http/**` (touched here only when an env flag rewires middleware, e.g. #1205/#2417).

## Review checklist
Severity tags: **BLOCK** = fix before merge; **FLAG** = raise it; **NIT** = optional.

1. **BLOCK — URL/image input (`image_utils.py`):** all SSRF layers intact and in order (input-enabled → backslash-in-netloc reject → prepare+re-parse → scheme → FQDN → whitelist → blacklist). Any relaxation adds a matching `ALLOW_*` opt-in flag defaulting to the safe value. See *SSRF hardening*.
2. **BLOCK — Default change:** does any `env.py` default flip change production behavior/backend? If so it must be justified, CI-pinned, and Docker-ENV-consistent (#2136). Otherwise block.
3. **BLOCK — Roboflow API calls:** every new/edited request uses `build_roboflow_api_headers(...)` (no bare `headers={}`) and `api_key_safe_raise_for_status(...)` (no raw `.raise_for_status(`).
4. **BLOCK — Cache/registry:** new cache reads degrade gracefully — wrapped, fall back to the API on outage, do not fail the request (#2387).
5. **BLOCK — Entities:** new fields are defaulted AND mirrored across the Pydantic model + `*DC` dataclass + `_is_*_dc_to_dict` mapper; serialized aliases (`class_name` → `"class"`) preserved (#2484).
6. **BLOCK — Ordering:** any "mutate then persist/cache" sequence mutates BEFORE the cache write (#966).
7. **FLAG — Version:** only when the diff itself touches `inference/core/version.py` — it must change exactly the one const (no imports/side effects), monotonically (except deliberate reverts, #2136). Never require a bump on a PR that doesn't touch it.
8. **FLAG — New env var:** typed coercion + sensible default, placed near related vars, AND documented (see *Env docs*).
9. **FLAG — Exceptions:** correct base class (`RoboflowAPIRequestError` for API failures; standalone `Exception` only for cross-cutting like `CacheUnavailableError`), docstring-with-Attributes style.
10. **FLAG — Logging:** log-event output stays JSON-serializable; no raw tracebacks / non-serializable objects into the event dict (#1225, #1340).
11. **FLAG — Tests:** matching `tests/inference/unit_tests/core/**` added/updated; security fixes assert the bypass is rejected AND that `requests.get` is NOT called for a rejected URL (#2500, #2501).
12. **NIT — User-facing strings:** package names / commands are correct (`inference`, not `roboflow-inference`, #154); model-id slugs use the platform's exact underscores-vs-hyphens (#1343).

### Not blocking
- No PR is required to bump `version.py` — inference releases are versioned separately; review the bump only when the diff includes one.
- `inference_models` has its own version+changelog — its `pyproject.toml` version is a separate concern.
- New env vars that are internal/experimental and off-by-default do not need a security-doc row (`environmental_variables.md` is enough).
- Don't demand a Docker-ENV row for a flag that is identical across all images.

## Standards
- **New env var = `os.getenv` + typed coercion + sensible default, added next to related vars.** Bools via `str2bool`, ints via `int(...)`, CSV via `set(x.split(","))`; complex coercion via `safe_env_to_type`. Defaults define production behavior on every deployment/Docker image, so a default change is a behavior change for all users. Examples: retry knobs `TRANSIENT_ROBOFLOW_API_ERRORS*` (#1004), `MD5_VERIFICATION_ENABLED` (#1492), `QWEN_2_5_ENABLED` (#1140), `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` (#957).
- **`version.py` holds exactly one truth: `__version__`.** It is executed as a script by every Docker build workflow (`python ./inference/core/version.py` → image tag). Do not add imports/side effects.
- **Model-enablement flags default `True` and gate the import site**, so a missing optional dependency degrades to a warning, not a crash — the flag wraps the `try: from inference.models import ...` (#1140 QWEN, #1389 Jetson VLM flags).
- **Backward compatibility for defaults is load-bearing.** Flipping a default that changes which backend/feature is active is a release-blocking decision — #2136 reverted the `USE_INFERENCE_MODELS` default flip *and* pinned the CI job with an explicit `USE_INFERENCE_MODELS=False`. Default-backend flips must be gated, CI-pinned, and reversible.
- **All Roboflow API HTTP calls go through `build_roboflow_api_headers(...)`** (injects `ROBOFLOW_API_EXTRA_HEADERS`, service secrets) — never a bare `headers={...}` (#932 retrofitted every call site; #2219 adds `X-Roboflow-Internal-Service-Secret` via the same builder).
- **Never call `response.raise_for_status()` directly** — use `api_key_safe_raise_for_status(response=...)` (or `api_key_safe_raise_for_status_aiohttp` for aiohttp) so API keys in URLs are redacted from exceptions/logs (#140 established the redaction). Grep diffs for raw `.raise_for_status(`.
- **Cache/registry access degrades gracefully.** Ephemeral (Redis/Dragonfly) cache failures must fall back to the API, not fail the request — wrap in `_try_*` helpers catching `CacheUnavailableError` (#2387 propagated an ephemeral-cache outage into a failed workflow-spec fetch).
- **New exceptions subclass the right base** (`RoboflowAPIRequestError` for API failures, e.g. `RoboflowAPITimeoutError` #1004; standalone `Exception` only for genuinely cross-cutting ones like `CacheUnavailableError` #2387). Keep the docstring-with-Attributes style used throughout `exceptions.py`.
- **Entity additions are additive & defaulted.** New fields carry a default so existing callers/serialization don't break, and are mirrored in BOTH the Pydantic model AND its `*DC` sibling AND the `_is_*_dc_to_dict` mapper — which is documented "Bit-equivalent to `...model_dump(by_alias=True, exclude_none=True)`". Field name vs. serialized alias (`class_name` → `"class"`) matters (#2484 added `mask_format: Literal["polygon"] = "polygon"` to model, dataclass, dict mapper, and tests).
- **Structured logging stays JSON-serializable** — no raw tracebacks or non-serializable objects into the log event dict; use the `structlog_exception_formatter` shape (#1225, #1340 emitted non-JSON output that broke log parsing).
- **Value mutation precedes caching.** Any value written to cache must be fully mutated first (#966 wrote the internal workflow `id` into the spec AFTER it was cached, so cached copies lacked the id).
- **NMS mask/indexing ops preserve the size-1 case.** Don't `squeeze` away the batch/element dimension for single-detection batches — `nms.py` guards this with `np.atleast_1d(...)` (#535 a bare `.squeeze()` collapsed a single-detection batch to 0-d and broke filtering).
- **SSRF hardening — keep ALL layers, in order.** `load_image_from_url` is defense-in-depth, gated on `ALLOW_URL_INPUT`. It does NOT do a DNS public-IP resolution check and does NOT set `allow_redirects=False` — do not claim it does. The actual layers, each raising on violation:
  1. `_ensure_url_input_allowed()` — enforces `ALLOW_URL_INPUT`.
  2. Reject if the raw parsed `netloc` contains a backslash `\` (#2500 allowlist bypass via `\@allowed.com` in userinfo).
  3. Build `requests.Request(...).prepare()` and re-parse the *prepared* URL (normalizes before any check runs).
  4. `_ensure_resource_schema_allowed(scheme)` — https only unless `ALLOW_NON_HTTPS_URL_INPUT`.
  5. Extract FQDN from `parsed_url.hostname` (not raw netloc) via `tldextract`; `_ensure_resource_fqdn_allowed` rejects bare-IP / no-FQDN unless `ALLOW_URL_INPUT_WITHOUT_FQDN`.
  6. `_ensure_location_matches_destination_whitelist` / `_ensure_location_matches_destination_blacklist` against `WHITELISTED_DESTINATIONS_FOR_URL_INPUT` / `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT` (None = no restriction).
  7. `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM` gates the local-fs loader path (#957).
  Removing, reordering, or short-circuiting any layer is a BLOCK.

## Required companions
- **Versioning:** no `version.py` bump is required — release chore PRs (#2505, #2396, #1838) handle that separately. When the diff does bump it, apply checklist item 7.
- **Env docs:** any new/renamed env var → row in `docs/server_configuration/environmental_variables.md`; input/security flags also in `docs/server_configuration/accepted_input_formats.md` (#957, #1004). Security-posture changes → `docs/install/security.md` (#2417).
- **Tests:** changes to `image_utils.py`, `roboflow_api.py`, `entities/**`, `nms.py`, `logger.py`, `environment.py` require unit tests under the matching `tests/inference/unit_tests/core/**` subdir. Security fixes assert rejection + that `requests.get` is NOT called for the rejected URL (#2500, #2501). Entity changes extend `tests/inference/unit_tests/core/entities/**` (#2484).
- **CI env parity:** if a default flips or a flag changes backend selection, the relevant CI workflow pins the flag so the matrix stays deterministic (#2136 pinned `USE_INFERENCE_MODELS=False`).
- **Docker image ENV parity:** capability/security flags that must differ per image are set in the relevant `docker/dockerfiles/Dockerfile.*` (#957 lambda, #1389 Jetson) — verify the flag is disabled where it should be (lambda/serverless lock-down).

## Key files & entry points
- `inference/core/env.py` — env declarations & defaults; SSRF flags `ALLOW_URL_INPUT`, `ALLOW_NON_HTTPS_URL_INPUT`, `ALLOW_URL_INPUT_WITHOUT_FQDN`, `WHITELISTED_DESTINATIONS_FOR_URL_INPUT`, `BLACKLISTED_DESTINATIONS_FOR_URL_INPUT`, `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`
- `inference/core/version.py` — `__version__` (build tag source)
- `inference/core/roboflow_api.py` — `build_roboflow_api_headers`, `get_workflow_specification`, `_try_retrieve_workflow_specification_from_ephemeral_cache` / `_try_cache_workflow_specification_in_ephemeral_cache`, `_get_from_url`
- `inference/core/utils/image_utils.py` — `load_image_from_url`, `_ensure_url_input_allowed`, `_ensure_resource_schema_allowed`, `_ensure_resource_fqdn_allowed`, `_ensure_location_matches_destination_whitelist` / `_blacklist`
- `inference/core/utils/requests.py` — `api_key_safe_raise_for_status`, `api_key_safe_raise_for_status_aiohttp`
- `inference/core/utils/environment.py` — `str2bool`, `safe_env_to_type`
- `requests/inference.py` — Pydantic + `*DC` siblings, `_is_pred_dc_to_dict` / `_is_response_dc_to_dict`
- `inference/core/exceptions.py`, `inference/core/logger.py` (`structlog_exception_formatter`), `inference/core/nms.py`
- Docs: `docs/server_configuration/environmental_variables.md`, `accepted_input_formats.md`, `docs/install/security.md`
- Tests: `tests/inference/unit_tests/core/**`

## Reference PRs
- [#2501](https://github.com/roboflow/inference/pull/2501) — SSRF via redirect-after-FQDN (security)
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
- [#1340](https://github.com/roboflow/inference/pull/1340) — log formatter emitted non-JSON, broke log parsing (bugfix)
- [#154](https://github.com/roboflow/inference/pull/154) — wrong package name in user-facing upgrade message (bugfix)

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-auth-and-tenant-security`
- `review-topic-input-boundary-security` (owns the SSRF / URL-input boundary rule)
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
