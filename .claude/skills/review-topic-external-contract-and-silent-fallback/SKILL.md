---
name: review-topic-external-contract-and-silent-fallback
description: Load when a PR calls an external/platform API (Roboflow API, model/weights registry, license server), changes an inference_sdk<->server response shape, or adds fallback / auto-conversion / backend-downgrade / default-on-missing behavior. Purpose — catch silent fallbacks that hide an unavailable or changed model/backend, and cross-service contract drift where one side requests fields/shapes the other side has not deployed.
---

# Review topic: External/platform contract drift & silent fallback

## When this applies
Load this skill when the diff shows ANY of these content signals (paths are hints, not the trigger):
- Calls into `inference/core/roboflow_api.py`, `inference/core/registries/roboflow.py`, the weights/model registry, or the license server proxy — new endpoint, new query param, new request header, or a changed URL suffix.
- Requests a NEW field from the platform, or reads a NEW nested key of an API/JSON response (e.g. `response["version"]["modelType"]`, `api_data.get("taskType")`, `modelVariant`), OR changes which nested shape is read.
- Changes an `inference_sdk` <-> server contract: response `model_dump`/serialization, `response_model`, field presence/aliases, or default values that a client parses.
- Adds/loosens FALLBACK, auto-conversion, `.get(key, <default>)`, backend/model auto-negotiation, backend downgrade, or "retry the other path" logic when a requested model/backend/platform capability is unavailable.
- Adds an env flag or secret that changes which service/endpoint/backend is contacted (`USE_INFERENCE_MODELS`, `DISABLED_INFERENCE_MODELS_BACKENDS`, `ROBOFLOW_INTERNAL_SERVICE_SECRET`, registry-redirect flags).
- Assumes a companion PR (platform/backend/SDK) is already deployed.

## What to protect
- **Correctness-on-exact-contract:** when the caller asked for a specific model / backend / platform capability, the system either delivers THAT or fails loudly. A silent downgrade (torch->onnx, requested model -> default model, unavailable task type -> `"object-detection"`) returns plausible-but-wrong results the user cannot detect.
- **Cross-service contract coherence:** a field/shape/header the client sends or reads must exist on the deployed server, and vice versa. Drift produces `KeyError`/`None` in prod, 4xx on unrecognized params, or auth bypass on a header the server ignores.
- **Fail-clear vs fail-silent:** unreachable dependencies (cache, registry, license server) must raise a typed, actionable error at a decision point — not be swallowed into a wrong-but-successful response.

## What to check
1. **New response field/shape** — is it guarded? If the code reads `data["x"]["y"]` or `.get("y", default)`, confirm the platform already returns `y` for the relevant model ages/tenants. Old projects often lack `modelType`/`type`; see the layered defaulting in `get_model_type` (`inference/core/registries/roboflow.py:279-316`). A new required field with no deploy guarantee is drift.
2. **Silent default on missing** — every `.get(key, <fallback>)` on external data: is the fallback SAFE, or does it mask a real unavailability? `api_data.get("taskType", "object-detection")` silently mislabels non-detection models if the field is genuinely missing. Prefer raising (`MissingDefaultModelError`, `ModelArtefactError`) over guessing when correctness depends on it.
3. **Backend / model downgrade** — auto-negotiation or backend selection must not silently substitute a different backend/quantization/model when the requested one is unavailable. When the caller pinned `requested_model_package_id`, resolution must return exactly that or raise (`NoModelPackagesAvailableError` / `AmbiguousModelPackageResolutionError`), never the "closest" package.
4. **Deployment capability guard** — after resolving a model/task type, is it verified against what THIS deployment can serve? Follow `_ensure_model_supported_on_this_deployment` (`inference/core/registries/roboflow.py:332`); a resolved-but-unserveable model must fail clearly, not attempt a wrong backend.
5. **SDK<->server both sides** — a serialization/`response_model`/field-alias change: does the client parser tolerate it, and does the server still emit fields the client requires (e.g. `parent_id` presence, PR #1599)? Check both directions in the same PR or confirm ordered rollout.
6. **New param / header actually honored** — a new query param or auth header (e.g. `service_secret`, `X-Roboflow-Internal-Service-Secret`, `countinference`) must be recognized by the deployed endpoint; an ignored security header is worse than an error. Confirm the server side exists (PR #2219).
7. **Error mapping preserved** — external errors should route through `wrap_roboflow_api_errors` -> typed exceptions (401->`RoboflowAPINotAuthorizedError`, 402->`PaymentRequiredError`, timeout->`RoboflowAPITimeoutError`). A new call path that catches broadly and returns `None`/defaults defeats this.
8. **Swallowed dependency failure** — a bare `except Exception` around a registry/cache/license call that returns a default is a red flag. Contrast with the CORRECT pattern in PR #2387: cache-unreachable raises a typed `CacheUnavailableError` at a decision point, and only THEN does the caller consciously fall through to the authoritative API.
9. **Companion-PR assumption** — if the change only works once a platform/backend PR ships, require a flag default-off, an ordered-rollout note, or graceful behavior pre-deploy.

## Common failure modes
- **Silent model/task-type substitution** — defaulting `taskType`/`modelType` to a guess instead of failing; masks that the registry did not return the requested model (`registries/roboflow.py` defaulting block; guard added around missing attrs in PR #2105).
- **Backend auto-downgrade hiding unavailability** — ranking/negotiation returns a different backend than requested with no signal; error-clarity and explicit discard-reason plumbing added in PR #1434, ranking changes PR #2047/#1811. Global backend disable must validate names and fail on typos (PR #2096).
- **Reading a not-yet-deployed nested field** — parsing a new response shape (`modelVariant` from RFAPI, PR #1641) before the platform emits it -> `None`/`KeyError` in prod.
- **SDK<->server shape drift** — server dropping a field a client relies on; fixed by preserving `parent_id` in OCR responses (PR #1599).
- **Ignored security/auth header** — sending `X-Roboflow-Internal-Service-Secret` to an endpoint that doesn't yet honor it (contract added deliberately, PR #2219) — assuming it's enforced when it isn't.
- **Swallowed cache/dependency error masking a stale/wrong answer** — the anti-pattern PR #2387 fixes by raising `CacheUnavailableError` and falling back to the API explicitly instead of silently.
- **Deprecated external model default flipping silently** — changing the default model version (Gemini 2.0->2.5, PR #2395) without keeping aliases for removed ids strands pinned callers.

## Example implementations (point here)
- `inference/core/roboflow_api.py` — `wrap_roboflow_api_errors` + `DEFAULT_ERROR_HANDLERS` (status->typed exception): the canonical "fail clear with an actionable message" boundary for every platform call. Est. by ephemeral-cache fallback PR #2387.
- `inference/core/registries/roboflow.py` — `get_model_type` (layered cache->registry->API resolution) and `_ensure_model_supported_on_this_deployment`: resolve-then-guard so an unserveable model fails loudly, not via a wrong backend.
- `inference/core/roboflow_api.py::get_roboflow_model_type` (~line 475) — the RIGHT way to handle a possibly-missing field: default only via `MODEL_TYPE_DEFAULTS`, else raise `MissingDefaultModelError`; never invent a model type.
- `inference_models/inference_models/models/auto_loaders/auto_negotiation.py` — `negotiate_model_packages` / `select_model_package_by_id`: pinned package id returns exactly-one-or-raises (`NoModelPackagesAvailableError`, `AmbiguousModelPackageResolutionError`) with `help_url`; no silent nearest-match. Clarity plumbing from PR #1434.
- `inference/core/env.py` — `DISABLED_INFERENCE_MODELS_BACKENDS` validated against `VALID_INFERENCE_MODELS_BACKENDS`, raising on unknown names (PR #2096): the pattern for backend-selection flags that fail on drift instead of silently no-op.
- `inference/core/interfaces/http/orjson_utils.py::orjson_response_keeping_parent_id` — preserves a contract field the SDK expects even when `exclude_none` would drop it (PR #1599): the reference for holding an SDK<->server shape stable.

## Severity guidance
- **Critical** — silent substitution/downgrade that returns wrong-but-plausible results when the caller pinned a model/backend; a security/auth header or credit-verification header assumed enforced but not honored by the deployed endpoint; reading a not-yet-deployed field on a hot path with no flag/guard (prod `KeyError`/`None`).
- **High** — `.get(key, default)` on external data where the default is semantically wrong (task-type guess); SDK<->server field/shape change on only one side with no ordered-rollout note; swallowed dependency error that yields a wrong success instead of a typed failure.
- **Medium** — new fallback that IS safe but lacks a log/metric to observe how often it triggers; default-model-version flip that keeps behavior but drops aliases for removed ids; missing `help_url`/typed exception on a new negotiation failure (reduced actionability, not incorrectness).
