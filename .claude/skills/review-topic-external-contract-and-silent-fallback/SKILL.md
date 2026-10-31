---
name: review-topic-external-contract-and-silent-fallback
description: Load when a PR touches inference/core/roboflow_api.py, inference/core/registries/roboflow.py, or the inference_models auto_negotiation resolver; changes an inference_sdk<->server response shape (response_model, model_dump, parent_id); reads a NEW nested platform key (api_data.get("taskType"), modelVariant); adds fallback / backend-downgrade / .get(key, default) on external data; or adds an env flag/secret changing which service/backend is contacted (USE_INFERENCE_MODELS, DISABLED_INFERENCE_MODELS_BACKENDS).
---

# Review topic: External/platform contract drift & silent fallback

## When this applies
Load when the diff shows ANY of these content signals (paths are hints, not the trigger):
- Calls into `inference/core/roboflow_api.py`, `inference/core/registries/roboflow.py`, the inference_models auto-loader (`auto_negotiation.py`), or the license/weights registry — new endpoint, new query param, new request header, or a changed URL suffix.
- Reads a NEW field/nested key of an API/JSON response (`response["version"]["modelType"]`, `api_data.get("taskType")`, `modelVariant`), or changes which nested shape is read.
- Changes an `inference_sdk` <-> server contract: `model_dump`/serialization, `response_model`, field presence/aliases, or default values a client parses.
- Adds/loosens FALLBACK, auto-conversion, `.get(key, <default>)`, backend/model auto-negotiation, backend downgrade, or "retry the other path" logic when a requested model/backend/capability is unavailable.
- Adds an env flag/secret changing which service/backend is contacted (`USE_INFERENCE_MODELS`, `DISABLED_INFERENCE_MODELS_BACKENDS`, `ROBOFLOW_INTERNAL_SERVICE_SECRET`).
- Assumes a companion PR (platform/backend/SDK) is already deployed.

## Review checklist
Tag each finding. Fix BLOCK before merge; raise FLAG; NIT is optional.

- **BLOCK** — Caller pinned a model/backend/package but resolution can silently substitute a different one. Pinned `requested_model_package_id` must return exactly that or raise (`NoModelPackagesAvailableError` / `AmbiguousModelPackageResolutionError`), never a nearest match. (See rule 3.)
- **BLOCK** — A `.get(key, <default>)` on external data whose default is semantically wrong. Raise instead when correctness depends on it. (Rule 2.)
- **BLOCK** — A new required field / nested response key read on a hot path with no deploy guarantee or flag guard → prod `KeyError`/`None` before the platform emits it. (Rule 1.)
- **BLOCK** — A security/credit header (`X-Roboflow-Internal-Service-Secret` / `service_secret`, `countinference`) assumed enforced but the deployed endpoint may ignore it — an ignored auth header is worse than an error. (Rule 6.)
- **BLOCK** — A dependency (cache / registry / license server) failure swallowed into a wrong-but-successful response; a bare `except Exception` that returns a default. Must raise a typed error (`CacheUnavailableError`) at a decision point, then fall through consciously. (Rule 8.)
- **FLAG** — Resolved model/task type not re-checked against what THIS deployment can serve. (Rule 4.)
- **FLAG** — SDK<->server field/shape change on only one side with no ordered-rollout note or companion PR. (Rules 5, 9.)
- **FLAG** — A new external-call path that catches broadly and returns `None`/defaults instead of routing through `wrap_roboflow_api_errors` → typed exception. (Rule 7.)
- **NIT** — A new fallback that IS safe but has no log/metric to observe how often it fires; a negotiation failure missing `help_url`.

### Not blocking
- Adding a `.get(key, default)` where the default is genuinely safe for older projects/tenants (the layered defaulting in `get_model_type` handling missing `modelType`/`type` is the SANCTIONED pattern, not a violation).
- Default-model-version flips that keep behavior AND retain aliases for removed ids — flag only if aliases are dropped.
- A fallback whose absence of a metric is the only gap — that is a NIT, not a merge blocker.
- Do not demand a companion-PR note when the change is behind a flag defaulting off.

## Standards (one canonical statement per rule)
1. **New response field/shape must be guarded.** Reading `data["x"]["y"]` or `.get("y", default)` requires confirming the platform already returns `y` for the relevant model ages/tenants. Old projects often lack `modelType`/`type` — see the layered `get_model_type` defaulting in `inference/core/registries/roboflow.py`. A not-yet-deployed nested field parsed before the platform emits it yields prod `None`/`KeyError` (`modelVariant` from RFAPI, #1641).
2. **Silent default on missing external data.** Every `.get(key, <fallback>)` on platform data: is the fallback SAFE or does it mask real unavailability? `api_data.get("taskType", "object-detection")` mislabels non-detection models if the field is genuinely absent. Prefer raising (`MissingDefaultModelError`, `ModelArtefactError`) over guessing (guard around missing attrs, #2105).
3. **Backend / model downgrade.** Auto-negotiation must not substitute a different backend/quantization/model when the requested one is unavailable. A pinned `requested_model_package_id` resolves to exactly one or raises (`NoModelPackagesAvailableError`, `AmbiguousModelPackageResolutionError`), never the closest package. Global backend disable must validate names against `VALID_INFERENCE_MODELS_BACKENDS` and fail on typos (#2096); ranking/discard-reason clarity plumbing (#1434, ranking #2047/#1811).
4. **Deployment capability guard.** After resolving a model/task type, verify it against what THIS deployment can serve — `_ensure_model_supported_on_this_deployment` in `inference/core/registries/roboflow.py`. A resolved-but-unserveable model must fail clearly, not attempt a wrong backend.
5. **SDK<->server both sides.** A serialization/`response_model`/field-alias change must keep the client parser tolerant AND keep the server emitting fields the client requires (e.g. `parent_id` presence in OCR responses, #1599). Check both directions in the same PR or confirm ordered rollout.
6. **New param / header actually honored.** A new query param or auth/credit header (`service_secret` / `X-Roboflow-Internal-Service-Secret`, `countinference`) must be recognized by the deployed endpoint. Confirm the server side exists — a contract added deliberately but not yet honored is worse than an error (#2219).
7. **Error mapping preserved.** External errors route through `wrap_roboflow_api_errors` → `DEFAULT_ERROR_HANDLERS` typed exceptions (401→`RoboflowAPINotAuthorizedError`, 402→`PaymentRequiredError`, timeout→`RoboflowAPITimeoutError`). A new path that catches broadly and returns `None`/defaults defeats this.
8. **Swallowed dependency failure.** A bare `except Exception` around a registry/cache/license call returning a default is a red flag. Correct pattern (#2387): cache-unreachable raises `CacheUnavailableError` at a decision point, and only then does the caller consciously fall through to the authoritative API.
9. **Companion-PR assumption.** If the change only works once a platform/backend/SDK PR ships, require a flag default-off, an ordered-rollout note, or graceful pre-deploy behavior. Deprecated-default flips (e.g. Gemini 2.0→2.5, #2395) must keep aliases for removed ids or they strand pinned callers.

## Key files & reference PRs
- `inference/core/roboflow_api.py` — `wrap_roboflow_api_errors` + `DEFAULT_ERROR_HANDLERS` (status→typed exception): the canonical fail-clear boundary for every platform call.
- `inference/core/roboflow_api.py::get_roboflow_model_type` — right way to handle a possibly-missing field: default only via `MODEL_TYPE_DEFAULTS`, else raise `MissingDefaultModelError`; never invent a model type.
- `inference/core/registries/roboflow.py::get_model_type` + `_ensure_model_supported_on_this_deployment` — layered cache→registry→API resolution, then resolve-then-guard so an unserveable model fails loudly.
- `inference_models/inference_models/models/auto_loaders/auto_negotiation.py` — `negotiate_model_packages` / `select_model_package_by_id`: pinned package id returns exactly-one-or-raises (`NoModelPackagesAvailableError`, `AmbiguousModelPackageResolutionError`) with `help_url`; no silent nearest-match.
- `inference/core/env.py` — `DISABLED_INFERENCE_MODELS_BACKENDS` validated against `VALID_INFERENCE_MODELS_BACKENDS`, raising on unknown names: pattern for backend-selection flags that fail on drift.
- `inference/core/interfaces/http/orjson_utils.py::orjson_response_keeping_parent_id` — holds an SDK<->server shape stable by preserving a `parent_id` field `exclude_none` would drop.

Reference PRs: #1434 (discard-reason clarity), #1599 (parent_id SDK<->server), #1641 (modelVariant not-yet-deployed field), #1811/#2047 (ranking), #2096 (backend-name validation), #2105 (missing-attr guard), #2219 (service-secret header contract), #2387 (CacheUnavailableError fallback), #2395 (Gemini default flip / aliases).
