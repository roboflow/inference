---
name: review-topic-auth-and-tenant-security
description: Load when a PR touches authentication, api-key/workspace resolution, permission or tenant/workspace scoping, assume-identity, serverless usage/authorization checks, or secret/token/api-key handling (env vars, headers, logs). Reviews the diff for fail-open authorization, cross-tenant leakage, and secret exposure.
---

# Review topic: Auth, tenant-boundary & secret handling

## When this applies
Trigger on CONTENT/behaviour, not just directory. Load this skill when the diff:
- Touches auth/authorization middleware: `inference/core/interfaces/http/http_api.py` (the `check_authorization_serverless` / `check_authorization` / workspace-allowlist middlewares), `middlewares/`, or `dependencies.py`.
- Resolves or compares api-keys, workspaces, or projects — `get_roboflow_workspace`, `get_roboflow_dataset_type`, `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`, `DEDICATED_DEPLOYMENT_WORKSPACE_URL`, cache keys keyed on api-key/workspace.
- Calls the serverless usage/authorization service (`get_serverless_usage_check_async`) or branches on its status code.
- Adds/reads/forwards secrets or tokens: anything in `inference/core/env.py` matching `*_SECRET`, `*_TOKEN`, `API_KEY`, `PASSWORD`, `PRELOAD_API_KEY`, Modal/HuggingFace tokens, `ROBOFLOW_INTERNAL_SERVICE_SECRET`, `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN`, or the assume-identity headers (`x-assume-identity-access-token`, `x-assume-identity-authorised-workspace`).
- Logs, echoes, serializes, or error-formats any request/URL/header that may carry an `api_key=` or bearer token.

## What to protect
1. **Fail-closed authorization.** When an authorization/usage-check outcome is ambiguous (upstream 5xx, timeout, unexpected status, missing field), the request MUST be denied, not admitted. A fail-open bug silently grants free/unauthorized inference and is invisible in normal traffic. This is the exact contract PR #2528 restored (see below).
2. **Tenant/workspace isolation.** A caller's api-key must only ever resolve to and act on ITS OWN workspace. Auth caches must be keyed so one tenant's result cannot be served to another. Assume-identity must scope to the authorised workspace only.
3. **No secret exposure.** Secrets, api-keys, and tokens must never be logged, echoed in error messages/responses, embedded in raised exceptions, or persisted (cache/DB/usage payloads) in clear form. URLs containing `api_key=` must be redacted before any `raise_for_status`, log, or metric label.

## What to check
1. **Every authorization branch has a deny default.** For each `if`/status-code branch in the auth middleware, trace the fall-through: does an unhandled/unexpected status still reach `call_next`? Unexpected upstream status must return an error (e.g. 503 "temporarily unavailable") and MUST NOT be cached (see PR #2528).
2. **Missing credential ⇒ reject.** Absent `api_key` (query and JSON body) returns 401, not a pass-through. Never store/act on usage when api-key is missing (PR #772).
3. **Cache key correctness.** Auth-result caches are keyed on `(api_key, enforce_credits_verification)` (or workspace), never on a broader key that could serve tenant A's grant to tenant B. Negative/denial results are cached deliberately; ambiguous results are NOT cached.
4. **Allowlisted-route audit.** Any addition to the unauthenticated-route allowlist (`/docs`, `/info`, `/workflows/blocks/describe`, health/metrics, `/model/registry`) must be genuinely safe to expose unauthenticated. Do not slip an inference/data route into the allowlist. Prefer exact-match over prefix-match on paths.
5. **Workspace boundary on identity.** Tenant checks compare the caller's resolved workspace against the deployment's allowed workspace(s) — `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` / `DEDICATED_DEPLOYMENT_WORKSPACE_URL` match (PR #988). Assume-identity headers are only added when `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN` is set AND an authorised workspace is present in context.
6. **Secret redaction on every egress.** `requests`/`aiohttp` responses whose URL may carry `api_key=` go through `api_key_safe_raise_for_status(_aiohttp)` before raising/logging. New log lines, exception `str()`, and Prometheus labels must not interpolate raw keys/tokens (PR #140, #255, #248).
7. **New secret env vars.** A new `*_SECRET`/`*_TOKEN`/api-key env in `env.py` defaults to `None`/absent (never a real value), is never printed at startup, and is not returned by any `/info`-style introspection endpoint.
8. **Revert awareness.** If the diff reverts an auth fix, confirm it does not re-open a fail-open hole (PR #2529 reverted #2528 — such reverts are High-risk and need explicit justification).

## Common failure modes
- **Fail-open on ambiguous upstream status** — an unexpected serverless usage-check status falls through to `call_next`, admitting the request for free. Fixed in **PR #2528** (return 503, do not cache); note it was reverted in **PR #2529**, so watch reverts.
- **Caching an ambiguous/unauthorized result under a reusable key**, later served as an allow to another caller. Caches must key on the credential and only cache definitive outcomes (**PR #772** — do not store usage when api-key missing).
- **api-key leaked in logs / error text / exceptions** via `raise_for_status()` on a URL containing `?api_key=...`, or SDK errors echoing the key (**PR #140** redact in requests utils; **PR #255**/**#248** deduct api-key from SDK client errors; **PR #188** safe missing-key handling). Also **PR #1831** — do not trim/mangle the api-key hash used for usage identity.
- **Over-broad tenant check** — matching on project/dataset existence rather than the caller's own workspace, or an extra check that adds latency without security value (**PR #988** simplified to rely on workspace-ID match).
- **Unauthenticated route allowlist creep** — adding a data/inference path to the "open" list (guard when reviewing changes near the `/docs`,`/info`,`/model/registry` allowlist).
- **SSRF / trusting user-supplied URL after validation** — validation bypassed by redirect, letting a request reach internal/cross-tenant hosts (**PR #2501** SSRF-via-redirect, **PR #2500**/#497/#957 image_utils path/host hardening). Relevant when auth/tenant boundary is enforced by URL/host validation.
- **Self-hosted server open by default** — new deployment modes must document/enforce that auth is off unless the workspace allowlist or an external proxy is configured (**PR #2417**).

## Example implementations (point here)
- `inference/core/interfaces/http/http_api.py` — `check_authorization_serverless` middleware: the canonical fail-closed serverless auth path (401 on missing/unauthorized key, deny + no-cache 503 on unexpected upstream status, credential-scoped `cached_api_keys` keyed on `(api_key, enforce_credits_verification)`). Established/repaired by **PR #2528**.
- `inference/core/interfaces/http/http_api.py` — the `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT` / `DEDICATED_DEPLOYMENT_WORKSPACE_URL` middleware: correct workspace-boundary enforcement with an explicit, minimal unauthenticated-route allowlist. Simplified to pure workspace-ID match in **PR #988**; local-deployment allowlist added in **PR #2417**.
- `inference/core/utils/requests.py` — `api_key_safe_raise_for_status`, `api_key_safe_raise_for_status_aiohttp`, `deduct_api_key` (regex `api_key=(.[^&]*)` → `api_key=xx***yy`): the required redaction seam before raising/logging on Roboflow API calls. From **PR #140**.
- `inference/core/roboflow_api.py` — `get_roboflow_workspace(api_key)` and `_add_assume_identity_headers`: correct api-key→workspace resolution (fails if workspace empty) and gated assume-identity header injection (`ASSUME_IDENTITY_ACCESS_TOKEN_HEADER` only added when the service token AND `assume_identity_authorised_workspace_db_id` context are both present).
- `inference/core/env.py` — secret/token declarations (`ROBOFLOW_INTERNAL_SERVICE_SECRET`, `ROBOFLOW_ASSUME_IDENTITY_SERVICE_ACCESS_TOKEN`, `MODAL_TOKEN_SECRET`, `PRELOAD_API_KEY`, `WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT`): every secret defaults to `None`/unset — pattern-match new secrets against this.
- `inference/usage_tracking/collector.py` — `_calculate_api_key_hash` / `_hashed_api_keys`: api-keys are hashed (never stored raw) in usage payloads, and usage is skipped when the key is absent (**PR #772**, #1831).

## Severity guidance
- **Critical** — any fail-open authorization path (ambiguous/upstream-error outcome admits the request), caching that can serve one tenant's grant to another, or a raw secret/api-key/token written to logs, responses, exceptions, or persistent storage. Reverting a prior fail-closed fix without compensating control is Critical.
- **High** — over-broad or missing workspace/tenant boundary check; adding a data/inference route to the unauthenticated allowlist; api-key resolvable to a workspace other than the caller's; assume-identity headers added without the authorised-workspace gate; SSRF/host-validation bypass on a tenant-boundary URL.
- **Medium** — auth-cache keyed correctly but with weak TTL/eviction; redaction applied inconsistently (some egress paths still raw) but not on the hottest logging path; new secret env var lacking a `None` default or exposed via introspection; partial-key prefixes revealed in logs beyond the 2-char convention.
