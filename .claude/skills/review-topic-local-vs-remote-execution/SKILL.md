---
name: review-topic-local-vs-remote-execution
description: Load when a PR touches how a step/model runs in-process vs. over the network — Workflow block execution-mode handling or restriction metadata (StepExecutionMode / get_restrictions / Runtime), hosted/serverless/dedicated-deployment behavior, backend or runtime routing (USE_INFERENCE_MODELS, WORKFLOWS_REMOTE_*, WORKFLOWS_STEP_EXECUTION_MODE, inference_models adapters, CUDA/Jetson gating), or adds an integration with network / same-process / LAN assumptions. Guards that local and remote execution paths stay behavior-compatible and honestly documented.
---

# Review topic: Local vs remote / hosted execution compatibility

## When this applies
Load this skill when a diff does any of:
- Adds or edits a `run()` dispatch on `self._step_execution_mode` (LOCAL vs REMOTE), a `run_locally` / `run_remotely` pair, or a block's `get_restrictions()` / `get_air_gapped_availability()` / `get_execution_engine_compatibility()`.
- Adds a new Workflow block that reaches the network, holds cross-request state (tracking, counting, cooldown, append-log), or needs same-process / LAN access (e.g. PTZ/ONVIF camera, local filesystem, GPU-resident handles).
- Touches runtime routing env: `USE_INFERENCE_MODELS`, `DISABLED_INFERENCE_MODELS_BACKENDS`, `WORKFLOWS_STEP_EXECUTION_MODE`, `WORKFLOWS_REMOTE_API_TARGET`, `LOCAL_INFERENCE_API_URL`, `HOSTED_*_URL`, or `inference/core/models/inference_models_adapters.py` / `inference/models/utils.py` backend selection.
- Changes hosted/serverless/dedicated-deployment behavior: serverless auth/usage gates in `http_api.py`, `LAMBDA` / `GCP_SERVERLESS` / `DEDICATED_DEPLOYMENT_ID` branches, cold-start / processing-time headers, or WebRTC/modal worker plumbing in the streaming pipeline.
- Gates behavior on hardware/deps: CUDA version, Jetson (`jetson`, `l4t`, JP5/JP6), ONNX execution providers, TRT — where one runtime path is fixed or tested but siblings are not.

This is a behavior/content trigger, not a single directory: paths above are hints.

## What to protect
1. **Metadata must match behavior.** A block's `get_restrictions()` / air-gapped / compatibility declarations are consumed by the describe endpoint and the platform UI to decide where a block may run. If the code raises `ValueError` on remote execution but declares no HARD restriction (or vice-versa), the platform offers an option that fails at runtime, or hides one that works. The declaration and the actual `run()` guard must agree.
2. **Local and remote paths stay semantically equivalent.** `run_locally` (via `self._model_manager`) and `run_remotely` (via `InferenceHTTPClient`) must accept the same params and return the same shape; a param wired into only one path is a silent divergence users hit only on the other runtime.
3. **Stateful blocks are honest about stateless runtimes.** In-process state (tracker IDs, counters, cooldown timers, append-log buffers) is meaningless on multi-replica / serverless HTTP where successive requests hit different workers. Such blocks must carry a SOFT restriction scoped to `StepExecutionMode.REMOTE` + hosted/dedicated runtimes.
4. **Runtime routing has no orphaned path.** A new backend (inference_models), env flag, or hardware gate must leave BOTH the on and off value working and tested. Fixing only the flag-on path is a regression for every deployment on the default.
5. **Fail closed at trust boundaries.** Serverless auth/usage checks that error out must deny (503/error), not silently allow or cache a bad decision.

## What to check
1. Does the block dispatch on `StepExecutionMode` with an explicit `else: raise ValueError(f"Unknown step execution mode...")`? A missing else swallows new modes silently.
2. Do `run_locally` and `run_remotely` have matching signatures and forward every manifest param? Diff the two arg lists.
3. If code paths raise/degrade for a runtime, is there a corresponding `RuntimeRestriction` with the right `severity`, `applies_to_step_execution_modes`, and `applies_to_runtimes`? Prefer reusing the presets in `prototypes/block.py` (`STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`, `COOLDOWN_HTTP_SOFT_RESTRICTION`).
4. New network/LAN/same-process block: does it override `get_air_gapped_availability()` (if it needs internet) and add a HARD restriction excluding hosted/dedicated (LAN) or REMOTE step execution (same-process)?
5. Runtime routing env change: is the non-default value still exercised? Check for a test that flips the flag both ways. New `USE_INFERENCE_MODELS` / `DISABLED_INFERENCE_MODELS_BACKENDS` behavior must not break the legacy ONNX path (registry entries present for both).
6. Platform-specific default (`_PLATFORM_SPECIFIC_USE_INFERENCE_MODELS_DEFAULT`, Windows/Jetson forks): does the diff keep the non-default OS working, or does it assume one platform?
7. Hosted-platform / serverless tests: does the change add or update a case in `tests/inference/hosted_platform_tests/` (parametrized over `PlatformEnvironment`: lambda / serverless / localhost)? Behavior asserted only on localhost is untested for hosted.
8. Serverless auth/usage edits: does every branch of the upstream status handling terminate in an explicit allow or deny (no fall-through that fails open)?

## Common failure modes
- **Metadata/behavior drift** — block raises on remote but the restriction is absent, wrong severity, or unscoped. See the correct shape in `onvif_movement/v1.py` (HARD + REMOTE, plus HARD + hosted for LAN).
- **One-path param wiring** — a new manifest field forwarded in `run_locally` but not `run_remotely` (or vice-versa); users see it work on one runtime only.
- **Fail-open serverless auth** — an unhandled upstream status silently allowing/caching. Fixed by failing closed with 503 in **PR #2528** (and note the earlier revert **PR #2529** — this behavior is contentious, review carefully).
- **Backend rollout breaks the default** — flipping inference-models to default without the legacy path intact. **PR #2136** reverted "make inference-models the default backend via env flag"; **PR #2128** added missing registry entries so RFDETR NAS works under the inference-models backend; **PR #2096** added `DISABLED_INFERENCE_MODELS_BACKENDS` so a backend can be turned off per-deployment.
- **Runtime introspection wrong per platform** — **PR #2515** fixed CUDA-version acquisition; **PR #1796** fixed Jetson environment detection in runtime introspection + dockerfiles. One-runtime introspection bugs mis-route model loading.
- **Remote/hosted plumbing regressions** — **PR #2222** reverted aggregating remote cold-start data in workflow headers; **PR #2179** fixed passing `ROBOFLOW_ENVIRONMENT` to the modal webrtc worker per project. Hosted-only code paths regress silently because local tests don't cover them.
- **Hosted-only signal not set** — **PR #817** set `hosted=True` (`DEDICATED_DEPLOYMENT_ID`) in usage tracking on dedicated deployments; local runs looked fine, hosted attribution was wrong.

## Example implementations (point here)
- `inference/core/workflows/prototypes/block.py` — the contract: `StepExecutionMode`, `Runtime`, `Severity`, `RuntimeRestriction`, and the shared presets (`STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`, `COOLDOWN_HTTP_SOFT_RESTRICTION`) plus `get_restrictions` / `get_air_gapped_availability` / `get_execution_engine_compatibility` hooks. Read this first.
- `inference/core/workflows/core_steps/models/roboflow/object_detection/v1.py` — canonical LOCAL/REMOTE dispatch: `run()` branches on `self._step_execution_mode` with an explicit `else: raise ValueError`, `run_locally` via `self._model_manager`, `run_remotely` via `InferenceHTTPClient` honoring `WORKFLOWS_REMOTE_API_TARGET` / `LOCAL_INFERENCE_API_URL` and matching params.
- `inference/core/workflows/core_steps/sinks/onvif_movement/v1.py` — same-process + LAN block done right: two HARD restrictions (REMOTE step execution, and hosted/dedicated runtimes can't reach customer LANs).
- `inference/core/workflows/core_steps/sinks/s3/v1.py` — SOFT restriction for in-process append-log buffering under remote/serverless step execution; shows scoping to both `applies_to_step_execution_modes` and `applies_to_runtimes`.
- `inference/core/models/inference_models_adapters.py` + `inference/core/env.py` (`USE_INFERENCE_MODELS`, `_PLATFORM_SPECIFIC_USE_INFERENCE_MODELS_DEFAULT`, `DISABLED_INFERENCE_MODELS_BACKENDS`) — backend routing with per-platform default and per-deployment disable, both paths preserved (established/repaired by **PR #2096**, **PR #2128**; guarded by revert **PR #2136**).
- `tests/inference/hosted_platform_tests/conftest.py` (+ `test_workflows.py`, `test_roboflow_models.py`) — how hosted/serverless/localhost coverage is parametrized over `PlatformEnvironment`; the place new hosted behavior must be exercised.

## Severity guidance
- **Critical** — code raises/hard-fails on a runtime the platform will still offer (metadata says it's allowed), OR a serverless auth/usage path fails open, OR a backend/flag change breaks the DEFAULT runtime with no working fallback. Users hit a hard failure or a security/billing gap in production.
- **High** — local vs remote param/return divergence, a new LAN/same-process/network block with no restriction or air-gapped declaration, or a routing change tested on only one flag/platform value. Wrong or missing results on one runtime, discoverable only in the field.
- **Medium** — restriction present but mis-scoped or wrong severity (e.g. HARD where SOFT fits, missing `applies_to_runtimes`), or hosted-platform test not updated for changed hosted behavior. Metadata is imprecise but not actively lying; degraded UX or thin coverage.
