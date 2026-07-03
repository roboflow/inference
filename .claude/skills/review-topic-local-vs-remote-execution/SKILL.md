---
name: review-topic-local-vs-remote-execution
description: 'Load when a PR touches how a step/model runs in-process vs. over the network: Workflow-block execution-mode dispatch (self._step_execution_mode, StepExecutionMode.LOCAL/REMOTE, run_locally/run_remotely), a block get_restrictions()/get_air_gapped_availability(), backend/runtime routing env (USE_INFERENCE_MODELS, DISABLED_INFERENCE_MODELS_BACKENDS, WORKFLOWS_REMOTE_API_TARGET, inference_models_adapters.py), CUDA/Jetson/ONNX gating, or a new block with network/LAN/same-process assumptions.'
---

# Review topic: Local vs remote / hosted execution compatibility

## When this applies
Load when a diff does any of:
- Adds/edits a `run()` dispatch on `self._step_execution_mode` (LOCAL vs REMOTE) or a `run_locally` / `run_remotely` pair.
- Adds/edits a block's `get_restrictions()` or `get_air_gapped_availability()` where the block reaches the network or needs same-process / LAN access (PTZ/ONVIF camera, local filesystem, GPU-resident handles).
- Touches runtime routing env: `USE_INFERENCE_MODELS`, `DISABLED_INFERENCE_MODELS_BACKENDS`, `WORKFLOWS_STEP_EXECUTION_MODE`, `WORKFLOWS_REMOTE_API_TARGET`, `LOCAL_INFERENCE_API_URL`, `_PLATFORM_SPECIFIC_USE_INFERENCE_MODELS_DEFAULT`, or backend selection in `inference/core/models/inference_models_adapters.py` / `inference/models/utils.py`.
- Gates behavior on hardware/deps: CUDA version, Jetson (`jetson`, `l4t`, JP5/JP6), ONNX execution providers, TRT — where one runtime path is fixed/tested but siblings are not.
- Changes hosted/serverless/dedicated cold-start / processing-time headers or WebRTC/modal worker plumbing.

This is a behavior/content trigger, not a single directory — the paths above are hints.

**Cross-refs (owned elsewhere, not reviewed here):**
- Serverless auth/usage fail-open behavior → `review-topic-auth-and-tenant-security`.
- Stateful-block restriction contract (in-process trackers/counters/cooldown/append-log needing a SOFT restriction) → `review-topic-workflow-state-management`.

## Review checklist

- **BLOCK** — `run()` dispatches on `StepExecutionMode` with an explicit `else: raise ValueError` for unknown modes (missing else swallows new modes silently).
- **BLOCK** — a runtime-routing change (`USE_INFERENCE_MODELS`, `DISABLED_INFERENCE_MODELS_BACKENDS`, platform default) leaves BOTH the on and off value working and tested; the legacy ONNX registry entries are still present. Breaking the DEFAULT runtime with no working fallback is a production regression.
- **BLOCK** — a block that raises / hard-fails on a runtime the platform will still offer carries a matching `RuntimeRestriction` (right `severity`, `applies_to_step_execution_modes`, `applies_to_runtimes`); metadata that lies makes the UI offer a path that fails at runtime.
- **FLAG** — `run_locally` and `run_remotely` have matching signatures and forward every manifest param; a param wired into only one path is a divergence users hit only on the other runtime.
- **FLAG** — a new network/LAN/same-process block overrides `get_air_gapped_availability()` if it needs internet, and adds a HARD restriction excluding hosted/dedicated (LAN) or REMOTE step execution (same-process).
- **FLAG** — a platform-specific default (`_PLATFORM_SPECIFIC_USE_INFERENCE_MODELS_DEFAULT`, Windows/Jetson forks) keeps the non-default OS working; the diff must not assume one platform.
- **FLAG** — CUDA/Jetson/ONNX-provider introspection changes are correct on every runtime, not just the author's box (one-runtime introspection bugs mis-route model loading).
- **NIT** — restriction present but mis-scoped or wrong severity (HARD where SOFT fits, missing `applies_to_runtimes`).

### Not blocking
- Do NOT demand a hosted-platform test for a change with no observable hosted/serverless behavior difference (pure local refactor, docstrings, typing). Hosted coverage under `tests/inference/hosted_platform_tests/` is required only when the diff changes what a request returns on lambda / serverless / dedicated.
- Do NOT demand a restriction for a block that is genuinely runtime-agnostic (stateless, no network, no LAN, no GPU-handle assumption).
- Do NOT demand a flip-the-flag-both-ways test for env vars that only tune values (URLs, timeouts) without forking code paths.

## Standards

1. **Execution-mode dispatch is exhaustive.** `run()` branches on `self._step_execution_mode` and terminates in `else: raise ValueError(f"Unknown step execution mode: ...")`. Canonical shape: `inference/core/workflows/core_steps/models/roboflow/object_detection/v1.py` — `run()` branches `StepExecutionMode.LOCAL` / `StepExecutionMode.REMOTE`, `run_locally` via `self._model_manager`, `run_remotely` via `InferenceHTTPClient` honoring `WORKFLOWS_REMOTE_API_TARGET` / `LOCAL_INFERENCE_API_URL`. (`StepExecutionMode` is defined in `inference/core/workflows/prototypes/block.py` and re-exported from `core_steps/common/entities.py`.)

2. **Local and remote paths stay semantically equivalent.** `run_locally` and `run_remotely` accept the same params and return the same shape; every manifest field is forwarded on both paths (one-path param wiring is a silent divergence).

3. **Metadata must match behavior.** A block's `get_restrictions()` / `get_air_gapped_availability()` declarations are consumed by the describe endpoint and the platform UI to decide where a block may run. The declaration and the actual `run()` guard must agree — a code path that raises for a runtime needs a `RuntimeRestriction` for it, and vice-versa. LAN/same-process shape: `onvif_movement/v1.py` (HARD + `StepExecutionMode.REMOTE`, plus HARD + `Runtime.HOSTED_SERVERLESS`/`Runtime.DEDICATED_DEPLOYMENT` because hosted can't reach a customer LAN).

4. **Runtime routing has no orphaned path.** A new backend, env flag, or hardware gate leaves BOTH the on and off value working and tested; the legacy path (ONNX registry entries) stays intact. Backend routing lives in `inference/core/models/inference_models_adapters.py` + `inference/core/env.py` (`USE_INFERENCE_MODELS`, `_PLATFORM_SPECIFIC_USE_INFERENCE_MODELS_DEFAULT`, `DISABLED_INFERENCE_MODELS_BACKENDS`). Rollout history: #2096 added per-deployment `DISABLED_INFERENCE_MODELS_BACKENDS`; #2128 added missing registry entries so RFDETR-NAS works under inference-models; #2136 reverted making inference-models the default backend (the default flip is contentious — review carefully).

5. **Per-platform introspection is correct on every runtime.** CUDA-version / Jetson / ONNX-provider detection that gates model loading must hold across platforms, not just the author's (CUDA acquisition fixed in #2515; Jetson environment detection in runtime introspection + dockerfiles fixed in #1796).

6. **Hosted-only code paths get hosted coverage.** Behavior that differs on lambda / serverless / dedicated is exercised in `tests/inference/hosted_platform_tests/` (parametrized over `PlatformEnvironment` via `conftest.py`; see `test_workflows.py`, `test_roboflow_models.py`). Hosted-only plumbing regresses silently under local-only tests (remote cold-start header aggregation reverted in #2222; `ROBOFLOW_ENVIRONMENT` propagation to the modal webrtc worker fixed in #2179; dedicated `hosted=True`/`DEDICATED_DEPLOYMENT_ID` usage attribution fixed in #817).

## Key files & Reference PRs

Files:
- `inference/core/workflows/prototypes/block.py` — the contract: `StepExecutionMode`, `Runtime`, `Severity`, `RuntimeRestriction`, presets (`STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION`, `COOLDOWN_HTTP_SOFT_RESTRICTION`), and `get_restrictions` / `get_air_gapped_availability` / `get_execution_engine_compatibility` hooks. Read first.
- `inference/core/workflows/core_steps/models/roboflow/object_detection/v1.py` — canonical LOCAL/REMOTE dispatch.
- `inference/core/workflows/core_steps/sinks/onvif_movement/v1.py` — same-process + LAN block done right (two HARD restrictions).
- `inference/core/models/inference_models_adapters.py` + `inference/core/env.py` — backend routing with per-platform default and per-deployment disable.
- `tests/inference/hosted_platform_tests/conftest.py` (+ `test_workflows.py`, `test_roboflow_models.py`) — hosted/serverless/localhost parametrization.

Reference PRs: #2096, #2128, #2136 (backend rollout / default flip), #2515, #1796 (per-platform introspection), #2222, #2179, #817 (hosted-only plumbing).
