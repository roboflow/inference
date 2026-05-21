## Why

Custom Python Blocks in Workflows execute in Modal sandboxes via a FastAPI web endpoint. Today the per-frame budget is hard-coded in two mismatched places — Modal kills the container at 20s (`modal/modal_app.py:64`), while the HTTPS client waits 30s (`inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py:303`) — and surfaces as a generic `Failed to connect to Modal endpoint` error with no diagnostic context. Batch-processing users running slower-but-legitimate blocks (image segmentation, large polygon ops, network calls to internal services) hit the 20s ceiling with no recourse and no actionable error, and the CS-237 failure classifier has no stable seam to recognise these timeouts and produce useful hint copy. Making the timeout user-configurable (up to 120s — Modal's per-invocation ceiling caps the architecture lower than we'd like) unblocks those workloads, lets the worker advance to the next frame promptly at the user's chosen deadline, and gives the classifier a recognisable error class to match on.

## What Changes

- Add `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` environment variable (1–120, default 20) consumed by `ModalExecutor` to derive both the in-handler watchdog deadline and the HTTPS client timeout (`N + 10` for headroom).
- Raise the Modal `@app.cls(timeout=...)` decorator from 20 → 130 so the configurable watchdog (max 120) always wins the race against Modal's per-invocation hard kill.
- Add an in-handler `ThreadPoolExecutor` watchdog around user-code execution in `modal/modal_app.py:execute_block` that returns a structured timeout response when the configured deadline elapses. Be explicit (in code and in this proposal) that the watchdog **cannot terminate the orphan thread** — user code may keep consuming Modal compute on the container until the container is recycled, because `@modal.fastapi_endpoint` does not honour `Cls.with_options(timeout=...)` per-call overrides.
- Add a thread-binding helper `bind_capture_to(stdout_buf, stderr_buf)` in `error_utils.py` so the watchdog can pre-allocate `StringIO` buffers in the main thread and have the worker thread tee its writes into them — preserving stdout/stderr in the timeout response.
- Add a new exception `DynamicBlockTimeoutError(DynamicBlockCodeError)` — execution-engine-side, not compiler-side — that `ModalExecutor` raises both on the new `error_type=CustomPythonBlockTimeout` server response and on bare `requests.exceptions.ReadTimeout` (defence in depth). The wire string `"CustomPythonBlockTimeout"` is promoted to a single named constant `MODAL_TIMEOUT_ERROR_TYPE` shared by client and server.
- Re-class the existing `requests.exceptions.RequestException` catch at `modal_executor.py:369-373` from `DynamicBlockError` (`WorkflowCompilerError`) to a `WorkflowExecutionEngineError`-side exception — fixing a pre-existing categorisation bug where runtime HTTP failures masquerade as compile-time errors.
- The validation codepath (`validate_code_in_modal`) explicitly passes a small fixed `timeout_seconds` (e.g. 30) in its request payload so that bumping the env var for the per-frame budget does not expand the syntax-check window.
- Add `--custom-python-block-timeout` CLI flag (1–120, validated) to both `process_images_with_workflow` and `process_videos_with_workflow` in `inference_cli/lib/roboflow_cloud/batch_processing/core.py`.
- Add `custom_python_block_timeout_seconds: Optional[int]` field on `WorkflowsProcessingSpecificationV1` (`inference_cli/.../entities.py`) with `serialization_alias="customPythonBlockTimeoutSeconds"`, threaded through `trigger_job_with_workflows_images_processing` / `..._videos_processing` payload builders in `api_operations.py`.

## Capabilities

### New Capabilities
- `custom-python-block-execution`: Per-frame execution semantics for Custom Python Blocks running on Modal — configurable timeout, structured timeout error class with captured stdout/stderr, layered watchdog/client/decorator deadlines, and the SDK/CLI surface that lets batch-job submitters set the timeout on a per-job basis.

### Modified Capabilities
<!-- None — no existing OpenSpec specs in this repo. -->

## Impact

- **Affected code (inference repo):**
  - `modal/modal_app.py` — decorator timeout bump, watchdog around user-code execution, structured timeout response.
  - `inference/core/env.py` — new `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` env var.
  - `inference/core/workflows/errors.py` — new `DynamicBlockTimeoutError(DynamicBlockCodeError)`.
  - `inference/core/workflows/execution_engine/v1/dynamic_blocks/error_utils.py` — new `bind_capture_to(...)` helper.
  - `inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py` — read env var, pass `timeout_seconds` in payload, use `N + 10` request timeout, raise typed timeout exception, re-class generic `RequestException` catch, pass explicit small `timeout_seconds` from `validate_code_in_modal`.
  - `inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py` (new) — `MODAL_TIMEOUT_ERROR_TYPE = "CustomPythonBlockTimeout"`.
  - `inference_cli/lib/roboflow_cloud/batch_processing/core.py` — `--custom-python-block-timeout` flag on both image and video commands.
  - `inference_cli/lib/roboflow_cloud/batch_processing/entities.py` — new field on `WorkflowsProcessingSpecificationV1`.
  - `inference_cli/lib/roboflow_cloud/batch_processing/api_operations.py` — thread field through trigger payload builders.
  - `tests/workflows/integration_tests/execution/test_workflow_with_custom_python_block_modal.py` — new timeout integration tests (gated on Modal credentials) + unit tests for the watchdog binding helper.

- **Wire/API contracts introduced:**
  - Modal request payload gains optional `timeout_seconds: int`.
  - Modal response gains stable `error_type: "CustomPythonBlockTimeout"` string on timeout (consumed by CS-237 classifier).
  - SDK/orchestrator wire format gains `customPythonBlockTimeoutSeconds` under `processingSpecification`.

- **Out of scope (explicit):**
  - Non-Modal execution mode (`WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=local` and self-hosted inference-server users running in-process).
  - Per-block-type or per-step timeouts.
  - Migrating away from `@modal.fastapi_endpoint` to Modal RPC (which would allow per-call `Cls.with_options(timeout=...)` and eliminate the orphan-thread compute waste). Tracked as future work.
  - The `batch-processing-services` orchestrator changes and the `roboflow` monorepo UI/backend changes from the broader CS-239 proposal — those land in separate proposals.
  - The CS-239 sub-bullet "bubble up timeout error to the user" UI work — satisfied by the CS-237 classifier matching on the new stable error class; no additional UI work in inference.

- **Deploy sequencing (required ordering):**
  1. Modal app (`modal_app.py`) deploys with `@app.cls(timeout=130)` and the watchdog **first**.
  2. Worker-side `inference` package release with `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` plumbing **second**.
  3. SDK/CLI release with the `--custom-python-block-timeout` flag **third**.
  Shipping worker changes that request `timeout_seconds > 20` before the Modal deploy lands will silently regress to the deployed 20s decorator kill.

- **Acknowledged limitation:** This proposal does not guarantee Modal compute stops at the configured deadline. The watchdog returns a structured response so the worker can advance, but Python cannot kill the running thread and `@modal.fastapi_endpoint` blocks per-call timeout overrides. Pathological user code (infinite loops, long network calls) keeps consuming Modal compute until the container is recycled. Eliminating this requires migrating to Modal RPC or subprocess isolation — out of scope here.
