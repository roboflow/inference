## 1. Shared wire constant

- [x] 1.1 Create `inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py` with `MODAL_TIMEOUT_ERROR_TYPE: Final[str] = "CustomPythonBlockTimeout"` and a brief module docstring marking the value as a public contract with the CS-237 classifier.

## 2. Modal app (server-side)

- [x] 2.1 In `modal/modal_app.py`, change the `@app.cls(...)` decorator at line ~61-70 to set `timeout=130` (replacing today's `timeout=20`).
- [x] 2.2 In `modal/modal_app.py:execute_block`, read an optional `timeout_seconds: int` from the request payload; clamp to `[1, 120]`, default `20` when absent or invalid.
- [x] 2.3 Refactor the user-function-call section (around lines ~351-363) to run `user_function(...)` inside a `concurrent.futures.ThreadPoolExecutor(max_workers=1)` worker, with `future.result(timeout=timeout_seconds)`. Catch `concurrent.futures.TimeoutError`.
- [x] 2.4 Pre-allocate `stdout_buf, stderr_buf = StringIO(), StringIO()` in the main thread; have the worker thread enter `with bind_capture_to(stdout_buf, stderr_buf):` (helper added in task 3.1) around `user_function(...)`. Remove the previous `with capture_output(): ...` block in this codepath; success and error branches read `getvalue()` from the same buffers.
- [x] 2.5 On `TimeoutError`, return `{"success": false, "error": f"Custom Python Block exceeded the configured timeout of {timeout_seconds}s", "error_type": MODAL_TIMEOUT_ERROR_TYPE, "stdout": stdout_buf.getvalue() or None, "stderr": stderr_buf.getvalue() or None}` and call `executor.shutdown(wait=False)`. Import `MODAL_TIMEOUT_ERROR_TYPE` from `constants.py`.
- [x] 2.6 Add an inline `# WATCHDOG NOTE` comment explaining: (a) Python cannot kill the worker thread, (b) the orphan continues consuming Modal compute until container recycle, (c) the @app.cls(timeout=130) decorator bounds a single invocation but does not fire for timed-out frames because the handler returned normally, (d) the value here is worker-advance + structured error + classifier seam — not bulletproof compute control.

## 3. Capture helper (worker-side library code)

- [x] 3.1 In `inference/core/workflows/execution_engine/v1/dynamic_blocks/error_utils.py`, add a new `@contextmanager bind_capture_to(stdout_buf: StringIO, stderr_buf: StringIO) -> Generator[None, None, None]` next to `capture_output`. It SHALL: call `_install_dispatchers()`, bind the caller's thread `_thread_local._capture_stdout` and `_capture_stderr` to the provided buffers, `yield`, and clear them on exit. Export from the module.
- [x] 3.2 Add a unit test for the bind_capture_to helper. _Placed at `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_error_utils_bind_capture.py` to match the existing dynamic-blocks test dir (the path in the original task was speculative; the real layout has tests under `tests/workflows/`)._  Covers: writes from the bound thread land in the provided buffers; writes from a concurrent unrelated thread do not; thread-local bindings cleared on `__exit__`; snapshot semantics safe while a still-running worker continues writing.

## 4. Env var & error class

- [x] 4.1 In `inference/core/env.py`, declare `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS: Optional[int]` near the other `MODAL_*` settings (around lines 705-723). Parse with the same idiom used for `MODEL_LOCK_ACQUIRE_TIMEOUT` (`float`/`int` cast inside a try, fall back to default on `ValueError`). Validate range `[1, 120]`; on out-of-range or unparseable, log a warning via `inference.core.logger.logger.warning(...)` and fall back to `None` (which downstream code maps to the 20s default). _Implementation note: used `warnings.warn` since `env.py` has no logger import; matches the existing warning idiom in this file._
- [x] 4.2 In `inference/core/workflows/errors.py`, add `class DynamicBlockTimeoutError(DynamicBlockCodeError): pass` (placed adjacent to `DynamicBlockCodeError`). Re-use the parent's `__init__` signature (it already carries `stdout`, `stderr`, `block_type_name`, `traceback_str`).

## 5. ModalExecutor (worker-side HTTP client)

- [x] 5.1 In `modal_executor.py`, update `ModalExecutor.__init__` signature to `(self, workspace_id: Optional[str] = None, custom_python_block_timeout_seconds: Optional[int] = None)`. Resolve precedence: constructor arg → env var (`CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS`) → `20`. Store as `self._timeout_seconds: int`.
- [x] 5.2 In `execute_remote`, compute `request_timeout = self._timeout_seconds + 10` and include `"timeout_seconds": self._timeout_seconds` in `request_payload`. Replace the hard-coded `timeout=30` at line 303 with `timeout=request_timeout`.
- [x] 5.3 After parsing the response at line 319, before the existing `result.get("success", False)` check, branch: if `result.get("error_type") == MODAL_TIMEOUT_ERROR_TYPE`, raise `DynamicBlockTimeoutError(...)`. Import the constant from `constants.py`.
- [x] 5.4 Restructure the existing `except requests.exceptions.RequestException` catch: catch `requests.exceptions.ReadTimeout` specifically → `DynamicBlockTimeoutError`. For remaining `RequestException` subtypes → `DynamicBlockCodeError` (option (a) from the original spec, reuses the existing execution-engine-side class). Also re-classed the non-200 status path from `DynamicBlockError` → `DynamicBlockCodeError`.
- [x] 5.5 In `validate_code_in_modal`, added a private `_timeout_override_seconds` parameter on `execute_remote` and pass `VALIDATION_TIMEOUT_SECONDS` (30) from `validate_code_in_modal`. Inline comment documents the why.

## 6. block_scaffolding wiring

- [x] 6.1 In `inference/core/workflows/execution_engine/v1/dynamic_blocks/block_scaffolding.py`, the call site `executor = ModalExecutor(workspace_id)` at line 92 needs no change — `ModalExecutor.__init__` reads the env var itself. Add a one-line comment noting the env-var fallback. Verify by reading the file end-to-end.

## 7. CLI surface

- [x] 7.1 In `inference_cli/lib/roboflow_cloud/batch_processing/entities.py`, add `custom_python_block_timeout_seconds: Optional[int] = Field(serialization_alias="customPythonBlockTimeoutSeconds", default=None, ge=1, le=120)` to `WorkflowsProcessingSpecificationV1`.
- [x] 7.2 In `inference_cli/lib/roboflow_cloud/batch_processing/api_operations.py`, add `custom_python_block_timeout_seconds: Optional[int] = None` parameter to both trigger functions; pass through into the `WorkflowsProcessingSpecificationV1(...)` constructor.
- [x] 7.3 In `inference_cli/lib/roboflow_cloud/batch_processing/core.py`, add a `--custom-python-block-timeout` typer option to `process_images_with_workflow` with `min=1, max=120`; forward to the API call.
- [x] 7.4 Repeat 7.3 for `process_videos_with_workflow`.

## 8. Tests

- [x] 8.1 Integration test class `TestModalCustomPythonBlockTimeout` appended to `tests/workflows/integration_tests/execution/test_workflow_with_custom_python_block_modal.py`, gated on `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. Covers: slow block succeeds within 60s budget; raises `DynamicBlockTimeoutError` with 5s budget; exception's `public_message` contains "5s"; exception's `stdout` contains the print output emitted before the timeout. _Not run locally — requires Modal credentials._
- [x] 8.2 ModalExecutor precedence tests in `tests/workflows/unit_tests/execution_engine/dynamic_blocs/test_modal_executor.py`: default, env wins, constructor wins, out-of-range falls back, non-int falls back.
- [x] 8.3 Mock-based tests in the same file: server timeout response → `DynamicBlockTimeoutError` with populated stdout; `ReadTimeout` → same typed exception; `ConnectionError` → `WorkflowExecutionEngineError` (NOT `WorkflowCompilerError`); payload includes `timeout_seconds`, client uses N+10.
- [x] 8.4 Error-hierarchy test in `tests/workflows/unit_tests/test_errors_hierarchy.py`. _Placed in `tests/workflows/` to match the actual test layout (the path in the original task was speculative)._
- [x] 8.5 CLI test in `tests/inference_cli/unit_tests/lib/roboflow_cloud/batch_processing/test_core.py`. Asserts `--custom-python-block-timeout 45` reaches the trigger function for both image+video commands; absence → `None`; out-of-range value rejected before reaching the API.
- [x] 8.6 Static-import audit test `test_wire_constant_audit.py` scans `inference/` and `modal/` for the literal `"CustomPythonBlockTimeout"` and asserts it appears only in `constants.py`.

## 9. Verification & rollout

- [x] 9.1 Ran the new unit tests locally: 22/22 pass across `test_errors_hierarchy.py`, `test_error_utils_bind_capture.py`, `test_modal_executor.py`, `test_wire_constant_audit.py`, `test_core.py` (CLI). The pre-existing `test_block_assembler.py` has an unrelated `ModuleNotFoundError: trackers` that is not in our diff.
- [ ] 9.2 **HUMAN FOLLOW-UP** — Deploy candidate `modal_app.py` to a staging Modal app name and `curl` the endpoint with `{"code_str": "def run(self): import time; time.sleep(30); return {}", "timeout_seconds": 5, ...}` to confirm the structured timeout response with `error_type=CustomPythonBlockTimeout` returns at ~5s.
- [ ] 9.3 **HUMAN FOLLOW-UP** — Confirm in staging that `timeout_seconds: 60` returns a structured response (not Modal's per-invocation kill), and that with the env var unset on the worker side the legacy 20s default applies. _Also gated on the Modal integration tests in `test_workflow_with_custom_python_block_modal.py::TestModalCustomPythonBlockTimeout`._
- [ ] 9.4 **HUMAN FOLLOW-UP** — Coordinate the rollout order with platform and orchestrator teams: Modal app deploy lands first, then the inference package release, then the SDK/CLI release. Document this on the rollout ticket linked to CS-239 Task 1.

## 10. Documentation

- [ ] 10.1 **HUMAN FOLLOW-UP** — Coordinate with CS-237 author: their hint copy at `state_report_processor.py:711-751` references "30s" which is incorrect (real deploy-time ceiling was 20s, now configurable). Either delete the literal value or template the configured `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS`. This lives in a different repo (batch-processing-services), so requires that owner.
- [x] 10.2 Added a "Per-frame timeout (Modal execution mode)" section to `docs/workflows/custom_python_code_blocks.md` describing the env var, CLI flag, default, max, the `DynamicBlockTimeoutError` raised on exceed, and the orphan-thread caveat in plain language.
