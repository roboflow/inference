## ADDED Requirements

### Requirement: Configurable per-frame timeout for Custom Python Blocks executed on Modal

The system SHALL allow the per-frame execution timeout for Custom Python Blocks running in Modal mode to be configured at runtime via the `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` environment variable, with a valid range of 1 through 120 seconds inclusive. When the variable is unset, the system SHALL apply a default of 20 seconds (preserving today's effective behaviour). When the variable is set to a value outside the valid range, the system SHALL reject the value and fall back to the default with a logged warning.

#### Scenario: Env var unset uses legacy default

- **WHEN** `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` is not present in the environment
- **THEN** `ModalExecutor` configures the in-handler watchdog deadline at 20 seconds and the HTTPS client read timeout at 30 seconds

#### Scenario: Env var set to a value within range

- **WHEN** `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` is set to `45`
- **THEN** `ModalExecutor` configures the in-handler watchdog deadline at 45 seconds and the HTTPS client read timeout at 55 seconds

#### Scenario: Env var set to a value outside the valid range

- **WHEN** `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` is set to `0`, `-5`, `121`, or `"abc"`
- **THEN** `ModalExecutor` logs a warning and configures the watchdog deadline at the 20-second default

### Requirement: In-handler watchdog returns structured timeout response

The Modal `execute_block` web endpoint SHALL accept an optional `timeout_seconds: int` field in its request payload. When present, the handler SHALL execute the user function on a worker thread and SHALL return a structured timeout response within the configured deadline if the worker has not completed. The response SHALL include `success: false`, `error_type: "CustomPythonBlockTimeout"`, a human-readable `error` message containing the configured timeout value, and `stdout` / `stderr` strings reflecting the user code's output up to the point of timeout (or `null` if empty). When the request omits `timeout_seconds`, the handler SHALL apply a 20-second default.

#### Scenario: User block exceeds configured timeout

- **WHEN** the request includes `timeout_seconds: 5` and the user function sleeps for 30 seconds while writing `print("step 1")` to stdout before the timeout fires
- **THEN** the handler returns a JSON response with `success: false`, `error_type: "CustomPythonBlockTimeout"`, `error` containing the string `"5s"`, and `stdout` containing `"step 1"`

#### Scenario: User block completes within configured timeout

- **WHEN** the request includes `timeout_seconds: 60` and the user function returns successfully after 2 seconds
- **THEN** the handler returns the normal success payload with `success: true` and the serialised result

#### Scenario: Request omits timeout_seconds

- **WHEN** the request does not include a `timeout_seconds` field
- **THEN** the handler uses a 20-second watchdog deadline

### Requirement: Stdout and stderr are captured across the watchdog thread boundary

The Modal handler SHALL preserve user-code stdout and stderr written before and up to the watchdog deadline, even when the timeout response is returned from the main thread while the user-code worker thread is still running. The capture mechanism SHALL bind a pair of caller-allocated `StringIO` buffers to the worker thread's per-thread capture targets via a `bind_capture_to(stdout_buf, stderr_buf)` helper in `error_utils.py`.

#### Scenario: Worker thread writes preserved on timeout

- **WHEN** a user block writes `"about to hang"` to stdout, then enters an infinite loop, and the watchdog fires
- **THEN** the timeout response's `stdout` field contains `"about to hang"`

#### Scenario: Helper binds the calling thread's capture targets

- **WHEN** a test thread enters `with bind_capture_to(buf_a, buf_b):` and writes to `sys.stdout` / `sys.stderr`
- **THEN** `buf_a.getvalue()` and `buf_b.getvalue()` reflect those writes after the `with` block exits
- **AND** writes from a concurrent unrelated thread do not appear in `buf_a` or `buf_b`

### Requirement: ModalExecutor raises a typed timeout error on watchdog-fired responses and on client read timeouts

`ModalExecutor.execute_remote` SHALL raise `DynamicBlockTimeoutError`, a subclass of `DynamicBlockCodeError`, when the Modal handler returns a response with `error_type == "CustomPythonBlockTimeout"`. As defence in depth, `ModalExecutor.execute_remote` SHALL also raise `DynamicBlockTimeoutError` when the underlying `requests.post` raises `requests.exceptions.ReadTimeout`. The raised exception's `public_message` SHALL include the configured timeout value and a hint mentioning the maximum cap (120s). The exception SHALL populate `stdout`, `stderr`, and `block_type_name` from the response payload when available.

#### Scenario: Server response indicates timeout

- **WHEN** the Modal handler returns `{"success": false, "error_type": "CustomPythonBlockTimeout", "error": "...", "stdout": "...", "stderr": null}`
- **THEN** `execute_remote` raises `DynamicBlockTimeoutError` with `public_message` containing the configured timeout value and `stdout` populated from the response

#### Scenario: Client read timeout fires before server responds

- **WHEN** `requests.post` raises `requests.exceptions.ReadTimeout`
- **THEN** `execute_remote` raises `DynamicBlockTimeoutError`

#### Scenario: Non-timeout HTTP failure

- **WHEN** `requests.post` raises any other `requests.exceptions.RequestException` (e.g., `ConnectionError`)
- **THEN** `execute_remote` raises an exception that is a subclass of `WorkflowExecutionEngineError`, not of `WorkflowCompilerError`

### Requirement: Wire constant is the single source of truth for the timeout error type string

The string `"CustomPythonBlockTimeout"` SHALL be defined as a module-level constant `MODAL_TIMEOUT_ERROR_TYPE` in `inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py`. Both `modal/modal_app.py` (when emitting the timeout response) and `inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py` (when matching the response) SHALL reference this constant. The literal string `"CustomPythonBlockTimeout"` SHALL not appear elsewhere in the inference codebase.

#### Scenario: Both sides import the same constant

- **WHEN** `modal_app.py` and `modal_executor.py` are inspected for the literal `"CustomPythonBlockTimeout"`
- **THEN** the literal appears in only one place — the definition of `MODAL_TIMEOUT_ERROR_TYPE` in `constants.py` — and both files import it

### Requirement: Validation codepath uses a fixed small timeout

`validate_code_in_modal` SHALL include `timeout_seconds: 30` in its request payload regardless of the value of `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS`. This ensures that increasing the per-frame budget does not also expand the syntax-check window.

#### Scenario: Validation request payload

- **WHEN** `validate_code_in_modal` builds a Modal request, with `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS=120` set in the environment
- **THEN** the request payload sent to Modal contains `timeout_seconds: 30`

### Requirement: Modal app deploy-time decorator allows up to 130 seconds per invocation

The `@app.cls(...)` decorator in `modal/modal_app.py` SHALL set `timeout=130` so that the in-handler watchdog (capped at 120 seconds) plus a 10-second client/network headroom always wins the race against Modal's per-invocation hard kill.

#### Scenario: Deploy-time configuration

- **WHEN** `modal/modal_app.py` is inspected
- **THEN** the `Executor` class is decorated with `@app.cls(..., timeout=130, ...)`

### Requirement: CLI exposes the timeout flag on batch-workflow image and video commands

The `inference rf-cloud batch-processing` CLI SHALL expose a `--custom-python-block-timeout` integer option on both `process-images-with-workflow` and `process-videos-with-workflow` commands. The option SHALL be validated as an integer in the range 1 through 120 inclusive. When omitted, the field SHALL not be included in the trigger payload (allowing the server to apply its default). When supplied, the value SHALL be threaded into the `processingSpecification.customPythonBlockTimeoutSeconds` field of the trigger payload.

#### Scenario: Valid timeout value passed via CLI

- **WHEN** the user runs `inference rf-cloud batch-processing process-images-with-workflow ... --custom-python-block-timeout 60`
- **THEN** the trigger request payload sent to the Roboflow API contains `processingSpecification.customPythonBlockTimeoutSeconds == 60`

#### Scenario: Out-of-range value rejected by CLI

- **WHEN** the user runs the same command with `--custom-python-block-timeout 200`
- **THEN** the CLI exits with a non-zero status and emits a validation error mentioning the 1–120 range

#### Scenario: Flag omitted leaves the field absent from the payload

- **WHEN** the user runs the command without `--custom-python-block-timeout`
- **THEN** the trigger request payload does not include `customPythonBlockTimeoutSeconds` in `processingSpecification`

### Requirement: Trigger payload model carries the timeout under processingSpecification

`WorkflowsProcessingSpecificationV1` in `inference_cli/lib/roboflow_cloud/batch_processing/entities.py` SHALL declare an optional integer field `custom_python_block_timeout_seconds` with `serialization_alias="customPythonBlockTimeoutSeconds"`. The trigger payload builders `trigger_job_with_workflows_images_processing` and `trigger_job_with_workflows_videos_processing` in `api_operations.py` SHALL accept the value as a parameter and pass it to the constructed `WorkflowsProcessingSpecificationV1`.

#### Scenario: Pydantic serialisation uses camelCase alias

- **WHEN** a `WorkflowsProcessingSpecificationV1(custom_python_block_timeout_seconds=45, ...)` instance is serialised with `model_dump(by_alias=True)`
- **THEN** the output contains the key `"customPythonBlockTimeoutSeconds": 45`

#### Scenario: Trigger function plumbs the value through

- **WHEN** `trigger_job_with_workflows_images_processing(custom_python_block_timeout=90, ...)` is called
- **THEN** the constructed `WorkflowsProcessingSpecificationV1` has `custom_python_block_timeout_seconds == 90`
