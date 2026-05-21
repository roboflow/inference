## Context

Custom Python Blocks in Workflows currently execute on Modal via a FastAPI web endpoint deployed from `modal/modal_app.py`. The worker (running inside the batch-processing data processor container, or anywhere `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal`) calls into Modal over HTTPS via `ModalExecutor.execute_remote()` in `inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py`.

Two hard-coded timeouts live in this path:

| Layer | File:line (today) | Value | Effect |
|---|---|---|---|
| Modal container | `modal/modal_app.py:64` | `@app.cls(timeout=20)` | Modal kills the container handler at 20s. |
| HTTPS client | `modal_executor.py:303` | `requests.post(..., timeout=30)` | Client gives up after 30s. |

When the 20s Modal kill fires, the client sees a connection error and the worker raises a generic `DynamicBlockError("Failed to connect to Modal endpoint...")`. The user gets no actionable error, no diagnostic stdout/stderr, and no way to extend the budget for legitimately-slow blocks. Modal's `@modal.fastapi_endpoint` decorator additionally **does not honour** `Cls.with_options(timeout=...)` per-call overrides — the only way to give an individual invocation more time without redeploying the app is to run a soft watchdog *inside the handler* and have the handler return a structured response before Modal kills it.

The CS-237 per-frame failure classifier (a sibling initiative) needs a stable seam to recognise timeouts and emit useful hint copy, but today there is none — `DynamicBlockError` with a generic message is indistinguishable from network failures and Modal outages.

Three repos cooperate to set per-frame budgets on batch jobs: `inference` (this change), `batch-processing-services` (separate), and the `roboflow` monorepo UI (separate). This change scopes only the `inference` slice.

## Goals / Non-Goals

**Goals:**
- Make the per-frame Custom Python Block execution budget configurable in the range 1–120 seconds, defaulting to 20s (the current effective behaviour).
- Surface a stable, classifier-friendly error class (`DynamicBlockTimeoutError`) and wire string (`"CustomPythonBlockTimeout"`) when the budget is exceeded.
- Preserve stdout/stderr captured up to the moment of timeout in the error payload so users have diagnostic context.
- Let the worker thread (`workflows-data-processor`) advance to the next frame at the user-chosen deadline, instead of waiting up to the deployed Modal decorator ceiling.
- Fix the pre-existing categorisation bug where runtime HTTP failures raise `DynamicBlockError` (a `WorkflowCompilerError` subclass) instead of an execution-engine-side error.

**Non-Goals:**
- Bulletproof Modal compute control. The watchdog is soft — Python cannot kill threads, and `@modal.fastapi_endpoint` blocks per-call timeout overrides. Orphan compute is acknowledged and accepted; eliminating it requires migrating to Modal RPC or subprocess isolation (tracked separately).
- Local-mode (`WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=local`) timeout enforcement. In-process custom blocks run with no soft cutoff today; that's a separate problem.
- Per-block-type or per-step timeouts. One value per workflow run.
- Workflow init-parameter plumbing for the timeout. The orchestrator already sets a per-container env var; no additional indirection earns its keep at this scope.
- The orchestrator (`batch-processing-services`) and platform UI (`roboflow` monorepo) work from the broader CS-239 proposal.

## Decisions

### Decision 1: Env var only, no workflow init-parameter plumbing

**Choice:** `ModalExecutor.__init__` reads `CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS` directly (with optional constructor override for tests). Nothing flows through workflow init parameters.

**Why over alternatives:**
- *Init-parameter threading* (block constructor → `self._timeout` → `run()` → executor): the existing `block_scaffolding.py:92` constructs a fresh `ModalExecutor` on every block invocation, so any init-time plumbing would either need restructuring or be redundant with reading the env var at construction time. Adds 3+ files of plumbing for a value that's already in the container's environment.
- *Per-call kwarg on `execute_remote(timeout_seconds=N)`*: cleaner than init-param plumbing, but for the current scope (batch-only) every call wants the same value. The env var is the simplest source of truth. Future extension to per-call control remains trivial if needed.

**Trade-off:** Interactive workflows running in `modal` mode (the inference server) will inherit whatever env var is set, which may be 20 (default). They cannot set a per-execution override without this change being extended. Acceptable — interactive use is out of scope for CS-239 Task 1.

### Decision 2: `DynamicBlockTimeoutError` extends `DynamicBlockCodeError`, not `DynamicBlockError`

**Choice:** New exception class lives under `WorkflowExecutionEngineError` (via `DynamicBlockCodeError`), not under `WorkflowCompilerError` (where today's `DynamicBlockError` sits).

**Why:** A per-frame execution timeout is a runtime event, not a compile-time event. The existing `DynamicBlockError` is the wrong parent for runtime failures — and the existing `RequestException` catch at `modal_executor.py:369-373` already raises `DynamicBlockError` for what are really runtime HTTP failures. Re-classing both the new timeout path **and** the generic HTTP-failure path to the execution-engine-side fixes a latent categorisation bug. `DynamicBlockCodeError` already carries the fields we want — `stdout`, `stderr`, `block_type_name`, `traceback_str` — so the timeout exception inherits a useful structured payload.

**Trade-off:** External callers who `except DynamicBlockError` on the assumption it catches Modal failures will no longer catch HTTP/timeout failures. Risk is low (this is internal to inference) but worth noting.

### Decision 3: Threading watchdog inside the Modal handler, not signal-based

**Choice:** Wrap user-code execution in a `ThreadPoolExecutor` with `.result(timeout=N)`. On `FutureTimeoutError`, return a structured response immediately and `shutdown(wait=False)` the executor.

**Why:** `signal.alarm()` only works on the main thread, and Modal FastAPI endpoints run inside uvicorn's worker thread pool. Threading is the only mechanism that works in this context. Python cannot terminate the running thread — accepted as a known limitation (Decision 4).

**Why not switch to `asyncio` + `asyncio.wait_for`:** Same fundamental problem — cancellation in Python doesn't reliably kill running CPU-bound code, and the existing handler is sync. Larger refactor without benefit at this scope.

### Decision 4: Acknowledge and accept orphan-thread compute waste

**Choice:** When the watchdog fires, the worker thread continues running user code on the Modal container. Modal's per-invocation `@app.cls(timeout=130)` decorator bounds a single invocation of the handler, but the handler has already returned normally at that point — so the decorator never fires for timed-out frames. The orphan thread continues until user code finishes naturally or Modal recycles the container (scaledown, deploy, lifecycle).

**Why:** Modal's `@modal.fastapi_endpoint` does not honour `Cls.with_options(timeout=...)` per-call. The only "real fix" is migrating to Modal RPC or subprocess isolation inside the handler — both larger refactors. We document this prominently in code and in this design.

**Trade-off:** Pathological user code (infinite loops, very long network calls) keeps consuming billable Modal compute past the user's deadline. The watchdog still delivers value: (1) worker advances to next frame at N seconds, (2) client gets a structured error with stdout/stderr, (3) classifier has a stable match seam. Tracked future work: RPC migration.

### Decision 5: Three-layer timeout (N watchdog, N+10 client, 130 decorator)

**Choice:**

| Layer | Value | Why |
|---|---|---|
| In-handler watchdog | `N` (configured, 1–120) | Returns structured response at exactly the user's deadline. |
| HTTPS client | `N + 10` | 10s headroom so the watchdog's response can reach the client before the client gives up. |
| Modal decorator | `130` (constant, set at deploy) | Bounds a single handler invocation. The watchdog (≤120) plus 10s of headroom always finishes before Modal's hard kill would fire. |

**Why 10s:** Generalisation of today's 30 − 20 = 10s gap; empirically sufficient for the network round-trip plus serialisation. The watchdog's response is tiny (no large payloads); a sub-second margin would also work but 10s adds safety against transient network slowness without harming user-perceived latency (the response usually arrives in <1s; the `requests.post` timeout is a *max* wait, not a min).

**Why 130, not exactly 120:** 10s gives the watchdog room to win the race even under cold-start or temporary stalls. With memory snapshots cold starts are <1s, but a redeployed image (any `INFERENCE_VERSION` change) can push it higher.

### Decision 6: `bind_capture_to(stdout_buf, stderr_buf)` helper for cross-thread capture

**Choice:** Add a new context manager in `error_utils.py` that takes pre-allocated `StringIO` buffers and binds them as the current thread's `threading.local` capture targets. Pre-allocate buffers in the main thread; the worker thread enters `bind_capture_to(...)` around user-code execution.

**Why:** `capture_output()` allocates its own buffers and binds them to whichever thread calls it. The existing implementation is per-thread safe via `_ThreadDispatchStream` + `threading.local`, but the buffers' lifetime is bound to the thread that opened the `with`. In a watchdog scenario the main thread owns the timeout/response path while the worker thread does the writes — they need to share buffers.

```python
# Main thread:
stdout_buf, stderr_buf = StringIO(), StringIO()
def _run_user_code():
    with bind_capture_to(stdout_buf, stderr_buf):   # binds on WORKER thread
        return user_function(**inputs)
future = executor.submit(_run_user_code)
try:
    result = future.result(timeout=N)
except FutureTimeoutError:
    # safe to read getvalue() — CPython GIL makes concurrent write+read atomic
    return {"stdout": stdout_buf.getvalue(), ...}
```

**Why not reuse `capture_output` and pass buffers out:** would require restructuring the helper's API and breaks the contract that `_thread_local` resets on `__exit__`. A separate `bind_capture_to(buf, buf)` for caller-provided buffers is cleaner and additive.

**StringIO concurrency:** `StringIO.write` and `getvalue` are individually atomic under the GIL in CPython. The orphan worker may continue appending after the main thread reads `getvalue()` — we get a "snapshot at timeout" semantic, which is exactly what we want.

### Decision 7: Single named constant for the wire string

**Choice:** `MODAL_TIMEOUT_ERROR_TYPE = "CustomPythonBlockTimeout"` lives in a new `inference/core/workflows/execution_engine/v1/dynamic_blocks/constants.py`. Both `modal_app.py` (returning the value) and `modal_executor.py` (parsing the response) import it.

**Why:** The wire string becomes a public contract with the CS-237 classifier. Defining it in one place enforces consistency and discoverability. The Python class name (`DynamicBlockTimeoutError`) is an internal artifact and may evolve; the wire string is the durable contract.

### Decision 8: Validation gets a fixed small `timeout_seconds`

**Choice:** `validate_code_in_modal` passes an explicit `timeout_seconds=30` (or similar small constant) in its request payload, regardless of the configured env var.

**Why:** Validation runs `compile() + ast.parse()` inside the Modal container — milliseconds in practice. Inheriting a 120s budget would let pathological future validation logic run unbounded for the user's configured budget. A fixed small value documents intent and is defence-in-depth.

**Trade-off:** Adds one explicit `timeout_seconds` argument to the validation-time payload, slightly diverging from the env-var-only model elsewhere. Worth the clarity.

### Decision 9: Re-class `RequestException` catch as execution-engine-side

**Choice:** The existing catch at `modal_executor.py:369-373` raises `DynamicBlockError` (compiler-side). Re-class to a `WorkflowExecutionEngineError`-side exception. The simplest path is to raise `DynamicBlockTimeoutError` if the underlying exception is `requests.exceptions.ReadTimeout` (defence in depth — the server-side watchdog should have responded first), and a new `DynamicBlockHTTPError(DynamicBlockCodeError)` (or reuse `DynamicBlockCodeError` directly) otherwise.

**Why:** Connection failures and reads/timeouts at the HTTP layer are runtime, not compile-time. Same justification as Decision 2.

## Risks / Trade-offs

- **[Risk]** Pathological user code keeps Modal compute (billable) running past the configured timeout until container recycle. **Mitigation:** Document prominently in code and proposal. Long-term fix is RPC migration (tracked separately).

- **[Risk]** Deploying worker changes (which start requesting `timeout_seconds > 20`) before the Modal app redeploy lands silently regresses to a 20s decorator kill — the watchdog never sees the request. **Mitigation:** Tasks file pins the deploy ordering (Modal first, worker second, SDK/CLI third) and tests for the worker include an assertion that the response carries `error_type=CustomPythonBlockTimeout` rather than a generic 5xx.

- **[Risk]** A redeployed Modal image (any `INFERENCE_VERSION` change) eats into the 10s headroom via cold-start. **Mitigation:** Memory snapshots already reduce cold starts to <1s on warm images. If a deploy pushes cold start above ~10s, the watchdog can lose the race against Modal's 130s decorator. Acceptable risk for the rollout window; revisit if observed in practice.

- **[Risk]** Pre-existing callers `except DynamicBlockError:` will stop catching Modal HTTP/timeout failures once those are re-classed to the execution-engine side. **Mitigation:** Grep audit at implementation time; the public surface is internal to inference, so the blast radius is contained.

- **[Risk]** `StringIO.getvalue()` from main thread while worker is appending. **Mitigation:** CPython GIL makes write/getvalue individually atomic; "snapshot at timeout" semantics are acceptable. Documented in code comments.

- **[Trade-off]** Env-var-only means no per-execution override from the inference-server interactive mode. Acceptable for batch-scope; extension path is documented.

- **[Trade-off]** 120s upper bound is dictated by Modal's per-invocation ceiling (130 with 10s headroom), not by a principled product limit. Users with legitimately longer blocks are blocked until RPC migration.

## Migration Plan

Three independent code changes, one deploy sequence.

**Deploy ordering (strict):**

1. **Modal app redeploy** with `@app.cls(timeout=130)` and the in-handler watchdog. Verify against a manual `curl` to the deployed endpoint that a request with `timeout_seconds=5` against a deliberately-slow block returns the structured response with `error_type=CustomPythonBlockTimeout`.
2. **Worker-side inference package release** (`modal_executor.py`, env var, error class, `bind_capture_to`). Default behaviour unchanged when env var unset.
3. **SDK/CLI release** with `--custom-python-block-timeout` flag.

**Rollback strategy:**
- Modal app: redeploy previous version of `modal_app.py` (rolls decorator back to 20, removes watchdog).
- Worker: previous inference release; env var unset → falls back to the legacy 20s default. Forward-compatible with the new Modal app (it ignores `timeout_seconds` if absent — default 20).
- SDK/CLI: previous release. Field omitted from payload → server applies default.

**Backward compatibility:**
- `customPythonBlockTimeoutSeconds` is optional throughout the chain. Older clients continue to work unchanged.
- New Modal handler accepts requests with or without `timeout_seconds`. When absent, applies a 20s default — matching legacy behaviour.

## Open Questions

- **Validation timeout value:** is 30s appropriate, or should it be tighter (e.g., 10s)? Compile/parse should complete in well under 1s; tightening reduces the orphan-thread risk for validation. Recommendation: 30s for the initial implementation, revisit if observed validation times are consistently sub-second.
- **Should `DynamicBlockTimeoutError` also carry the orphan-thread caveat in its public message?** Pro: user understands their compute keeps running. Con: clutters the error and most users don't think in those terms. Recommendation: no; document in the workflow execution docs, not the runtime error.
- **Should the watchdog `executor` be reused across requests (one per container) instead of created per call?** Probably yes for efficiency, but it adds shutdown-on-recycle complexity. Defer until a perf concern surfaces.
