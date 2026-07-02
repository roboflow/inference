---
name: review-topic-test-hygiene
description: Load for EVERY PR that changes runtime behavior — verify the changed behavior is covered by a real (non-mock-only) test that actually runs in CI, that tests are isolated (no leaking monkeypatch/module stubs), and that runtime SELECTORS/branches are exercised (not just one literal parameter). Trigger on new/edited code under inference/, inference_models/, inference_sdk/, inference_cli/ with no corresponding real test, on new @pytest.mark.skipif/skip or env-gated tests, on new global monkeypatch/sys.modules stubs, and on GPU/slow tests whose CI runner is unclear.
---

# Review topic: Test hygiene & coverage of changed behavior

## When this applies
This topic is ALWAYS in play — every behavior-changing PR must be checked for coverage. Load and apply it whenever the diff shows any of:
- New or modified runtime code (a block, model I/O contract, endpoint, SDK/CLI path, execution-engine rule) with **no new/changed test**, or only a test that mocks out the very thing that changed.
- A new/edited dispatch on a runtime **selector**: enum→lambda maps (e.g. `PROPERTIES_EXTRACTORS`), `if kind == ...`, block `type`/`property_name` switches, `get_execution_engine_compatibility`. Adding a branch without a test that drives *that* branch by its runtime value is the core signal.
- New `@pytest.mark.skipif(...)`, `@pytest.mark.skip`, `@pytest.mark.slow`, or an env-gated test (`os.getenv("SKIP_*_TESTS", True)`, `RUN_*`) — confirm a CI workflow actually flips the gate.
- New global test state: module-level `monkeypatch` outside a fixture, `sys.modules[...] = stub`, `os.environ[...] =` at import time, `autouse` fixtures that mutate shared singletons/registries.
- GPU/hardware-only tests — confirm which `.github/workflows/*.yml` runs them.

## What to protect
- **Changed behavior is verified by a test that runs in the PR's CI.** A test that is skipped-by-default (env gate defaults to skip) or only lives on the self-hosted GPU runner does NOT protect a change merged through x86 CI. Failure mode: green PR, silent regression shipped.
- **Selectors, not just literals.** Blocks/UQL dispatch on runtime values; a test that passes one hardcoded literal leaves the added branch and the whole selector path unverified. Failure mode: new enum value crashes at runtime while tests are green.
- **Test isolation.** A leaked monkeypatch, stubbed `sys.modules`, or mutated global registry corrupts *other* tests → order-dependent flakiness and false greens. Failure mode: passes alone, fails (or hides bugs) in the suite.
- **Real coverage over mock theater.** A test whose mocks reproduce the exact code under change asserts nothing about the change.

## What to check
1. **Map the change → a test that runs.** For each behavior in the diff, find the covering test AND the CI workflow that executes it: unit workflows → `unit_tests_workflows_x86.yml` (`tests/workflows/unit_tests`); workflow execution → `integration_tests_workflows_x86.yml`; inference_models I/O → `integration_tests_inference_models.yml` (`tests/inference/models_predictions_tests`); HTTP server → `integration_tests_inference_server_x86.yml`; SDK → `unit_tests_inference_sdk_x86.yml`; CLI → `unit_tests_inference_cli_x86.yml`. If the only test is GPU-gated, it runs *only* on `test.nvidia_t4.yml`/`test.jetson_*` — not on PR x86 CI.
2. **Selector coverage.** For every branch/enum value/`property_name`/`kind` added, is there a test that supplies *that* runtime value and asserts its output? One test per new branch is the bar (see PR #1129).
3. **Isolation.** Monkeypatching must go through the function-scoped `monkeypatch` fixture (auto-undone), not module-level assignment. Module stubs belong in a registered `stub_plugins/` package, not `sys.modules` hacks. `autouse` fixtures must restore any shared global.
4. **Gate direction.** For any new `skipif`/env gate, grep the workflows for the var — confirm a job sets it so the test actually runs (`SKIP_SAM2_TESTS=False`, `SKIP_LMM_TEST`, etc.). A gate that defaults to skip and is never flipped = dead test.
5. **Marker correctness.** New marker used → declared in `pytest.ini` (only `slow`, `workflows` are registered). Unregistered markers silently no-op the intended deselection (PR #1724).
6. **Real vs mock.** The assertion must observe the changed output, not a mock return value the test itself set.
7. **Assertion stability.** Model-prediction tests should assert shape/contract/tolerances, not brittle exact floats on unstable ops (PRs #2026, #1646, #1394).

## Common failure modes
- **Selector added, only a literal tested** — new enum branch merged with a single-value test; the branch is unverified. Correct pattern: PR #1129 added CENTER/TOP_LEFT/TOP_RIGHT/BOTTOM_LEFT/BOTTOM_RIGHT to `PROPERTIES_EXTRACTORS` *and* one behavior test per new `property_name`.
- **Test never runs in the merging CI** — GPU/env-gated test defaults to skip (`os.getenv("SKIP_SAM2_TESTS", True)` in `tests/inference/integration_tests/test_sam2.py`) and is only flipped `False` on the self-hosted `test.nvidia_t4.yml` runner; a PR going through x86 CI merges with it skipped. Confirm the runner.
- **Unregistered marker** — PR #1724 had to add the `workflows` marker to `pytest.ini`; before that, `-m "not workflows"` deselection silently did nothing.
- **Test broken by an unrelated refactor** — PR #2526 fixed `test_workflow_model_handler` after `WorkflowRunner` moved args from `run_workflow(...)` to the constructor; the test still referenced the old signature. Watch for tests that were not updated alongside the API they cover.
- **Broken/missing test assets** — PR #2168 fixed a non-existent image reference in the gaze integration test (would error, not test). PR #1539 updated a stale stub after a dependency move.
- **Flaky exact-value assertions** — PRs #2026, #1646, #1100, #1044, #1033 loosened brittle prediction assertions; over-tight asserts on numerically unstable ops cause false reds and get muted, eroding coverage.

## Example implementations (point here)
- `tests/workflows/unit_tests/core_steps/formatters/test_property_extraction.py` — canonical **selector coverage**: one `test_property_extraction_block_with_<anchor>` per runtime `property_name`, driven through the block, asserting real output. Established by PR #1129.
- `tests/workflows/integration_tests/execution/test_workflow_top_prediction.py` — **behavior-level execution test**: full workflow compiled + run with `$inputs.*`/`$steps.*` selectors, covering the non-empty and empty-detections paths. The model for "exercise the runtime wiring, not just a function."
- `tests/workflows/integration_tests/execution/stub_plugins/` — **isolated module stubs done right**: fake blocks live in registered plugin packages loaded per-test, instead of `sys.modules` monkeypatching that leaks.
- `tests/inference/unit_tests/conftest.py` (`_patch_aioresponses_stream_writer`, `autouse=True` + `monkeypatch`) — **scoped, auto-undone patching** via the fixture; contrast with module-level global assignment.
- `tests/inference/integration_tests/test_sam2.py` (`@pytest.mark.skipif(bool_env(os.getenv("SKIP_SAM2_TESTS", True)))`) + `.github/workflows/test.nvidia_t4.yml` (sets `SKIP_SAM2_TESTS=False`) — the **gate-plus-runner pair**; use it to check any new env-gated test has a CI job that flips it.
- `pytest.ini` — the only registered markers (`slow`, `workflows`); any new marker in the diff must be added here (PR #1724).

## Severity guidance
- **Critical** — changed runtime behavior ships with no test that runs in the PR's CI, OR a bugfix lands with no regression test reproducing the bug (regression can silently return).
- **High** — a new selector/branch/enum value is untested (only literals covered); a new test is GPU/env-gated but no CI job flips the gate (dead test giving false confidence); a global monkeypatch/`sys.modules` stub leaks across tests.
- **Medium** — mock-only test that asserts the mock rather than the change; unregistered marker; brittle exact-value assertion on an unstable op; broken/stale test asset or stub; test not updated alongside the API it covers.
