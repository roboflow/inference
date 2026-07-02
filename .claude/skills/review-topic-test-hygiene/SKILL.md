---
name: review-topic-test-hygiene
description: Load when a diff shows — (1) new/changed runtime code under inference/, inference_models/, inference_sdk/, inference_cli/ with no companion test (or a test that mocks the changed thing); (2) a new @pytest.mark.skipif/skip/slow or env-gated test (SKIP_*_TESTS, RUN_*); (3) a new module-level monkeypatch, sys.modules[...] stub, or autouse fixture mutating a shared singleton/registry; (4) a GPU-only test whose CI runner is unclear. Greppable: PROPERTIES_EXTRACTORS, sys.modules[, pytest.ini.
---

# Review topic: Test hygiene & coverage of changed behavior

## When this applies
Load and apply this skill whenever the diff shows any of:
- New/modified runtime code (block, model I/O contract, endpoint, SDK/CLI path, execution-engine rule) with **no new/changed test**, or only a test that mocks out the very thing that changed.
- A new/edited dispatch on a runtime **selector**: enum→lambda maps (e.g. `PROPERTIES_EXTRACTORS`), `if kind == ...`, block `type`/`property_name` switches, `get_execution_engine_compatibility`.
- A new `@pytest.mark.skipif`/`skip`/`slow` or env-gated test (`os.getenv("SKIP_*_TESTS", True)`, `RUN_*`).
- New global test state: module-level `monkeypatch`, `sys.modules[...] = stub`, `os.environ[...] =` at import time, or `autouse` fixtures mutating shared singletons/registries.
- A GPU/hardware-only test whose CI runner is unclear.

## Review checklist
Tag each finding BLOCK / FLAG / NIT. See **Standards** below for the one canonical statement of each rule.

- **BLOCK** — Changed runtime behavior ships with no test that actually runs in the PR's x86 CI (Standard 1), OR a bugfix lands with no regression test reproducing the bug.
- **BLOCK** — A new env-gated / GPU test's gate is never flipped `False` by any CI job → dead test giving false confidence (Standard 4).
- **BLOCK** — A leaked module-level `monkeypatch`, `sys.modules` stub, or mutated global registry corrupts other tests (Standard 3).
- **FLAG** — A new selector/branch/enum value is exercised only by literals, not by a test driving *that* runtime value (Standard 2).
- **FLAG** — Mock-only test that asserts the mock's own return value rather than the changed output (Standard 6).
- **FLAG** — New marker used in the diff but not declared in `pytest.ini` → silent no-op deselection (Standard 5).
- **NIT** — Brittle exact-value assertion on a numerically unstable op (Standard 7).
- **NIT** — Broken/stale test asset, stub, or a test not updated alongside the API it covers.

### Not blocking
- Do NOT demand a test that duplicates an existing behavior test at a lower layer (e.g. a unit test when the same path is already driven end-to-end by an integration test that runs in CI).
- Do NOT demand exact-float assertions on model predictions — shape/contract/tolerance is the correct bar; over-tight asserts get muted and erode coverage.
- Do NOT block a GPU/hardware test for running only on the self-hosted runner *if* the same behavior also has an x86-CI test; the GPU test is additive.
- Pure refactors with no behavior change need no new test, only that existing tests still pass and were updated to the new signature.

## Standards

1. **Changed behavior is verified by a test that runs in the PR's CI.** Map each behavior in the diff to the covering test AND the CI workflow that executes it:
   - unit workflows → `unit_tests_workflows_x86.yml` (`tests/workflows/unit_tests`)
   - workflow execution → `integration_tests_workflows_x86.yml`
   - `inference_models` I/O → `integration_tests_inference_models.yml` (`tests/inference/models_predictions_tests`)
   - HTTP server → `integration_tests_inference_server_x86.yml` (`tests/inference/integration_tests`)
   - SDK → `unit_tests_inference_sdk_x86.yml` (`tests/inference_sdk/unit_tests`)
   - CLI → `unit_tests_inference_cli_x86.yml` (`tests/inference_cli/unit_tests`)

   A test that is skipped-by-default or lives only on the self-hosted GPU runner (`test.nvidia_t4.yml`, `test.jetson_*.yml`) does NOT protect a change merged through x86 CI.

2. **Selectors, not just literals.** Blocks/UQL dispatch on runtime values (`PROPERTIES_EXTRACTORS` in `inference/core/workflows/core_steps/common/query_language/operations/detections/base.py`). For every branch / enum value / `property_name` / `kind` added, a test must supply *that* runtime value through the block and assert its output. One test per new branch is the bar (PR #1129 added the anchor `property_name`s plus one behavior test each).

3. **Test isolation.** Monkeypatching goes through the function-scoped `monkeypatch` fixture (auto-undone), never module-level assignment. Module stubs belong in a registered `stub_plugins/` package, not `sys.modules` hacks. `autouse` fixtures must restore any shared global they touch.

4. **Gate direction.** For any new `skipif`/env gate, grep the workflows for the var and confirm a job sets it so the test actually runs (e.g. `SKIP_SAM2_TESTS=False` in `test.nvidia_t4.yml`, `SKIP_LMM_TEST` in `integration_tests_inference_server_x86.yml`). A gate that defaults to skip and is never flipped is a dead test.

5. **Marker correctness.** Any new marker used in the diff must be declared in `pytest.ini` — only `slow` and `workflows` are registered. Unregistered markers silently no-op the intended deselection (PR #1724 had to add `workflows`).

6. **Real coverage over mock theater.** The assertion must observe the changed output, not a mock return value the test itself set. A test whose mocks reproduce the exact code under change asserts nothing about the change.

7. **Assertion stability.** Model-prediction tests assert shape/contract/tolerances, not brittle exact floats on numerically unstable ops (PRs #2026, #1646, #1394).

## Example implementations (point authors here)
- `tests/workflows/unit_tests/core_steps/formatters/test_property_extraction.py` — canonical **selector coverage**: one `test_property_extraction_block_with_<anchor>` per runtime `property_name`, driven through the block, asserting real output (PR #1129).
- `tests/workflows/integration_tests/execution/test_workflow_top_prediction.py` — **behavior-level execution test**: full workflow compiled + run with `$inputs.*`/`$steps.*` selectors, covering non-empty and empty-detections paths.
- `tests/workflows/integration_tests/execution/stub_plugins/` — **isolated module stubs done right**: fake blocks in registered plugin packages loaded per-test, not `sys.modules` monkeypatching.
- `tests/inference/unit_tests/conftest.py` (`_patch_aioresponses_stream_writer`, `autouse=True` + `monkeypatch`) — **scoped, auto-undone patching**; contrast with module-level global assignment.
- `tests/inference/integration_tests/test_sam2.py` (`SKIP_SAM2_TESTS` gate) + `.github/workflows/test.nvidia_t4.yml` (sets `SKIP_SAM2_TESTS=False`) — the **gate-plus-runner pair** for checking any new env-gated test has a CI job that flips it.
- `pytest.ini` — the only registered markers (`slow`, `workflows`); any new marker must be added here.

## Reference PRs
- #1129 — selector coverage established (one behavior test per new `property_name`).
- #1724 — added the `workflows` marker to `pytest.ini`.
- #2526 — fixed a test left on an old `run_workflow(...)` signature after args moved to the `WorkflowRunner` constructor.
- #2168, #1539 — broken/missing test asset and stale stub after a dependency move.
- #2026, #1646, #1394, #1100, #1044, #1033 — loosened brittle exact-value prediction assertions.
