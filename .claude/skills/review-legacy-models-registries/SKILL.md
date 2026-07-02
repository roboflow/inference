---
name: review-legacy-models-registries
description: Review guidance for PRs touching inference/models/**, inference/core/models/** and inference/core/registries/** — enforces the legacy (ORT/HF) model implementations, the base inference contract (preprocess/predict/postprocess/make_response), and the Roboflow model-type registry / model-id resolution standards, including required version bumps, adapter parity, and thread-safety/companion requirements.
---

# Reviewing legacy-models-registries changes

## Scope
Trigger this skill when a PR changes any of:
- `inference/models/**` — concrete legacy model implementations (rfdetr, owlv2, sam/sam2/sam3, clip, gaze, yolo*, easy_ocr, doctr, florence2, paligemma, qwen*, etc.), `inference/models/utils.py` (the `ROBOFLOW_MODEL_TYPES` registry + `get_roboflow_model`), and `inference/models/aliases.py`.
- `inference/core/models/**` — base contracts (`base.py`, `roboflow.py`), task bases (`object_detection_base.py`, `instance_segmentation_base.py`, `keypoints_detection_base.py`, `classification_base.py`, `semantic_segmentation_base.py`), `stubs.py`, and `inference_models_adapters.py` (the `USE_INFERENCE_MODELS` bridge adapters).
- `inference/core/registries/**` — `base.py` (`ModelRegistry`) and `roboflow.py` (`get_model_type`, auth checks, metadata cache).

OUT of scope (defer to other skills): Workflows blocks, HTTP API handlers (`inference/core/interfaces/**`), model managers (`inference/core/managers/**`), the standalone `inference_models` package itself, and the tensor-data-representation `_tensor` sibling effort.

## What this surface is
Two layers a reviewer must keep coherent:
1. **Model implementations + base contract.** Every model is a `Model` (`inference/core/models/base.py`) whose `infer()` runs `preprocess → predict → postprocess` under telemetry spans, and whose `infer_from_request()` sets `response.time`, `inference_id`, and optional `visualization`. `@usage_collector("model")` wraps `BaseInference.infer`; models overriding `infer_from_request` (SAM2/SAM3) must re-apply the decorator (#1768). ONNX Roboflow models flow through `inference/core/models/roboflow.py` (artifact download → cache → `environment.json` parse → class names/colors/preproc).
2. **Registry / id resolution.** `inference/core/registries/roboflow.py::get_model_type(model_id, api_key)` returns `(task_type, model_type)` by: resolving aliases, checking local `inference_models` dirs, `GENERIC_MODELS`, stub versions, then the Roboflow API; the `(task_type, model_type)` tuple keys into `ROBOFLOW_MODEL_TYPES` in `inference/models/utils.py` to pick a class.

CONTRACTS/INVARIANTS to protect:
- `get_model_type` must return a non-null `(project_task_type, model_type)`; unknowns fall back via `MODEL_TYPE_DEFAULTS` (`inference/core/roboflow_api.py`) — never silently return `None` (`roboflow.py` raises `ModelArtefactError`).
- A new `(task, variant)` entry in `ROBOFLOW_MODEL_TYPES` MUST have a corresponding class import guarded by its enablement flag; a missing entry silently produces `ModelNotRecognisedError` (#2128 added `rfdetr`/`rfdetr` seg keys that the NAS/inference-models backend needed).
- Response schemas (`inference/core/entities/responses/inference.py`) are a public API — field additions must stay backward-compatible (default values, not `Optional`, because callers use `model_dump(exclude_none=True)`, #1036).
- Alias resolution must happen once, up front, everywhere a raw `model_id` enters (#193, #1226, #225-adjacent).

## Standards enforced here
- **Version bump is mandatory on any behavioral change.** Nearly every merged PR bumps `inference/core/version.py` (#1920, #495, #225, #1036, #1848, #2105, #2270). A model/registry PR with no version bump is a block.
- **Alias resolution first.** Call `resolve_roboflow_model_alias(model_id=...)` before `get_model_id_chunks`, before obtaining chunks/weights, and before auth checks. #1226 fixed a bug where alias resolution was missing before obtaining chunks in `_check_if_api_key_has_access_to_model`; #193 established the pattern in both the model layer and registry.
- **New model wiring is a triple.** Adding a model requires: (a) an env enablement flag in `inference/core/env.py` (e.g. `QWEN_3_5_ENABLED`, #2105), (b) a guarded import + `ROBOFLOW_MODEL_TYPES` entry in `inference/models/utils.py`, and (c) registry recognition (`GENERIC_MODELS` or API task type). Missing any leg = block.
- **Guard optional imports with try/except + `warnings.warn(category=ModelDependencyMissing)`.** Deprecated/optional models (Gaze) must not hard-crash import of `utils.py`; #2338 wrapped the Gaze import in `try/except` emitting a `ModelDependencyMissing` warning. New optional-model wiring must follow this or it breaks slim images.
- **Thread-safety for stateful/session models.** ONNX sessions and cache-mutating models need instance-level `threading.Lock`. #1510 added `self._session_lock` around `run_session_via_iobinding`/`ort_session.run` for classification, clip (per visual/textual session), gaze, rfdetr. #1549 wrapped whole `infer_from_request` of SAM/SAM2 in a `self._state_lock` (renamed from `_ort_session_lock`) because embedding/cache state is shared. Any new ONNX `.run(...)` or mutable per-image cache without a lock is a finding.
- **`environment.json` / `CLASS_MAP` parsing must be order-stable.** Class names come from `CLASS_MAP` indexed by integer string keys `str(i)` in order, NOT `sorted(keys())` (#225 fixed lexicographic sort mangling `"10"` before `"2"`). See `get_class_names_from_environment_file` in `inference/core/models/roboflow.py`.
- **`USE_INFERENCE_MODELS` bridge parity.** When adding a legacy model that also has an inference-models backend, add the matching `InferenceModels*Adapter` swap in the `if USE_INFERENCE_MODELS:` block of `utils.py` and register local-dir task types (#2479, #2105). Adapters must respect `DISABLED_INFERENCE_MODELS_BACKENDS` / `VALID_INFERENCE_MODELS_BACKENDS` when computing `backend=` (#2096).
- **`countinference` / `service_secret` threading.** Weight/metadata fetches accept and forward `countinference` and `service_secret` down to `get_roboflow_*`/`get_extra_weights_provider_headers` (#1382, #1575, #1768-adjacent). New fetch paths must plumb them through, not drop them.
- **Logging is lazy and low-noise.** Use `logger.debug` for per-inference logs (#1793 moved OWLv2 per-inference from INFO→DEBUG) and `%`-style lazy args, not f-strings, in hot/logging paths (#1578).
- **Concurrency-safe artifact download.** Downloading HF/transformer artifacts must be guarded by a `FileLock` to avoid races between workers (#1578).
- **`postprocess`/NMS index math is load-bearing.** Padding-undo uses `infer_shape[1]` for x and `[0]` for y (#495); keypoint slicing is `pred[7:]` with class id at `pred[4 + len(class_names)]` (#216); `argpartition` must guard `len(logits_flat) > max_detections` or it raises (#1920); empty predictions per image append `[]` and `continue` (#387). Treat changes to these as high-scrutiny.

## Required companions
- **`inference/core/version.py`** — bump `__version__` on any behavioral change (every reference bugfix/feature PR does this).
- **Env flag in `inference/core/env.py`** — for any new model or new global toggle (`*_ENABLED`, `DISABLED_INFERENCE_MODELS_BACKENDS`); #2096, #2105, #2338.
- **`inference/models/utils.py` registry entry** — new `(task, variant)` → class mapping, guarded by the enablement flag and try/except; plus the `USE_INFERENCE_MODELS` adapter swap when a backend exists.
- **`GENERIC_MODELS` in `inference/core/registries/roboflow.py`** — for foundation/core models keyed by whole `model_id` or `dataset_id`.
- **Tests.** Registry/id changes → `tests/inference/unit_tests/core/registries/test_roboflow.py`; base/adapter changes → `tests/inference/unit_tests/core/models/test_roboflow.py`, `test_inference_models_adapters.py`, `test_stubs.py`; aliases → `tests/inference/unit_tests/models/test_aliases.py`; concrete models → `tests/inference/unit_tests/models/test_*` and prediction/latency tests under `tests/inference/models_predictions_tests/`. Reference PRs add matching tests (#2479, #193, #1036).
- **Docs.** New foundation model → `docs/foundation/<model>.md`; new fine-tuned family → `docs/fine-tuned/<model>.md` (#193 documented pretrained YOLOv8 ids; #141 added stubs doc; #562 documented `ONNXRUNTIME_EXECUTION_PROVIDERS`).
- **Response schema defaults** — schema field additions in `inference/core/entities/responses/inference.py` must ship a default, not `Optional` (#1036).

## Common pitfalls & past regressions
- **#2128** — a new model variant works via inference-models backend but the *old* `ROBOFLOW_MODEL_TYPES` registry lacks the bare `("object-detection","rfdetr")` / `("instance-segmentation","rfdetr")` keys → `ModelNotRecognisedError`. Check registry keys cover every served variant name.
- **#1226** — alias not resolved before obtaining chunks in the auth check → aliased models fail authorization. Verify `resolve_roboflow_model_alias` precedes any `get_model_id_chunks`/API call.
- **#225** — `sorted(CLASS_MAP.keys())` sorts lexicographically (`"10"` < `"2"`), scrambling class order. Must index by `str(i)` in numeric range.
- **#1036 / #873** — confidence threshold not applied to classification (#873 dropped `**kwargs` on `infer`; #1036 must filter scores AND keep `top`/`confidence` defaulting to `""`/`0.0` when the filtered list is empty). Watch for empty-list `results[0]` IndexError.
- **#1510 / #1549** — concurrent GPU/ONNX session access and shared embedding caches corrupt state without a lock. Any new `ort_session.run` or per-image cache mutation needs a lock.
- **#495** — swapped x/y in padding-undo (`pad_x`/`pad_y`) shifts every box; also `width_remainder` reuse. Check dimension indexing in preprocess/postprocess.
- **#216** — keypoint index math off by class-count; `process_inference_request` must be `await`ed (async regression).
- **#1920** — `np.argpartition(-x, k)` raises when `k >= len(x)`; guard with the length check.
- **#387** — empty per-image predictions in instance segmentation must append `[]`/`continue`, not index into an empty array.
- **#2338** — an optional/deprecated model import (Gaze) crashing at module import breaks the whole `utils.py` load in slim images; must be try/except-guarded.
- **#2270 / #2271 / #1046 / #976 / #943 / #1329 / #1196 / #1787** — OWLv2 `torch.compile` monkey-patching and "background compilation" / precomputed-embeddings / streaming-download changes were repeatedly reverted. Treat `torch.compile`, monkey-patching, background compilation, and download-mode changes as high-risk; expect a revert plan and strong test evidence.
- **#1404** — over-eager `ModelArtefactError` wrapping and retry loops around cache reads hid corrupt-download handling; corruption should be treated at the root cache layer.
- **#1054** — checking `torch` presence via `dir()` is unreliable; don't gate on `"torch" in dir()`.

## Review checklist
1. Version bumped in `inference/core/version.py`? (block if a behavioral change lacks it.)
2. New model? Confirm the triple: env flag (`env.py`) + guarded `ROBOFLOW_MODEL_TYPES` entry (`utils.py`) + registry recognition (`GENERIC_MODELS` or task-type mapping). Optional imports wrapped in try/except with `ModelDependencyMissing` warning.
3. Every raw `model_id` entry point calls `resolve_roboflow_model_alias` before chunking/auth/weight fetch.
4. `get_model_type` cannot return `None`; new branches keep `MODEL_TYPE_DEFAULTS` fallback and `_ensure_model_supported_on_this_deployment` gating intact.
5. Any new `ort_session.run` / `run_session_via_iobinding` / shared cache mutation is wrapped in an instance `Lock`.
6. Class-name/`environment.json` parsing indexes by numeric `str(i)`, not `sorted(keys)`.
7. Response schema additions use defaults (backward-compat with `model_dump(exclude_none=True)`), not `Optional`.
8. `countinference` and `service_secret` are threaded through new metadata/weight fetch paths.
9. `USE_INFERENCE_MODELS` adapter parity: matching `InferenceModels*Adapter` swap added; `backend` respects `DISABLED_INFERENCE_MODELS_BACKENDS`.
10. Postprocess/NMS index math and empty-prediction handling unchanged unless intentionally fixed with a test.
11. Logging in hot paths is `debug` + lazy `%` args; artifact downloads use `FileLock`.
12. Tests added/updated in the matching area; docs added for new foundation/fine-tuned models.
13. `torch.compile` / monkey-patch / background-compilation / download-streaming changes carry explicit justification and tests (revert history).

## Key files & entry points
- `inference/core/registries/roboflow.py` — `get_model_type`, `_check_if_api_key_has_access_to_model`, `GENERIC_MODELS`, metadata cache.
- `inference/models/utils.py` — `ROBOFLOW_MODEL_TYPES`, `get_roboflow_model`, `USE_INFERENCE_MODELS` swap block, per-model enablement wiring.
- `inference/core/models/base.py` — `BaseInference`/`Model` contract, `@usage_collector`, `infer_from_request`.
- `inference/core/models/roboflow.py` — ONNX artifact download/cache, `environment.json` + `CLASS_MAP` parsing, `get_class_names_from_environment_file`.
- `inference/core/models/inference_models_adapters.py` — inference-models bridge adapters.
- `inference/models/aliases.py` — `resolve_roboflow_model_alias`, `REGISTERED_ALIASES`.
- `inference/core/models/{object_detection,instance_segmentation,keypoints_detection,classification,semantic_segmentation}_base.py`, `stubs.py`.
- `inference/core/env.py` (flags), `inference/core/version.py`, `inference/core/roboflow_api.py` (`MODEL_TYPE_DEFAULTS`).

## Reference PRs
- [#2105](https://github.com/roboflow/inference/pull/2105) — redirect versionless model auth/metadata to new registry via `USE_INFERENCE_MODELS` (feature; flag + registry + version bump).
- [#2479](https://github.com/roboflow/inference/pull/2479) — load locally-stored `inference_models` packages via registry (`_get_local_model_type`, adapter wiring, tests).
- [#2128](https://github.com/roboflow/inference/pull/2128) — add missing `ROBOFLOW_MODEL_TYPES` keys so RFDETR/NAS works with inference-models backend (registry-completeness bugfix).
- [#2338](https://github.com/roboflow/inference/pull/2338) — try/except-guard Gaze import with `ModelDependencyMissing` warning (optional-import safety).
- [#1510](https://github.com/roboflow/inference/pull/1510) — per-instance ONNX session `Lock` to avoid GPU concurrency corruption.
- [#1549](https://github.com/roboflow/inference/pull/1549) — wrap state-sensitive SAM/SAM2 handlers in `_state_lock` for thread safety.
- [#1036](https://github.com/roboflow/inference/pull/1036) — apply confidence to classification + backward-compatible response defaults.
- [#225](https://github.com/roboflow/inference/pull/225) — fix `CLASS_MAP` ordering (numeric `str(i)` not `sorted`).
- [#193](https://github.com/roboflow/inference/pull/193) — introduce model aliases + resolve at model and registry layers (+ docs + tests).
- [#495](https://github.com/roboflow/inference/pull/495) — fix x/y swap in padding-undo for predicted boxes.
- [#1920](https://github.com/roboflow/inference/pull/1920) — guard `argpartition` when `max_detections >= len(logits)` in RF-DETR postprocess.
- [#2270](https://github.com/roboflow/inference/pull/2270) / [#1046](https://github.com/roboflow/inference/pull/1046) — OWLv2 `torch.compile` monkey-patch fix / background-compilation revert (high-risk compile changes).

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-prediction-integrity`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
