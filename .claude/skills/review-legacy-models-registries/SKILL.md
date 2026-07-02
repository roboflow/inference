---
name: review-legacy-models-registries
description: Load when a PR touches inference/models/** (legacy ORT/HF models, models/utils.py ROBOFLOW_MODEL_TYPES + get_roboflow_model, models/aliases.py), inference/core/models/** (base.py, roboflow.py CLASS_MAP parsing, *_base.py, stubs.py, inference_models_adapters.py), or inference/core/registries/** (get_model_type, GENERIC_MODELS). Diff signals: new ROBOFLOW_MODEL_TYPES key, resolve_roboflow_model_alias, _session_lock/_state_lock, ort_session.run, USE_INFERENCE_MODELS, no version.py bump.
---

# Reviewing legacy-models-registries changes

## Scope
Trigger this skill when a PR changes any of:
- `inference/models/**` — concrete legacy model implementations (rfdetr, owlv2, sam/sam2/sam3, clip, gaze, yolo*, easy_ocr, doctr, florence2, paligemma, qwen*, smolvlm, etc.), `inference/models/utils.py` (`ROBOFLOW_MODEL_TYPES` + `get_roboflow_model`), and `inference/models/aliases.py`.
- `inference/core/models/**` — base contracts (`base.py`, `roboflow.py`), task bases (`object_detection_base.py`, `instance_segmentation_base.py`, `keypoints_detection_base.py`, `classification_base.py`, `semantic_segmentation_base.py`), `stubs.py`, and `inference_models_adapters.py` (the `USE_INFERENCE_MODELS` bridge adapters).
- `inference/core/registries/**` — `base.py` (`ModelRegistry`) and `roboflow.py` (`get_model_type`, auth checks, metadata cache).

OUT of scope (defer to other skills): Workflows blocks, HTTP API handlers (`inference/core/interfaces/**`), model managers (`inference/core/managers/**`), the standalone `inference_models` package itself, and the tensor-data-representation `_tensor` sibling effort.

## Review checklist
Severity-tagged. Each item points at its canonical rule in **Standards** below.

- **BLOCK** — behavioral change with no `__version__` bump in `inference/core/version.py`. (S1)
- **BLOCK** — new model missing any leg of the wiring triple: env flag (`env.py`) + guarded `ROBOFLOW_MODEL_TYPES` entry (`utils.py`) + registry recognition (`GENERIC_MODELS` or API task type). (S3)
- **BLOCK** — a served `(task, variant)` has no `ROBOFLOW_MODEL_TYPES` key → `ModelNotRecognisedError`. (S3)
- **BLOCK** — a raw `model_id` reaches chunking / auth / weight fetch without `resolve_roboflow_model_alias` first. (S2)
- **BLOCK** — `get_model_type` can return `None`; new branch drops the `MODEL_TYPE_DEFAULTS` fallback or `_ensure_model_supported_on_this_deployment` gating. (S4)
- **BLOCK** — new `ort_session.run` / `run_session_via_iobinding` / shared per-image cache mutation without an instance `Lock`. (S5)
- **BLOCK** — class-name / `environment.json` parsing indexes by `sorted(keys)` instead of numeric `str(i)`. (S6)
- **BLOCK** — optional/deprecated model import not wrapped in try/except + `ModelDependencyMissing` warning (breaks slim-image `utils.py` load). (S3)
- **FLAG** — response-schema field added as `Optional` instead of with a default (breaks `model_dump(exclude_none=True)` callers). (S7)
- **FLAG** — new metadata/weight fetch path drops `countinference` / `service_secret`. (S8)
- **FLAG** — `USE_INFERENCE_MODELS` adapter parity missing, or `backend=` ignores `DISABLED_INFERENCE_MODELS_BACKENDS` / `VALID_INFERENCE_MODELS_BACKENDS`. (S9)
- **FLAG** — postprocess / NMS index math or empty-prediction handling changed without a covering test. (S10)
- **FLAG** — `torch.compile` / monkey-patch / background-compilation / download-streaming change without explicit justification + tests (repeatedly reverted). (S11)
- **FLAG** — model overriding `infer_from_request` (SAM2/SAM3) drops the `@usage_collector("model")` decorator. (S12)
- **NIT** — per-inference logging above `debug`, or f-strings instead of lazy `%`-args in hot paths. (S13)
- **NIT** — HF/transformer artifact download not guarded by `FileLock`. (S13)

### Not blocking
- Do NOT demand a version bump for pure comment/docstring/test-only edits, or for a change that is itself the version bump.
- Do NOT demand the full wiring triple when the PR only edits an existing model's internals (no new `(task, variant)`).
- Do NOT demand a new adapter when the touched model has no `inference_models` backend.
- Do NOT block on `Optional`-vs-default when the field is genuinely nullable in the response contract and callers do not use `exclude_none`.
- Do NOT flag missing locks on stateless, read-only compute that never touches a shared ORT session or cache.

## Standards

**The base contract.** Every model is a `Model` (`inference/core/models/base.py`). `BaseInference.infer()` runs `preprocess → predict → postprocess` under telemetry spans and is wrapped by `@usage_collector("model")`. `make_response(...)` is a separate abstract method that builds the `InferenceResponse`. `Model.infer_from_request(request)` orchestrates: calls `infer(...)`, then stamps `response.time`, `response.inference_id`, and optional `visualization`. ONNX Roboflow models flow through `inference/core/models/roboflow.py` (artifact download → cache → `environment.json` parse → class names/colors/preproc).

**Registry / id resolution.** `inference/core/registries/roboflow.py::get_model_type(model_id, api_key)` returns `(task_type, model_type)` by: resolving aliases, `GENERIC_MODELS` (whole `model_id` then `dataset_id`), `_get_local_model_type` (local `inference_models` dirs), stub versions, then the Roboflow API. The tuple keys into `ROBOFLOW_MODEL_TYPES` (`inference/models/utils.py`) to pick a class.

- **S1 — Version bump is mandatory on any behavioral change.** Bump `__version__` in `inference/core/version.py` (every reference bugfix/feature PR does this — #1920, #495, #225, #1036, #1848, #2105, #2270).
- **S2 — Alias resolution first.** Call `resolve_roboflow_model_alias(model_id)` (`inference/models/aliases.py`) before `get_model_id_chunks`, before obtaining chunks/weights, and before auth checks (missing before chunk-fetch in `_check_if_api_key_has_access_to_model` broke aliased-model authorization, #1226; pattern established in both model and registry layers, #193).
- **S3 — New model wiring is a triple.** (a) env enablement flag in `inference/core/env.py` (e.g. `QWEN_3_5_ENABLED`, `CORE_MODEL_*_ENABLED`, #2105); (b) guarded import + `ROBOFLOW_MODEL_TYPES` entry in `inference/models/utils.py`; (c) registry recognition via `GENERIC_MODELS` or API task type. A registry key must exist for every served variant name (missing bare `("object-detection","rfdetr")` / `("instance-segmentation","rfdetr")` keys yielded `ModelNotRecognisedError` even though the inference-models backend served the variant, #2128). Optional/deprecated imports (Gaze) must be wrapped in try/except emitting a `warnings.warn(category=ModelDependencyMissing)`, or a crash at import breaks the whole `utils.py` load in slim images (#2338).
- **S4 — `get_model_type` never returns `None`.** Unknowns fall back via `MODEL_TYPE_DEFAULTS` (`inference/core/roboflow_api.py`); `_ensure_model_supported_on_this_deployment` still gates. A silent `None` surfaces downstream as `ModelArtefactError`.
- **S5 — Thread-safety for stateful/session models.** ONNX sessions and cache-mutating models need an instance-level `threading.Lock`. `self._session_lock` (initialized in `inference/core/models/roboflow.py`) guards `run_session_via_iobinding` / `ort_session.run` in classification, clip, gaze, rfdetr, yolo* (#1510). SAM/SAM2 wrap the whole `infer_from_request` in `self._state_lock` (renamed from `_ort_session_lock`) because embedding/cache state is shared (#1549). Any new `.run(...)` or mutable per-image cache without a lock is a finding.
- **S6 — `environment.json` / `CLASS_MAP` parsing must be order-stable.** Class names come from `CLASS_MAP` indexed by integer string keys `str(i)` in numeric order, NOT `sorted(keys())` — lexicographic sort put `"10"` before `"2"` and scrambled classes (#225). See `get_class_names_from_environment_file` in `inference/core/models/roboflow.py`.
- **S7 — Response-schema additions ship a default, not `Optional`.** Schemas in `inference/core/entities/responses/inference.py` are public API; callers use `model_dump(exclude_none=True)`, so a new field needs a concrete default. Classification also keeps `top`/`confidence` defaulting to `""`/`0.0` when the filtered list is empty — watch for `results[0]` IndexError on empty predictions (#1036; earlier #873 dropped `**kwargs` on classification `infer`).
- **S8 — `countinference` / `service_secret` threading.** Weight/metadata fetches accept and forward `countinference` and `service_secret` down to `get_roboflow_*` / `get_extra_weights_provider_headers` (#1382, #1575). New fetch paths must plumb them through, not drop them.
- **S9 — `USE_INFERENCE_MODELS` bridge parity.** When adding a legacy model that also has an inference-models backend, add the matching `InferenceModels*Adapter` swap in the `if USE_INFERENCE_MODELS:` block of `utils.py` and register local-dir task types (#2479, #2105). Adapters must respect `DISABLED_INFERENCE_MODELS_BACKENDS` / `VALID_INFERENCE_MODELS_BACKENDS` (`inference/core/env.py`) when computing `backend=` (#2096).
- **S10 — Postprocess / NMS index math is load-bearing.** Padding-undo uses `infer_shape[1]` for x and `[0]` for y (x/y swap shifted every box, #495); keypoint slicing is `pred[7:]` with class id at `pred[4 + len(class_names)]` (#216); `np.argpartition(-logits_flat, max_detections)` must guard `max_detections < len(logits_flat)` or it raises (`inference/models/rfdetr/rfdetr.py`, #1920); empty per-image predictions append `[]` and `continue` rather than indexing an empty array (#387). Treat changes as high-scrutiny.
- **S11 — Compile / download-mode changes are high-risk.** OWLv2 `torch.compile` monkey-patching, background compilation, precomputed-embeddings, and streaming-download changes were repeatedly reverted (#2270/#2271/#1046/#976/#943/#1329/#1196/#1787). Also: `ModelArtefactError`-wrapping retry loops around cache reads hid corrupt-download handling — treat corruption at the root cache layer (#1404); don't gate on `"torch" in dir()` (#1054). Expect a revert plan and strong test evidence.
- **S12 — `@usage_collector("model")` must survive an override.** `BaseInference.infer` carries the decorator; models overriding `infer_from_request` (SAM2/SAM3) must re-apply `@usage_collector("model")` themselves (#1768).
- **S13 — Logging is lazy and low-noise; downloads are race-safe.** Use `logger.debug` for per-inference logs (OWLv2 moved INFO→DEBUG, #1793) and `%`-style lazy args, not f-strings, in hot paths (#1578). Guard HF/transformer artifact downloads with `FileLock` to avoid worker races (#1578).

## Required companions
- **`inference/core/version.py`** — bump `__version__` on any behavioral change.
- **`inference/core/env.py`** — env flag for any new model or global toggle (`*_ENABLED`, `DISABLED_INFERENCE_MODELS_BACKENDS`).
- **`inference/models/utils.py`** — new `(task, variant)` → class mapping, guarded by the enablement flag and try/except, plus the `USE_INFERENCE_MODELS` adapter swap when a backend exists.
- **`GENERIC_MODELS` in `inference/core/registries/roboflow.py`** — for foundation/core models keyed by whole `model_id` or `dataset_id`.
- **Tests.** Registry/id changes → `tests/inference/unit_tests/core/registries/test_roboflow.py`; base/adapter changes → `tests/inference/unit_tests/core/models/test_roboflow.py`, `test_inference_models_adapters.py`, `test_stubs.py`; aliases → `tests/inference/unit_tests/models/test_aliases.py`; concrete models → `tests/inference/unit_tests/models/test_*` and `tests/inference/models_predictions_tests/`.
- **Docs.** New foundation model → `docs/foundation/<model>.md`; new fine-tuned family → `docs/fine-tuned/<model>.md`.

## Key files & entry points
- `inference/core/registries/roboflow.py` — `get_model_type`, `_get_local_model_type`, `_check_if_api_key_has_access_to_model`, `_ensure_model_supported_on_this_deployment`, `GENERIC_MODELS`.
- `inference/models/utils.py` — `ROBOFLOW_MODEL_TYPES`, `get_roboflow_model`, `USE_INFERENCE_MODELS` swap block, per-model enablement wiring.
- `inference/core/models/base.py` — `BaseInference`/`Model`, `infer`, `infer_from_request`, `make_response`, `@usage_collector`.
- `inference/core/models/roboflow.py` — ONNX artifact download/cache, `get_class_names_from_environment_file`, `_session_lock`, `FileLock`.
- `inference/core/models/inference_models_adapters.py` — `InferenceModels{ObjectDetection,InstanceSegmentation,KeyPointsDetection,Classification,SemanticSegmentation}Adapter`.
- `inference/models/aliases.py` — `resolve_roboflow_model_alias`, `REGISTERED_ALIASES`.
- `inference/core/models/{object_detection,instance_segmentation,keypoints_detection,classification,semantic_segmentation}_base.py`, `stubs.py`.
- `inference/core/env.py` (flags), `inference/core/version.py`, `inference/core/roboflow_api.py` (`MODEL_TYPE_DEFAULTS`).

## Reference PRs
- [#2105](https://github.com/roboflow/inference/pull/2105) — redirect versionless model auth/metadata to new registry via `USE_INFERENCE_MODELS`.
- [#2479](https://github.com/roboflow/inference/pull/2479) — load locally-stored `inference_models` packages via registry (`_get_local_model_type`, adapter wiring, tests).
- [#2128](https://github.com/roboflow/inference/pull/2128) — add missing `ROBOFLOW_MODEL_TYPES` keys so RFDETR/NAS works with inference-models backend.
- [#2338](https://github.com/roboflow/inference/pull/2338) — try/except-guard Gaze import with `ModelDependencyMissing` warning.
- [#1510](https://github.com/roboflow/inference/pull/1510) — per-instance ONNX `_session_lock` against GPU concurrency corruption.
- [#1549](https://github.com/roboflow/inference/pull/1549) — wrap SAM/SAM2 handlers in `_state_lock`.
- [#1036](https://github.com/roboflow/inference/pull/1036) — apply confidence to classification + backward-compatible response defaults.
- [#225](https://github.com/roboflow/inference/pull/225) — fix `CLASS_MAP` ordering (numeric `str(i)`, not `sorted`).
- [#193](https://github.com/roboflow/inference/pull/193) — introduce model aliases + resolve at model and registry layers.
- [#495](https://github.com/roboflow/inference/pull/495) — fix x/y swap in padding-undo.
- [#1920](https://github.com/roboflow/inference/pull/1920) — guard `argpartition` when `max_detections >= len(logits_flat)`.
- [#2270](https://github.com/roboflow/inference/pull/2270) / [#1046](https://github.com/roboflow/inference/pull/1046) — OWLv2 `torch.compile` monkey-patch / background-compilation reverts.

## Related topic skills
When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):
- `review-topic-prediction-integrity`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-backward-compat-and-versioning`
- `review-topic-test-hygiene`
