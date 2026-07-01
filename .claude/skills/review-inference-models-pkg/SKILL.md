---
name: review-inference-models-pkg
description: Review guidance for PRs touching inference_models/inference_models/** (models, auto_loaders, weights_providers, runtime_introspection, base entities, errors, configuration) and inference_models/docs/** — enforces the inference-models package contracts, model-package negotiation/auto-loading invariants, prediction-entity shapes, error+help_url conventions, and the version/changelog/lock companion requirements.
---

# Reviewing inference-models-pkg changes

## Scope
Triggers when a PR changes any of:
- `inference_models/inference_models/**` — model backends, `models/base/` prediction entities, `models/common/` (onnx/trt/torch/rle helpers), `models/auto_loaders/**`, `weights_providers/**`, `runtime_introspection/**`, `errors.py`, `configuration.py`, `constants.py`, `entities.py`.
- `inference_models/pyproject.toml`, `inference_models/uv.lock`, `inference_models/docs/**` (changelog, errors/, models/, api-reference/, contributors/).
- Legacy `inference_experimental/inference_exp/**` paths in older PRs — the package was renamed `inference-exp` → `inference-models` (#1851, #1902). Treat both trees as the same surface.

OUT of scope (defer to sibling skills): the `inference/core/**` model adapters, workflow blocks, HTTP API, and the ENABLE_TENSOR_DATA_REPRESENTATION `_tensor` block pivot — those belong to the core/workflows review skills. Only review the `inference_models` side of a cross-cutting PR here.

## What this surface is
`inference-models` is a standalone pip package (`inference-models`, currently `0.29.x`) that loads and runs CV models. Architecture and the contracts a reviewer must protect:

- **`AutoModel.from_pretrained(model_id, ...)`** is the public entrypoint. It negotiates a *model package* against the runtime environment, downloads/caches weights, then calls the model class's **`from_pretrained` classmethod contract** (the one hard requirement per `docs/contributors/adding-model.md`). New models MUST implement it and accept `**kwargs` so `AutoModel` can forward negotiation-derived params.
- **Model registry** `models/auto_loaders/models_registry.py`: `REGISTERED_MODELS` keyed by `(ModelArchitecture, TaskType, BackendType)` → `LazyClass` / `RegistryEntry`. A new model/backend is invisible until registered here. `RegistryEntry.supported_model_features` / `required_model_features` (e.g. `{"nms_fused"}`) gate feature negotiation.
- **Backends**: ONNX / TRT / Torch / TorchScript. Auto-loader **ranks ONNX over Torch** (#2047, #1811) — do not silently reorder ranking.
- **Prediction entities** (`models/base/*.py`): `Detections`, `InstanceDetections`, `KeyPoints`, `ClassificationPrediction`, etc. — torch-tensor dataclasses each exposing `to_supervision()`. Their field shapes/dtypes and `to_supervision()` output are a public contract consumed by `inference/core` adapters and Workflows; changing a field is a breaking change.
- **Errors** (`errors.py`): a class tree under `BaseInferenceModelsError` carrying an optional `help_url`. Raised errors point at a `docs/errors/<page>#<anchor>` page. The `inference/core` step-error handlers map specific classes (`ModelNotFoundError`, `UnauthorizedModelAccessError`, `ModelPackageRestrictedError`, `ModelPackageAlternativesExhaustedError`) to HTTP/workflow errors — renaming/removing a class breaks that mapping.

## Standards enforced here
1. **`from_pretrained` + `**kwargs` contract.** Every model class is loaded through it and must tolerate extra negotiation kwargs; new load-time params are threaded as optional kwargs, not positional (e.g. `torchscript_state_global_lock: Optional[Lock] = None` in #2373, `rf_detr_max_input_resolution` in #2145). Evidence: `docs/contributors/adding-model.md`, #2373.
2. **Register the model.** New arch/task/backend combos must land in `REGISTERED_MODELS` with the right `(arch, task, BackendType)` key and `LazyClass(module_name, class_name)`; feature-gated backends declare `supported_model_features` (e.g. `nms_fused`). Evidence: `models_registry.py`, #1637/#1645/#1786 feature PRs.
3. **Prediction-entity shape stability.** Do not change `xyxy`/`class_id`/`confidence`/`mask` semantics without updating every backend + `to_supervision()` + api-reference doc. New representations are added as `Union`/opt-in, not by mutating existing fields — RLE masks added as `Union[torch.Tensor, InstancesRLEMasks]` + `InstanceSegmentationMaskFormat` literal (#2260); `KeyPoints` gained `covariance`/`detection_confidence` additively (changelog `0.29.7`).
4. **Device / dtype discipline.** Tensors kept on-device until `to_supervision()` does `.cpu().numpy()`. No leaked global autocast — a leaked bf16 `torch.autocast` context that poisons the whole process was a real bug (#2363). Cast processor inputs to the resolved model dtype per device (Jetson bfloat16 fix, changelog `0.29.4`).
5. **Third-party compat via feature-detection, not version pins in code.** Optional-signature params are probed with `inspect.signature(...)` and passed conditionally. On this surface, guard transformers/flash-attn API drift this way (#2257 owlv2, #2266 GLM-OCR). (The supervision `sv.KeyPoints(confidence=...)` guard often cited for #2467 lives in the workflows viz block — out of scope here.)
6. **Errors carry `help_url` to a real docs anchor.** New `errors.py` classes get a `docs/errors/<page>.md` section and are raised with `help_url="https://inference-models.roboflow.com/errors/<page>/#<lowercased-classname>"`. Evidence: `configuration.py:341`, `weights_providers/*.py`, `ModelPackageRestrictedError` doc added in #2145.
7. **Env-var config is validated and documented.** New env flags follow the `INFERENCE_MODELS_*` naming, are validated (raising `InvalidEnvVariable` on bad values, `configuration.py`), and documented in `docs/how-to/environment-variables.md` + changelog (Triton RF-DETR flags, changelog `0.29.6`).
8. **Concurrency safety for global-mutating loaders.** `torch.jit.load/script` mutate a non-thread-safe process-global registry — must be serialized behind a shared lock threaded from the model manager (#2373). Any new global-state loader needs the same treatment.
9. **Runtime introspection must degrade gracefully.** CUDA/Jetson/L4T detection is best-effort; a failed probe must not crash negotiation (CUDA-version acquisition fix #2515, Jetson env spec #1796, JetsonTypeResolutionError). 
10. **ONNX/TRT edge-cases.** Batch-size fast paths must respect both `min_batch_size` and `max_batch_size` bounds (#2112). TRT artefacts are inconsistent — output-name verification is skipped where artefacts are known-bad (#1790 YOLO-NAS), and `engine_host_code_allowed` must be set where required (#1791 YOLOv10).

## Required companions
Block the PR unless these accompany a functional change:
- **Version bump** in `inference_models/pyproject.toml` (`version = "0.29.x"`) — present on essentially every fix/feature (#2467, #2145, #2180, #2047, #2260, #2449). Bump `inference/core/version.py` too when the change reaches the server (`1.2.3`→`1.2.4` in #2260).
- **Changelog entry** in `inference_models/docs/changelog.md` under a new `## \`0.29.x\`` heading with `Added`/`Fixed`/`Changed` — every reference fix/feature does this. Missing changelog = block.
- **Lockfile** `inference_models/uv.lock` regenerated whenever `pyproject.toml` deps change (`uv sync`), plus mirrored pins in `requirements/*.txt` when the dep is also a server dep (#2047, #2449 security bumps, #2415/#2510 dep updates). A dep change without a matching `uv.lock` update = block.
- **Docs**: new error class → `docs/errors/*` section; new model → `docs/models/<model>.md` + registry + tests (per `adding-model.md`); new env var → `docs/how-to/environment-variables.md`.
- **Tests** in `inference_models/tests/`: `integration_tests/models/` (mandatory for a new model — "the most important tests" per `writing-tests.md`), `e2e_platform_tests/` (AutoModel path), `unit_tests/` for pure logic (ranking, negotiation, post-processing, introspection — see #2047, #2515).

## Common pitfalls & past regressions
- **#2467** — supervision 0.29.0 renamed the `sv.KeyPoints(confidence=...)` kwarg. The guard + viz breakage live in the workflows keypoint viz block (**out of scope here**); on THIS surface #2467 was only an additive `keypoints_detection.py` enrichment. Don't treat supervision-constructor guards as an inference_models concern.
- **#2363** — `Sam3TrackerPredictor.__init__` entered a CUDA bf16 `torch.autocast` and never exited it, leaking globally and breaking unrelated models in the same process/pytest session. Check: no autocast/`torch.set_grad_enabled`/dtype context escapes its scope.
- **#2373** — concurrent `torch.jit.load/script` corrupts a process-global TorchScript registry. Check: TorchScript/global-mutating loads are serialized behind a shared `Lock`.
- **#2112** — ONNX batch fast-path ignored `min_batch_size` for fixed-batch models, sending under-sized batches. Check: batch bounds compare against *both* min and max.
- **#1626** — YOLOv8 keypoint post-processing indexed with `5 + num_classes` and `view(..., -1, 3)`; multi-class models broke. Correct is fixed offset `6:` and explicit `_key_points_slots_in_prediction`. Check: keypoint/segmentation tensor slicing uses model-declared slot counts, not `-1` inference.
- **#2075 / #1619 / #2029** — RF-DETR class remapping and TRT post-processing off-by-mapping; #1619 also model-cache-path sanitisation. Check: class-id remapping and cache-path sanitisation for RF-DETR variants.
- **#2257 / #2266** — transformers/flash-attn API drift silently broke OwlV2 / GLM-OCR. Check: HF-backed models guard against library version drift.
- **#1436** — MPS enabled for rfdetr but kept non-default on MacOS. Check: new device support does not change platform defaults.
- **Numerically-unstable e2e/GPU assertions** (#2026, #1646, #1649, #1809, #1638, #1843) — exact box-count/class-set assertions flake on GPU/ONNX. Check: new integration assertions are tolerant (`>=`, loosened tolerances) not brittle equality.
- **Reverts** (#2241 optimal-confidence, #2087 Qwen 3.5) — signal that speculative model/feature additions get rolled back cleanly; a revert must remove the feature from registry, CI, deps and changelog together.

## Review checklist
1. Functional change → is `inference_models/pyproject.toml` version bumped AND a `docs/changelog.md` entry added? If not, block.
2. Deps changed in `pyproject.toml` → is `uv.lock` regenerated and are `requirements/*.txt` pins updated where shared? 
3. New model/backend → registered in `REGISTERED_MODELS` with correct `(arch, task, BackendType)` key, `LazyClass` path resolvable, `from_pretrained(**kwargs)` implemented, integration + e2e tests + `docs/models/*` present?
4. New load-time param → threaded as optional `**kwargs`, defaulted, not positional; forwarded from `AutoModel`/manager?
5. Prediction entity touched → all backends + `to_supervision()` + `docs/api-reference/*` updated? New representation additive (`Union`/opt-in), not a field-shape break?
6. New/renamed `errors.py` class → docs/errors section + `help_url` anchor added; core step-error mapping still resolves the class?
7. New env var → `INFERENCE_MODELS_*` name, validated (`InvalidEnvVariable`), documented in `environment-variables.md`?
8. Torch: no global autocast/dtype/grad context leaks; global-mutating loads (`torch.jit.*`) serialized; tensors moved to CPU only at `to_supervision()`.
9. ONNX/TRT: batch bounds respect min+max; TRT flags (`engine_host_code_allowed`) set; graceful handling of inconsistent artefacts.
10. Runtime introspection changes degrade gracefully (no hard crash on missing CUDA/Jetson probe) and have unit tests.
11. Third-party (supervision/transformers/flash-attn) constructor/kwarg use is `inspect.signature`-guarded, not version-pinned in code.
12. New integration/GPU test assertions are tolerant, not brittle exact-equality.

## Key files & entry points
- `inference_models/inference_models/models/auto_loaders/` — `core.py` (AutoModel), `models_registry.py`, `auto_negotiation.py`, `ranking.py`, `model_cache_paths.py`.
- `inference_models/inference_models/models/base/*.py` — prediction entities + base model classes + `to_supervision()`.
- `inference_models/inference_models/models/common/` — `onnx.py`, `trt.py`, `torch.py` (torchscript lock), `rle_utils.py`.
- `inference_models/inference_models/{errors,configuration,constants,entities}.py`; `runtime_introspection/core.py`; `weights_providers/{core,roboflow,trt_manifest}.py`.
- `inference_models/pyproject.toml`, `uv.lock`; `inference_models/docs/{changelog.md,errors/,models/,api-reference/,contributors/adding-model.md,contributors/writing-tests.md}`.

## Reference PRs
- [#2373](https://github.com/roboflow/inference/pull/2373) — serialize TorchScript load/script behind a global lock (concurrency).
- [#2363](https://github.com/roboflow/inference/pull/2363) — eject leaked bf16 autocast from Sam3TrackerPredictor (dtype-context leak).
- [#2467](https://github.com/roboflow/inference/pull/2467) — additive `keypoints_detection.py` enrichment (the supervision-kwarg guard itself is in the workflows viz block, out of scope here).
- [#2112](https://github.com/roboflow/inference/pull/2112) — ONNX batch-size fast path respects min+max bounds.
- [#1626](https://github.com/roboflow/inference/pull/1626) — multi-class keypoint post-processing slot indexing fix.
- [#2260](https://github.com/roboflow/inference/pull/2260) — additive RLE mask representation across segmentation models (entity-shape + version/changelog companions).
- [#2145](https://github.com/roboflow/inference/pull/2145) — RF-DETR max input resolution + `ModelPackageRestrictedError` + errors doc.
- [#2180](https://github.com/roboflow/inference/pull/2180) — report model-loading failures from violated env constraints (error → core mapping).
- [#2047](https://github.com/roboflow/inference/pull/2047) — AutoLoader ranks ONNX over Torch (ranking + unit tests + docs).
- [#1959](https://github.com/roboflow/inference/pull/1959) — inference 1.0 RC1 wiring inference_models adapters (cross-package contract).
- [#1851](https://github.com/roboflow/inference/pull/1851) — rename inference-exp → inference-models (path history).
- [#2449](https://github.com/roboflow/inference/pull/2449) — security dep bumps with mirrored requirements + uv.lock + changelog.

## Related topic skills

When the PR also exhibits these cross-cutting concerns, load the matching topic skill too (see each skill's `description` for the trigger):

- `review-topic-prediction-integrity`
- `review-topic-backward-compat-and-versioning`
- `review-topic-concurrency-and-resource-safety`
- `review-topic-external-contract-and-silent-fallback`
- `review-topic-test-hygiene`
