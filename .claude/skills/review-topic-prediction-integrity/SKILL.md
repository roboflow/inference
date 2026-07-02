---
name: review-topic-prediction-integrity
description: Load when a diff touches `sv_detections_to_root_coordinates`, `scale_sv_detections`, `move_boxes`/`move_masks`, `POLYGON_KEY_IN_SV_DETECTIONS`, `add_inference_keypoints_to_sv_detections`, `serialise_sv_detections`, `mask_to_polygon`, `enforce_dense_masks_in_inference_models`, `binarization_threshold`, `run_nms_for_*`; or `xyxy +=`/`* scale` arithmetic; crop/stitch/perspective/dynamic_zones blocks; `inference_models/` pre/post-processing (resize/BGR-RGB/NMS/slice); a `supervision` bump.
---

# Review topic: Prediction, coordinate & pre/post-processing integrity

## When this applies
Content trigger (not one directory). Load when a diff does any of:
- Edits coordinate transforms / recovery: `sv_detections_to_root_coordinates`, `scale_sv_detections`, `attach_parent_coordinates_to_detections`, `attach_parents_coordinates_to_sv_detections`, `move_boxes`/`move_masks`, or any `xyxy +=` / `* scale` / shift arithmetic (`inference/core/workflows/core_steps/common/utils.py`).
- Touches crop / stitch / slice / perspective blocks: `transformations/dynamic_crop`, `transformations/stitch_images`, `transformations/stitch_ocr_detections`, `transformations/perspective_correction`, `transformations/dynamic_zones`, `fusion/detections_stitch`.
- Changes mask representation: binary mask ↔ polygon ↔ RLE, `mask_to_polygon`, `rle_masks_to_polygons`, `coco_rle_masks_to_numpy_mask`, `supported_mask_formats`, `enforce_dense_masks_in_inference_models`, `binarization_threshold`.
- Touches keypoints: parsing into `sv.Detections`, keypoint xy/confidence/class arrays, keypoint tensor slice offsets in model post-processing, keypoint visualization/edges.
- Edits model pre/post-processing in `inference_models/`: normalization, resize/letterbox, BGR/RGB, sigmoid/softmax, NMS params, class/background-index handling, tensor slice layout.
- Changes prediction serialization / response shape: `serializers.py`, `deserializers.py`, prediction entity models, block `outputs`/manifest keys.
- Bumps `supervision` — its internals (indexing, `mask_to_polygons`, annotators, edge maps) are load-bearing here.

## Review checklist

### BLOCK — must be fixed before merge
- [ ] **All geometry channels move together.** A transform that edits `xyxy` must apply the *same* shift/scale to `mask`, `data[POLYGON_KEY_IN_SV_DETECTIONS]`, and `keypoints_xy` when present. A channel left in a different frame ships wrong results with green happy-path tests (see Standards §1).
- [ ] **Manifest-declared output keys emitted on every branch.** Empty / zero-area / no-detections early returns still emit all declared keys (see Standards §2).
- [ ] **Backend parity.** A post-processing change in one backend is applied to every sibling backend; threshold/sigmoid/slice-offset/index drift changes real users' scores or classes (see Standards §4, §5).
- [ ] **Serialization contract intact.** Field names, types, and rounding downstream expects (`predictions`, `points`, `keypoints`, `mask`/`rle`, `class`, `confidence`) are preserved; per-keypoint `class`/`class_id`/`confidence`/`x`/`y` all present (see Standards §3).
- [ ] **No class/background-index off-by-one.** Class remapping and background index verified against the reference implementation (see Standards §5).

### FLAG — reviewer should raise
- [ ] **Threshold / activation change is gated.** New default threshold, sigmoid-vs-raw-logits, or column-slice change is justified against the reference and made configurable if it shifts scores (see Standards §5).
- [ ] **No ragged object-dtype arrays.** New `sv.Detections` data arrays are fixed-shape numeric; keypoints padded to uniform length (see Standards §6).
- [ ] **RLE ↔ dense honored per consumer.** Mask-format seam not broken for a downstream consumer (see Standards §3).
- [ ] **Polygon vertex source.** A stored precise polygon is preferred over `mask_to_polygon` regeneration; int rounding present (see Standards §3).

### NIT
- [ ] BGR/RGB channel order and PIL-vs-CV2 resize path match the reference decode (see Standards §5).
- [ ] `supervision` bump: annotators, `mask_to_polygons`, keypoint edge maps, and indexing still behave (see Standards §7).

### Not blocking — do NOT demand
- A missing empty/zero-area guard on a branch that is currently **unreachable** given the block's manifest (note it, don't block).
- Rounding/precision drift that stays within documented numerical tolerance.
- Cosmetic annotator drift from a `supervision` bump that does not change geometry, scores, or serialized shape.
- Polygon simplification that loses vertices without affecting boxes or class labels.
- Requiring config-gating for a threshold change that provably reproduces the reference output (the change *restores* parity rather than shifting it — e.g. the YOLO-ultralytics mask default `mask_binarization_threshold` `0.0`→`0.5` set via `INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASK_BINARIZATION_THRESHOLD`, #2212).

## Standards

1. **Coordinate frames stay consistent.** Every geometry channel of a detection — `xyxy`, `mask`, `data[POLYGON_KEY_IN_SV_DETECTIONS]`, `keypoints_xy` — is transformed *together* by the same shift/scale. `sv_detections_to_root_coordinates` and `scale_sv_detections` in `common/utils.py` are the canonical seam: both apply the shift/scale to `POLYGON_KEY_IN_SV_DETECTIONS` after `xyxy`/`mask` (polygon shift in root recovery #2473; polygon scale #1268). Failure mode: masks/polygons drawn or uploaded at the wrong location, invisible to a test that only checks `xyxy`.

2. **Response shape is a contract.** Block outputs match the manifest's declared keys on *every* branch, including empty/zero-area. Guard `len(detections) == 0` before any index-0 (`[0]`) access into metadata arrays. (`dynamic_crop`'s early return emitted `{"crops": None}` but omitted `"predictions"` — #2346.)

3. **Serialization round-trips the promised representation.** `serialise_sv_detections`/`serialise_rle_sv_detections`/`mask_to_polygon` in `common/serializers.py` are the response-shape contract: masks emit the polygon/RLE the schema promises with int rounding (`.astype(float).round().astype(int).tolist()`, #1236) and prefer a stored `POLYGON_KEY_IN_SV_DETECTIONS` over regenerating via `mask_to_polygon`; keypoints emit `class`, `class_id`, `confidence`, `x`, `y` per point. RLE ↔ dense mask abstraction lives in `inference_models/models/base/instance_segmentation.py` (`supported_mask_formats`, `coco_rle_masks_to_numpy_mask`); the `enforce_dense_masks_in_inference_models` toggle is a **manifest bool field** on the instance-segmentation v1/v2 blocks (`core_steps/models/roboflow/instance_segmentation/{v1,v2}.py`), threaded into the request — not an adapter function (#2384, #2260, #2484).

4. **Reference-backend parity.** ONNX / TRT / TorchScript paths and legacy-`inference` vs `inference_models` paths produce the same predictions within numerical tolerance. When one backend's post-processing changes, every sibling changes too — e.g. the keypoint slice offset fix touched `yolov8_key_points_detection_onnx.py`, `_trt.py`, `_torch_script.py` together (#1626). Failure mode: a model "works" but boxes/scores subtly differ per backend.

5. **Threshold / activation / index conventions justified against the reference.** New default threshold, sigmoid vs raw logits, background/class index offset, or tensor column slice must match the reference implementation and be config-gated if it shifts scores. The keypoint slice is fixed at `image_detections[:, 6:]` in `run_nms_for_key_points_detection` (`inference_models/models/common/roboflow/post_processing.py`) — not `5 + num_classes` (#1626). The YOLO-ultralytics mask default `mask_binarization_threshold` was set to `0.5` via env const `INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASK_BINARIZATION_THRESHOLD` (#2212, #2217); note `align_instance_segmentation_results`'s own `binarization_threshold` parameter default stays `0.0`. Class remapping / background-index off-by-one bit RF-DETR seg (#2075, #1619, #1920, #1590). Perspective anchor/extension math (#1234, #1287, #1310, #972). Preprocessing must not silently swap BGR/RGB or introduce PIL-vs-CV2 resize drift.

6. **Array shape/dtype discipline in `sv.Detections`.** Ragged object-dtype arrays break supervision's indexing/comparison. `add_inference_keypoints_to_sv_detections` in `common/utils.py` pads keypoints to fixed-shape numeric arrays (`padded_xy`/`padded_conf`/`padded_class_id`, uniform max length) rather than `dtype=object` (#2170).

7. **`supervision` bumps are load-bearing.** On any version change, verify annotators, `mask_to_polygons`, keypoint edge maps, and indexing still behave (#2467, #1725, #1424/#1425 pin history).

## Key files & Reference PRs
- `inference/core/workflows/core_steps/common/utils.py` — `sv_detections_to_root_coordinates`, `scale_sv_detections`, `attach_parent_coordinates_to_detections`, `add_inference_keypoints_to_sv_detections`. All geometry channels transformed together + keypoint padding (#2473, #1268, #2170).
- `inference/core/workflows/core_steps/common/serializers.py` — `serialise_sv_detections`, `serialise_rle_sv_detections`, `mask_to_polygon`. Response-shape/serialization contract (#1236).
- `inference/core/workflows/core_steps/fusion/detections_stitch/v1.py` — SAHI merge via `move_boxes`/`move_masks` + `OverlapFilter`. Pair with `transformations/dynamic_crop/v1.py` (`crop_image`, `WorkflowImageData.create_crop` in `execution_engine/entities/base.py`, origin-coordinate bookkeeping) (#2346).
- `inference_models/inference_models/models/base/instance_segmentation.py` — `supported_mask_formats`, `dense`/`rle` abstraction, `coco_rle_masks_to_numpy_mask`. The `enforce_dense_masks_in_inference_models` toggle is a manifest bool field on `core_steps/models/roboflow/instance_segmentation/{v1,v2}.py` selecting dense vs RLE (#2384, #2260).
- `inference_models/inference_models/models/common/roboflow/post_processing.py` and `inference_models/inference_models/models/yolov8/yolov8_instance_segmentation_{onnx,trt,torch_script}.py` — backend-parity reference: `run_nms_for_key_points_detection` keypoint slice `[:, 6:]` (#1626), `align_instance_segmentation_results` (`binarization_threshold` param default `0.0`; the `0.5` YOLO default comes from `INFERENCE_MODELS_YOLO_ULTRALYTICS_DEFAULT_MASK_BINARIZATION_THRESHOLD`, #2212). One backend changes → all siblings change.
- `inference/core/utils/rle_to_polygon.py` — `rle_masks_to_polygons`, COCO/uncompressed counts → polygon; compact-mask ↔ polygon reference.

## Severity guidance
- **Critical (BLOCK)** — silent geometry corruption or parity break shipping wrong results with green happy-path tests: a channel (mask/polygon/keypoints) in the wrong coordinate frame; a backend diverging in scores/classes; class/background-index off-by-one; serialization dropping or mislabeling `class`/`confidence`/`points`.
- **High (FLAG)** — a manifest-declared output key missing on a reachable branch; ragged object-dtype arrays that break supervision indexing; threshold/activation change altering scores without config gate or reference justification; RLE↔dense mismatch that breaks a consumer.
- **Medium (NIT / Not blocking)** — rounding within tolerance; empty-guard on an unreachable branch; supervision-bump cosmetic annotator drift; polygon simplification not affecting labels/boxes.
