---
name: review-topic-prediction-integrity
description: Load when a PR touches bounding boxes, masks (binary/compact/RLE), keypoints, coordinate transforms (crop/stitch/perspective/parent-root recovery), image pre- or post-processing (resize/BGR-RGB/normalization), NMS, or prediction serialization / response shape. Guards prediction, coordinate, and pre/post-processing integrity — geometry that lands in the wrong place, masks/scores that silently drift, and response fields that break downstream consumers.
---

# Review topic: Prediction, coordinate & pre/post-processing integrity

## When this applies
Load this skill when a diff does ANY of the following (content trigger, not one directory):
- Edits coordinate transforms or recovery: `sv_detections_to_root_coordinates`, `scale_sv_detections`, `attach_parent(s)_coordinates*`, `move_boxes`/`move_masks`, or any `xyxy +=` / `* scale` / shift arithmetic (`inference/core/workflows/core_steps/common/utils.py`).
- Touches crop / stitch / slice / perspective blocks: `transformations/dynamic_crop`, `transformations/stitch_images`, `transformations/stitch_ocr_detections`, `transformations/perspective_correction`, `transformations/dynamic_zones`, `fusion/detections_stitch`.
- Changes mask representation: binary mask ↔ polygon ↔ RLE, `mask_to_polygon(s)`, `supported_mask_formats`, `enforce_dense_masks`, `mask_binarization_threshold`, `rle_to_polygon`.
- Touches keypoints: parsing into `sv.Detections`, keypoint xy/confidence/class arrays, keypoint slice offsets in model post-processing, keypoint visualization/edges.
- Edits model pre/post-processing in `inference_models/` or `inference_experimental/`: normalization, resize/letterbox, BGR/RGB, sigmoid/softmax, NMS params, class/background-index handling, tensor slice layout.
- Changes prediction serialization / response shape: `serializers.py`, `deserializers.py`, prediction entity models, block `outputs`/manifest keys.
- Bumps `supervision` — its internals (indexing, `mask_to_polygons`, annotators, edge maps) are load-bearing here.

## What to protect
- **Coordinate frames stay consistent.** Every geometry channel of a detection — `xyxy`, `mask`, `data[polygon]`, `keypoints_xy` — must be transformed *together* by the same shift/scale. Recovering to root coordinates and forgetting one channel puts boxes and masks in different frames. Failure mode: masks/polygons drawn or uploaded at the wrong location, silent and invisible in the happy-path test that only checks `xyxy`.
- **Reference-backend parity.** ONNX / TRT / TorchScript / PyTorch paths and the legacy-`inference` vs `inference_models` paths must produce the *same* predictions (within numerical tolerance). Threshold, sigmoid, slice-offset, or index-convention drift changes scores/classes for real users. Failure mode: a model "works" but boxes/scores subtly differ per backend.
- **Response shape is a contract.** Block outputs must match the manifest's declared keys on *every* branch (including empty/zero-area). Serialized predictions must keep the field names, types, and rounding that downstream SDK/UI expect (`predictions`, `points`, `keypoints`, `mask`/`rle`, `class`, `confidence`).
- **Array shape/dtype discipline in `sv.Detections`.** Ragged object-dtype arrays break supervision's indexing/comparison; padding + fixed dtype keeps them valid.

## What to check
1. **All geometry channels move together.** If a transform edits `xyxy`, confirm it *also* handles `mask`, `data[POLYGON_KEY_IN_SV_DETECTIONS]`, and `keypoints_xy` when present — with the *same* shift/scale. This is exactly what PR #2473 (polygon shift in root recovery) and #1268 (polygon scale) added after `xyxy`/mask were already handled.
2. **Empty / zero-area / no-detections branches.** Does the early-return path still emit every manifest-declared output key? (dynamic_crop returned `{"crops": None}` but omitted `"predictions"` — PR #2346.) Does `len(detections) == 0` short-circuit before index-0 access (`[0]`) into metadata arrays?
3. **Round-trip serialization.** For masks: is the output polygon/RLE the same representation the response schema promises, with correct rounding (`.round().astype(int).tolist()`, PR #1236) and vertex source (stored polygon preferred over `mask_to_polygons`)? For keypoints: are `class`, `class_id`, `confidence`, `x`, `y` all present per keypoint?
4. **Backend parity for model changes.** If post-processing changes in one backend file, verify the *same* change is applied to every sibling backend (onnx/trt/torch_script/pytorch). PR #1626 fixed a keypoint slice offset across four YOLOv8 backend files at once.
5. **Threshold / activation / index conventions.** Any new default threshold, sigmoid vs raw logits, background/class index offset, or tensor column slice (`[:, 6:]` vs `[:, 5+num_classes:]`) must be justified against the reference implementation and made configurable if it shifts scores (mask binarization default `0.0`→`0.5` to match old inference — PR #2212).
6. **BGR/RGB and PIL-vs-CV2.** New preprocessing must not silently swap channel order or introduce PIL/CV2 resize numerical drift versus the reference decode path.
7. **dtype / shape.** New `sv.Detections` data arrays: are they fixed-shape numeric arrays, not ragged `dtype=object` (PR #2170)? Are keypoints padded to uniform length?
8. **supervision bumps.** On any `supervision` version change, check annotators, `mask_to_polygons`, keypoint edge maps, and indexing still behave (PRs #2467, #1725, #1424/#1425 pin history).

## Common failure modes
- Root/parent-coordinate recovery updates `xyxy` and `mask` but forgets `polygon` or `keypoints_xy` → geometry lands in mixed frames (#2473, #1268).
- Block early-return omits a manifest-declared output key → downstream steps get `KeyError`/`None` mismatch (#2346).
- Object-dtype ragged keypoint arrays → break supervision `is_data_equal` / indexing (#2170).
- Mask serialized via `mask_to_polygons` when a precise stored polygon exists, or missing int rounding → wrong/lossy polygons in response (#1236).
- Keypoint tensor slice offset assumed `5 + num_classes` when layout is fixed at `6:`, or `-1` reshape instead of explicit slot count → keypoints scrambled across backends (#1626).
- Post-processing threshold / sigmoid divergence from reference inference → masks bleed at edges, scores shift (#2212, #2217).
- Class remapping / background-index off-by-one in RF-DETR seg → wrong class labels (#2075, #1619, #1920, #1590).
- Perspective correction anchor/extension math wrong → warped detections misplaced (#1234, #1287, #1310, #972).
- Dense vs RLE mask format not honored per-platform → memory blowup or serialization mismatch (#2384, #2260, #2484).

## Example implementations (point here)
- `inference/core/workflows/core_steps/common/utils.py` — `sv_detections_to_root_coordinates`, `scale_sv_detections`, `attach_parent_coordinates_to_detections`, `add_inference_keypoints_to_sv_detections`. The canonical place where *all* geometry channels (xyxy + mask + polygon + keypoints) must be transformed together and keypoints padded to fixed dtype. Contract established/repaired by PRs #2473, #1268, #2170.
- `inference/core/workflows/core_steps/common/serializers.py` — `serialise_sv_detections`, `serialise_rle_sv_detections`, `mask_to_polygon`. Canonical response-shape/serialization contract (polygon vertex source, int rounding, keypoint fields, RLE counts decode). PR #1236.
- `inference/core/workflows/core_steps/fusion/detections_stitch/v1.py` — SAHI merge via `move_boxes`/`move_masks` + `OverlapFilter`; correct slice→original coordinate reconstruction. Pair with `transformations/dynamic_crop/v1.py` (`crop_image`, `WorkflowImageData.create_crop`, origin-coordinate bookkeeping) — PR #2346.
- `inference_models/inference_models/models/base/instance_segmentation.py` — `supported_mask_formats`, `dense`/`rle` mask abstraction, `coco_rle_masks_to_numpy_mask`; the mask-format seam. Adapter side: `inference/core/models/inference_models_adapters.py` (`enforce_dense_masks`) — PR #2384. RLE across models — PR #2260.
- `inference_experimental/inference_exp/models/common/roboflow/post_processing.py` and `inference_models/.../models/yolov8/yolov8_instance_segmentation_*.py` (onnx/trt/torch_script) — reference for backend-parity post-processing (keypoint slice offsets #1626, mask binarization threshold #2212). When one backend changes, all siblings must.
- `inference/core/utils/rle_to_polygon.py` — `rle_masks_to_polygons`, COCO/uncompressed counts → polygon; the compact-mask ↔ polygon conversion reference.

## Severity guidance
- **Critical** — silent geometry corruption or parity break that ships wrong results with green happy-path tests: a channel (mask/polygon/keypoints) left in the wrong coordinate frame; a backend diverging from the reference in scores/classes; class/background-index off-by-one; serialization dropping or mislabeling `class`/`confidence`/`points`.
- **High** — a manifest-declared output key missing on some branch; ragged object-dtype arrays that break supervision indexing; threshold/activation change altering scores without config gate or reference justification; RLE↔dense mismatch that breaks a consumer.
- **Medium** — rounding/precision drift within tolerance; missing empty/zero-area guard that is currently unreachable; supervision-bump cosmetic annotator drift; polygon simplification that loses vertices without affecting labels/boxes.
