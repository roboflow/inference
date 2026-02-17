# Wood Block Defect Edge Detection - Project Notes

## Problem Statement
A customer wants to know when a defect is close to the edge of a wood block. The camera is stationary and wood blocks move top-to-bottom through the frame. They have a trained defect detection model and need to determine if detected defects are near the wood block boundaries.

## Solution Approach
1. Use the existing defect detection model to find defects
2. Use classical CV (grayscale → blur → threshold → contours) to find wood block boundaries
3. Use a **new custom workflow block** to measure the distance from each defect to the nearest contour and flag those within a threshold

## What Was Built

### New Workflow Block: `detection_to_contour_distance`

**Type identifier:** `roboflow_core/detection_to_contour_distance@v1`

**Files created:**
- `inference/core/workflows/core_steps/classical_cv/detection_to_contour_distance/__init__.py` (empty)
- `inference/core/workflows/core_steps/classical_cv/detection_to_contour_distance/v1.py` (block implementation)
- `tests/workflows/unit_tests/core_steps/classical_cv/test_detection_to_contour_distance.py` (13 tests, all passing)

**File modified:**
- `inference/core/workflows/core_steps/loader.py` (import + registration of `DetectionToContourDistanceBlockV1`)

**Inputs:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | OBJECT_DETECTION_PREDICTION_KIND / INSTANCE_SEGMENTATION_PREDICTION_KIND | required | Defect detections from model |
| `contours` | CONTOURS_KIND | required | Contours from the contours block |
| `distance_threshold` | int | 50 | Pixel distance threshold for "close to edge" |

**Outputs:**
| Output | Type | Description |
|--------|------|-------------|
| `close_to_edge` | sv.Detections | Filtered detections within threshold distance of a contour |
| `distances` | List[float] | Min distance from each detection center to nearest contour |
| `all_detections_with_flag` | sv.Detections | All detections with `close_to_edge` boolean data field |

**How it works:**
- For each detection, computes bounding box center point
- Uses `cv2.pointPolygonTest(contour, center, measureDist=True)` to get signed distance to each contour
- Takes `abs(distance)` (distance to edge regardless of inside/outside)
- Keeps minimum distance across all contours
- Flags detections where min distance <= threshold

### Workflow JSON

**File:** `wood_block_defect_edge_workflow.json` (in inference project root)

**Pipeline:**
```
Image → Defect Model ──────────────────────┐
  │                                         ▼
  └→ Grayscale → Blur → Threshold → Contours → Detection to Contour Distance
```

**Configurable inputs:**
- `distance_threshold` (default 50px) - what counts as "close to edge"
- `confidence_threshold` (default 0.4) - defect model confidence
- `model_id` - **MUST BE REPLACED** with the customer's actual model ID (currently `YOUR-DEFECT-MODEL-ID/1`)

**Tuning notes:**
- `thresh_value: 127` may need adjustment depending on wood/background contrast
- Consider `"threshold_type": "otsu"` if lighting varies across frames
- The blur step helps suppress wood grain texture before thresholding
- `distance_threshold` depends on image resolution and customer's definition of "close"

## Tests

Run tests with:
```bash
cd /Users/jenniferkuchta/github/inference
python -m pytest tests/workflows/unit_tests/core_steps/classical_cv/test_detection_to_contour_distance.py -v
```

13 tests covering: manifest validation, empty inputs, detection far/close/inside/on contours, mixed detections, custom thresholds.

## TODO / Next Steps
- [ ] Replace `YOUR-DEFECT-MODEL-ID/1` with the customer's actual model ID
- [ ] Test the workflow end-to-end with real wood block images
- [ ] Tune threshold values (thresh_value for binarization, distance_threshold for proximity) based on actual camera setup
- [ ] Consider whether Otsu's thresholding works better than fixed threshold for variable lighting
- [ ] Consider adding a visualization step to the workflow (e.g., draw only edge-defects on the image)
- [ ] Decide if this block should be included in a PR to the inference repo
- [ ] No git commit has been made yet - all changes are uncommitted
