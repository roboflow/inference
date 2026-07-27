
# Nearest Neighbor Detection Match



??? "Class: `DetectionsNearestNeighborBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/detections_nearest_neighbor/v1.py">inference.core.workflows.core_steps.classical_cv.detections_nearest_neighbor.v1.DetectionsNearestNeighborBlockV1</a>
    



Perform a nearest-neighbor spatial join between two detection sets by finding, for each detection in a query set, the closest detection (or detections, on a tie) in a target set, using plain 2D Euclidean pixel distance between configurable anchor points.

## How This Block Works

This block matches each detection in a query set to its nearest neighbor(s) in a target set. The block:

1. Receives query predictions and target predictions (object detection, instance segmentation, or keypoint detection) - the same set can be passed as both query and target
2. Resolves an anchor point for every detection in each set, based on the selected `query_point` and `target_point` options: the center, one of the four corners, one of the four edge midpoints of the bounding box, or a named keypoint (requires keypoint detection predictions and `query_keypoint_name`/`target_keypoint_name`)
3. Computes a full pairwise pixel-distance matrix between every query anchor point and every target anchor point (brute-force, not a spatial tree - appropriate for the tens-of-detections-per-frame scale this block targets)
4. For each query detection, excludes any target detection that shares its `detection_id` (self-match exclusion), so passing the same detections as both query and target does not match a detection against itself
5. If `max_distance` is set, discards any remaining candidate target farther than `max_distance` pixels from the query anchor point before ranking - a target beyond the limit is treated the same as an excluded self-match
6. Finds the minimum distance for each query detection, then includes every remaining target detection whose distance is within a fixed 1-pixel epsilon of that minimum, so exact or near-exact ties produce multiple matches instead of an arbitrary single winner
7. Attaches the nearest distance (in pixels) to each query detection's metadata; detections with no eligible target (empty target set, every target excluded as a self-match, or every remaining target farther than `max_distance`) get `None`
8. Returns three flat, aligned outputs: the enriched query predictions, and two same-length/same-order detection sets - `matched_query_detections` and `matched_target_detections` - with one row per query-target pairing. A query detection with a tie appears once per tied match (its row is duplicated); a query detection with no valid match is omitted from both, and only shows up in `query_predictions` (with `nearest_target_distance` set to `None`)

Distance is always plain, uncalibrated 2D Euclidean pixel distance - there is no real-world unit conversion, unlike the Distance Measurement block. This keeps the block simple for use cases that only need relative/pixel-space proximity.

## Common Use Cases

- **Zone/Anchor Assignment**: Assign each detected object to its nearest reference point or anchor detection (e.g. match people to the nearest exit sign, match vehicles to the nearest parking spot marker), enabling proximity-based assignment workflows
- **Cross-Class Association**: Associate detections of one class with the nearest detection of another class (e.g. match detected hands to the nearest detected tool, match players to the nearest ball detection), enabling relationship analysis between different object types
- **Duplicate/Overlap Analysis**: Compare a detection set against itself to find each detection's closest neighbor (e.g. find the closest other person in a crowd, identify tightly clustered detections), enabling density and clustering analysis
- **Tracking Handoff**: Match detections between two model outputs on the same frame (e.g. match a fast general detector's output to a slower specialist model's output) to combine results, enabling multi-model fusion workflows

## Connecting to Other Blocks

This block receives two detection sets and produces the enriched query predictions plus two flat, aligned detection sets representing every query-target pairing - no dimensionality tricks or intermediate merge step required, since `matched_query_detections` and `matched_target_detections` are ordinary detection sets of the same shape any detection-producing block returns:

- **After object detection, instance segmentation, or keypoint detection model blocks** to build the query and target sets to match, enabling detection-to-matching workflows
- **Before logic blocks** like Continue If or Detections Filter to make decisions based on `nearest_target_distance` (e.g. continue if the nearest match is within a threshold distance) or to filter `matched_query_detections`/`matched_target_detections` directly, enabling proximity-based decision workflows
- **Before visualization blocks** like Label Visualization or Bounding Box Visualization to display the enriched `query_predictions`, or `matched_query_detections`/`matched_target_detections` side by side to draw matched pairs, enabling distance-annotated visualizations
- **Before data storage blocks** like CSV Formatter to export every query-target pairing as a row, enabling matched-pair logging workflows

## Requirements

This block requires two sets of detection predictions (object detection, instance segmentation, or keypoint detection); the same set can be used for both `query_predictions` and `target_predictions`. To use the `KEYPOINT` anchor option for either set, that set must be keypoint detection predictions and the corresponding `query_keypoint_name`/`target_keypoint_name` must be provided. Self-match exclusion relies on `detection_id` being present on both sets - this is populated automatically for all Roboflow object detection, instance segmentation, and keypoint detection model blocks. `max_distance` is optional; leave it unset to match every query detection to its nearest target regardless of distance.

Note that `query_predictions` is enriched in place - the same `sv.Detections` object passed in is mutated (a new `nearest_target_distance` field is added to its `.data`) and returned, the same convention used by blocks like Velocity and Time in Zone. Avoid feeding the same selector into two independent branches of a workflow if each branch needs to see its own, unmodified `nearest_target_distance`.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_nearest_neighbor@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `query_point` | `str` | Anchor point used to represent each query detection's location: CENTER, a corner (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), an edge midpoint (TOP_CENTER, BOTTOM_CENTER, CENTER_LEFT, CENTER_RIGHT), or KEYPOINT (requires `query_predictions` to be keypoint detection predictions and `query_keypoint_name` to be set).. | ✅ |
| `target_point` | `str` | Anchor point used to represent each target detection's location. Same options as `query_point`, set independently. Use KEYPOINT to match against a named keypoint on `target_predictions` (requires `target_predictions` to be keypoint detection predictions and `target_keypoint_name` to be set).. | ✅ |
| `query_keypoint_name` | `str` | Name of the keypoint class to use as the query anchor point. Required only when `query_point` is set to KEYPOINT; `query_predictions` must then be keypoint detection predictions.. | ✅ |
| `target_keypoint_name` | `str` | Name of the keypoint class to use as the target anchor point. Required only when `target_point` is set to KEYPOINT; `target_predictions` must then be keypoint detection predictions.. | ✅ |
| `max_distance` | `int` | Maximum allowed pixel distance between a query detection and a candidate target. Targets farther than this are excluded before ranking, the same way self-matches are excluded. A query detection whose nearest remaining target is still farther than this (or has no remaining target at all) gets `nearest_target_distance = None` and is omitted from `matched_query_detections`/`matched_target_detections`. Leave unset for no limit.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Nearest Neighbor Detection Match` in version `v1`.

    - inputs: [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Template Matching`](template_matching.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM 3`](sam3.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`Cosmos 3`](cosmos3.md), [`Overlap Filter`](overlap_filter.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Detection Offset`](detection_offset.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Gaze Detection`](gaze_detection.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Email Notification`](email_notification.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Detections Filter`](detections_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Combine`](detections_combine.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Byte Tracker`](byte_tracker.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Merge`](detections_merge.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Halo Visualization`](halo_visualization.md), [`Frame Delay`](frame_delay.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Time in Zone`](timein_zone.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Path Deviation`](path_deviation.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Filter`](detections_filter.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Merge`](detections_merge.md), [`Line Counter`](line_counter.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Nearest Neighbor Detection Match` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `query_predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Detections to find nearest-neighbor matches for. Each detection in this set is matched against `target_predictions`. The same selector can be used for both `query_predictions` and `target_predictions` (e.g. to find each detection's closest neighbor within one set) - a detection is never matched against itself..
        - `target_predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Detections to search for the nearest match within, for every detection in `query_predictions`. Can be the same selector as `query_predictions`..
        - `query_point` (*[`string`](../kinds/string.md)*): Anchor point used to represent each query detection's location: CENTER, a corner (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), an edge midpoint (TOP_CENTER, BOTTOM_CENTER, CENTER_LEFT, CENTER_RIGHT), or KEYPOINT (requires `query_predictions` to be keypoint detection predictions and `query_keypoint_name` to be set)..
        - `target_point` (*[`string`](../kinds/string.md)*): Anchor point used to represent each target detection's location. Same options as `query_point`, set independently. Use KEYPOINT to match against a named keypoint on `target_predictions` (requires `target_predictions` to be keypoint detection predictions and `target_keypoint_name` to be set)..
        - `query_keypoint_name` (*[`string`](../kinds/string.md)*): Name of the keypoint class to use as the query anchor point. Required only when `query_point` is set to KEYPOINT; `query_predictions` must then be keypoint detection predictions..
        - `target_keypoint_name` (*[`string`](../kinds/string.md)*): Name of the keypoint class to use as the target anchor point. Required only when `target_point` is set to KEYPOINT; `target_predictions` must then be keypoint detection predictions..
        - `max_distance` (*[`integer`](../kinds/integer.md)*): Maximum allowed pixel distance between a query detection and a candidate target. Targets farther than this are excluded before ranking, the same way self-matches are excluded. A query detection whose nearest remaining target is still farther than this (or has no remaining target at all) gets `nearest_target_distance = None` and is omitted from `matched_query_detections`/`matched_target_detections`. Leave unset for no limit..

    - output
    
        - `query_predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.
        - `matched_query_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.
        - `matched_target_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Nearest Neighbor Detection Match` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_nearest_neighbor@v1",
	    "query_predictions": "$steps.model.predictions",
	    "target_predictions": "$steps.model.predictions",
	    "query_point": "CENTER",
	    "target_point": "CENTER",
	    "query_keypoint_name": "left_shoulder",
	    "target_keypoint_name": "left_shoulder",
	    "max_distance": 50
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

