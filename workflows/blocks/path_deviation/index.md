
# Path Deviation



## v2

??? "Class: `PathDeviationAnalyticsBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/path_deviation/v2.py">inference.core.workflows.core_steps.analytics.path_deviation.v2.PathDeviationAnalyticsBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Measure how closely tracked objects follow a reference path by calculating the Fréchet distance between the object's actual trajectory and the expected reference path, enabling path compliance monitoring, route deviation detection, quality control in automated systems, and behavioral analysis workflows.

## How This Block Works

This block compares the actual movement path of tracked objects against a predefined reference path to measure deviation. The block:

1. Receives tracked detection predictions with unique tracker IDs, an image with embedded video metadata, and a reference path definition
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts video_identifier to maintain separate path tracking state for different videos
   - Uses video metadata to initialize and manage path tracking state per video
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Initializes or retrieves path tracking state for the video:
   - Maintains a history of positions for each tracked object per video
   - Stores object paths using video_identifier to separate state for different videos
   - Creates new path tracking entries for objects appearing for the first time
5. Extracts anchor point coordinates for each detection:
   - Uses the triggering_anchor to determine which point on the bounding box to track (default: CENTER)
   - Gets the (x, y) coordinates of the anchor point for each detection in the current frame
   - The anchor point represents the position of the object used for path comparison
6. Accumulates object paths over time:
   - Appends each object's anchor point to its path history as frames are processed
   - Maintains separate path histories for each unique tracker_id
   - Builds complete trajectory paths by accumulating positions across all processed frames
7. Calculates Fréchet distance for each tracked object:
   - **Fréchet Distance**: Measures the similarity between two curves (paths) considering both location and ordering of points
   - Compares the object's accumulated path (actual trajectory) against the reference path (expected trajectory)
   - Uses dynamic programming to compute the minimum "leash length" required to traverse both paths simultaneously
   - Accounts for the order of points along each path, not just point-to-point distances
   - Lower values indicate the object follows the reference path closely, higher values indicate greater deviation
8. Stores path deviation in detection metadata:
   - Adds the Fréchet distance value to each detection's metadata
   - Each detection includes path_deviation representing how much it deviates from the reference path
   - Distance is measured in pixels (same units as image coordinates)
9. Maintains persistent path tracking:
   - Path histories accumulate across frames for the entire video
   - Each object's deviation is calculated based on its complete path from the start of tracking
   - Separate tracking state maintained for each video_identifier
10. Returns detections enhanced with path deviation information:
    - Outputs detection objects with added path_deviation metadata
    - Each detection now includes the Fréchet distance measuring its deviation from the reference path

The Fréchet distance is a metric that measures the similarity between two curves by finding the minimum length of a "leash" that connects a point moving along one curve to a point moving along the other curve, where both points move forward along their respective curves. Unlike simple Euclidean distance, Fréchet distance considers the ordering and continuity of points along paths, making it ideal for comparing trajectories where the sequence of movement matters. An object that follows the reference path exactly will have a Fréchet distance of 0, while objects that deviate significantly will have larger distances.

## Common Use Cases

- **Path Compliance Monitoring**: Monitor whether vehicles, robots, or objects follow predefined routes (e.g., verify vehicles stay in lanes, check robots follow programmed paths, ensure objects follow expected routes), enabling compliance monitoring workflows
- **Quality Control**: Detect deviations in manufacturing or assembly processes where objects should follow specific paths (e.g., detect conveyor belt deviations, monitor assembly line paths, check product movement patterns), enabling quality control workflows
- **Traffic Analysis**: Analyze vehicle movement patterns and detect lane departures or route deviations (e.g., detect vehicles leaving lanes, monitor route adherence, analyze traffic pattern compliance), enabling traffic analysis workflows
- **Security Monitoring**: Detect suspicious movement patterns or deviations from expected paths in security scenarios (e.g., detect unauthorized route deviations, monitor perimeter breach attempts, track movement compliance), enabling security monitoring workflows
- **Automated Systems**: Monitor and validate that automated systems (robots, AGVs, drones) follow expected paths correctly (e.g., verify robot navigation accuracy, check automated vehicle paths, validate drone flight paths), enabling automated system validation workflows
- **Behavioral Analysis**: Study movement patterns and path adherence in behavioral research (e.g., analyze animal movement patterns, study path following behavior, measure route preference deviations), enabling behavioral research workflows

## Connecting to Other Blocks

This block receives tracked detections, an image with embedded video metadata, and a reference path, and produces detections enhanced with path_deviation metadata:

- **After Byte Tracker blocks** to measure path deviation for tracked objects (e.g., measure tracked vehicle path compliance, analyze tracked person route adherence, monitor tracked object path deviations), enabling tracking-to-path-analysis workflows
- **After object detection or instance segmentation blocks** with tracking enabled to analyze movement paths (e.g., analyze vehicle paths, track object route compliance, measure path deviations), enabling detection-to-path-analysis workflows
- **Before visualization blocks** to display path deviation information (e.g., visualize paths and deviations, display reference and actual paths, show deviation metrics), enabling path deviation visualization workflows
- **Before logic blocks** like Continue If to make decisions based on path deviation thresholds (e.g., continue if deviation exceeds limit, filter based on path compliance, trigger actions on route violations), enabling path-based decision workflows
- **Before notification blocks** to alert on path deviations or compliance violations (e.g., alert on route deviations, notify on path compliance issues, trigger deviation-based alerts), enabling path-based notification workflows
- **Before data storage blocks** to record path deviation measurements (e.g., log path compliance data, store deviation statistics, record route adherence metrics), enabling path deviation data logging workflows

## Version Differences

**Enhanced from v1:**

- **Simplified Input**: Uses `image` input that contains embedded video metadata instead of requiring a separate `metadata` field, simplifying workflow connections and reducing input complexity
- **Improved Integration**: Better integration with image-based workflows since video metadata is accessed directly from the image object rather than requiring separate metadata input

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The reference path must be defined as a list of at least 2 points, where each point is a tuple or list of exactly 2 coordinates (x, y). The image's video_metadata should include video_identifier to maintain separate path tracking state for different videos. The block maintains persistent path tracking across frames for each video, accumulating complete trajectories, so it should be used in video workflows where frames are processed sequentially. For accurate path deviation measurement, detections should be provided consistently across frames with valid tracker IDs. The Fréchet distance is calculated in pixels (same units as image coordinates).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/path_deviation_analytics@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `triggering_anchor` | `str` | Point on the bounding box used to track object position for path calculation. Options include CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. This anchor point's coordinates are accumulated over frames to build the object's trajectory path, which is compared against the reference path using Fréchet distance.. | ✅ |
| `reference_path` | `List[Any]` | Expected reference path as a list of at least 2 points, where each point is a tuple or list of [x, y] coordinates. Example: [(100, 200), (200, 300), (300, 400)] defines a path with 3 points. The Fréchet distance measures how closely tracked objects follow this reference path. Points should be ordered along the expected trajectory.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Path Deviation` in version `v2`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Size Measurement`](size_measurement.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Path Deviation` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate path tracking state for different videos. Required for persistent path accumulation across frames..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the bounding box used to track object position for path calculation. Options include CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. This anchor point's coordinates are accumulated over frames to build the object's trajectory path, which is compared against the reference path using Fréchet distance..
        - `reference_path` (*[`list_of_values`](../kinds/list_of_values.md)*): Expected reference path as a list of at least 2 points, where each point is a tuple or list of [x, y] coordinates. Example: [(100, 200), (200, 300), (300, 400)] defines a path with 3 points. The Fréchet distance measures how closely tracked objects follow this reference path. Points should be ordered along the expected trajectory..

    - output
    
        - `path_deviation_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Path Deviation` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/path_deviation_analytics@v2",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "triggering_anchor": "CENTER",
	    "reference_path": [
	        [
	            100,
	            200
	        ],
	        [
	            200,
	            300
	        ],
	        [
	            300,
	            400
	        ]
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `PathDeviationAnalyticsBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/path_deviation/v1.py">inference.core.workflows.core_steps.analytics.path_deviation.v1.PathDeviationAnalyticsBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Measure how closely tracked objects follow a reference path by calculating the Fréchet distance between the object's actual trajectory and the expected reference path, enabling path compliance monitoring, route deviation detection, quality control in automated systems, and behavioral analysis workflows.

## How This Block Works

This block compares the actual movement path of tracked objects against a predefined reference path to measure deviation. The block:

1. Receives tracked detection predictions with unique tracker IDs, video metadata, and a reference path definition
2. Validates that detections have tracker IDs (required for tracking object movement across frames)
3. Initializes or retrieves path tracking state for the video:
   - Maintains a history of positions for each tracked object per video
   - Stores object paths using video_identifier to separate state for different videos
   - Creates new path tracking entries for objects appearing for the first time
4. Extracts anchor point coordinates for each detection:
   - Uses the triggering_anchor to determine which point on the bounding box to track (default: CENTER)
   - Gets the (x, y) coordinates of the anchor point for each detection in the current frame
   - The anchor point represents the position of the object used for path comparison
5. Accumulates object paths over time:
   - Appends each object's anchor point to its path history as frames are processed
   - Maintains separate path histories for each unique tracker_id
   - Builds complete trajectory paths by accumulating positions across all processed frames
6. Calculates Fréchet distance for each tracked object:
   - **Fréchet Distance**: Measures the similarity between two curves (paths) considering both location and ordering of points
   - Compares the object's accumulated path (actual trajectory) against the reference path (expected trajectory)
   - Uses dynamic programming to compute the minimum "leash length" required to traverse both paths simultaneously
   - Accounts for the order of points along each path, not just point-to-point distances
   - Lower values indicate the object follows the reference path closely, higher values indicate greater deviation
7. Stores path deviation in detection metadata:
   - Adds the Fréchet distance value to each detection's metadata
   - Each detection includes path_deviation representing how much it deviates from the reference path
   - Distance is measured in pixels (same units as image coordinates)
8. Maintains persistent path tracking:
   - Path histories accumulate across frames for the entire video
   - Each object's deviation is calculated based on its complete path from the start of tracking
   - Separate tracking state maintained for each video_identifier
9. Returns detections enhanced with path deviation information:
   - Outputs detection objects with added path_deviation metadata
   - Each detection now includes the Fréchet distance measuring its deviation from the reference path

The Fréchet distance is a metric that measures the similarity between two curves by finding the minimum length of a "leash" that connects a point moving along one curve to a point moving along the other curve, where both points move forward along their respective curves. Unlike simple Euclidean distance, Fréchet distance considers the ordering and continuity of points along paths, making it ideal for comparing trajectories where the sequence of movement matters. An object that follows the reference path exactly will have a Fréchet distance of 0, while objects that deviate significantly will have larger distances.

## Common Use Cases

- **Path Compliance Monitoring**: Monitor whether vehicles, robots, or objects follow predefined routes (e.g., verify vehicles stay in lanes, check robots follow programmed paths, ensure objects follow expected routes), enabling compliance monitoring workflows
- **Quality Control**: Detect deviations in manufacturing or assembly processes where objects should follow specific paths (e.g., detect conveyor belt deviations, monitor assembly line paths, check product movement patterns), enabling quality control workflows
- **Traffic Analysis**: Analyze vehicle movement patterns and detect lane departures or route deviations (e.g., detect vehicles leaving lanes, monitor route adherence, analyze traffic pattern compliance), enabling traffic analysis workflows
- **Security Monitoring**: Detect suspicious movement patterns or deviations from expected paths in security scenarios (e.g., detect unauthorized route deviations, monitor perimeter breach attempts, track movement compliance), enabling security monitoring workflows
- **Automated Systems**: Monitor and validate that automated systems (robots, AGVs, drones) follow expected paths correctly (e.g., verify robot navigation accuracy, check automated vehicle paths, validate drone flight paths), enabling automated system validation workflows
- **Behavioral Analysis**: Study movement patterns and path adherence in behavioral research (e.g., analyze animal movement patterns, study path following behavior, measure route preference deviations), enabling behavioral research workflows

## Connecting to Other Blocks

This block receives tracked detections, video metadata, and a reference path, and produces detections enhanced with path_deviation metadata:

- **After Byte Tracker blocks** to measure path deviation for tracked objects (e.g., measure tracked vehicle path compliance, analyze tracked person route adherence, monitor tracked object path deviations), enabling tracking-to-path-analysis workflows
- **After object detection or instance segmentation blocks** with tracking enabled to analyze movement paths (e.g., analyze vehicle paths, track object route compliance, measure path deviations), enabling detection-to-path-analysis workflows
- **Before visualization blocks** to display path deviation information (e.g., visualize paths and deviations, display reference and actual paths, show deviation metrics), enabling path deviation visualization workflows
- **Before logic blocks** like Continue If to make decisions based on path deviation thresholds (e.g., continue if deviation exceeds limit, filter based on path compliance, trigger actions on route violations), enabling path-based decision workflows
- **Before notification blocks** to alert on path deviations or compliance violations (e.g., alert on route deviations, notify on path compliance issues, trigger deviation-based alerts), enabling path-based notification workflows
- **Before data storage blocks** to record path deviation measurements (e.g., log path compliance data, store deviation statistics, record route adherence metrics), enabling path deviation data logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The reference path must be defined as a list of at least 2 points, where each point is a tuple or list of exactly 2 coordinates (x, y). The block requires video metadata with video_identifier to maintain separate path tracking state for different videos. The block maintains persistent path tracking across frames for each video, accumulating complete trajectories, so it should be used in video workflows where frames are processed sequentially. For accurate path deviation measurement, detections should be provided consistently across frames with valid tracker IDs. The Fréchet distance is calculated in pixels (same units as image coordinates).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/path_deviation_analytics@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `triggering_anchor` | `str` | Point on the bounding box used to track object position for path calculation. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. This anchor point's coordinates are accumulated over frames to build the object's trajectory path, which is compared against the reference path using Fréchet distance.. | ✅ |
| `reference_path` | `List[Any]` | Expected reference path as a list of at least 2 points, where each point is a tuple or list of [x, y] coordinates. Example: [(100, 200), (200, 300), (300, 400)] defines a path with 3 points. The Fréchet distance measures how closely tracked objects follow this reference path. Points should be ordered along the expected trajectory.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Path Deviation` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Size Measurement`](size_measurement.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Path Deviation` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `metadata` (*[`video_metadata`](../kinds/video_metadata.md)*): Video metadata containing video_identifier to maintain separate path tracking state for different videos. Required for persistent path accumulation across frames..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the bounding box used to track object position for path calculation. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. This anchor point's coordinates are accumulated over frames to build the object's trajectory path, which is compared against the reference path using Fréchet distance..
        - `reference_path` (*[`list_of_values`](../kinds/list_of_values.md)*): Expected reference path as a list of at least 2 points, where each point is a tuple or list of [x, y] coordinates. Example: [(100, 200), (200, 300), (300, 400)] defines a path with 3 points. The Fréchet distance measures how closely tracked objects follow this reference path. Points should be ordered along the expected trajectory..

    - output
    
        - `path_deviation_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Path Deviation` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/path_deviation_analytics@v1",
	    "metadata": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "triggering_anchor": "CENTER",
	    "reference_path": [
	        [
	            100,
	            200
	        ],
	        [
	            200,
	            300
	        ],
	        [
	            300,
	            400
	        ]
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

