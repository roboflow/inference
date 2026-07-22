
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Path Deviation` in version `v2`.

    - inputs: [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`GeoTag Detection`](geo_tag_detection.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections Filter`](detections_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Size Measurement`](size_measurement.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CSV Formatter`](csv_formatter.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Velocity`](velocity.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Path Deviation` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate path tracking state for different videos. Required for persistent path accumulation across frames..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path..
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Path Deviation` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`GeoTag Detection`](geo_tag_detection.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections Filter`](detections_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Size Measurement`](size_measurement.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CSV Formatter`](csv_formatter.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Velocity`](velocity.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Path Deviation` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `metadata` (*[`video_metadata`](../kinds/video_metadata.md)*): Video metadata containing video_identifier to maintain separate path tracking state for different videos. Required for persistent path accumulation across frames..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block tracks anchor point positions across frames to build object trajectories and compares them against the reference path. Output detections include path_deviation metadata containing the Fréchet distance from the reference path..
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

