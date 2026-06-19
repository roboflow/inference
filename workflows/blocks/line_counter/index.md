
# Line Counter



## v2

??? "Class: `LineCounterBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/line_counter/v2.py">inference.core.workflows.core_steps.analytics.line_counter.v2.LineCounterBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Count objects crossing a defined line segment in video using tracked detections, maintaining separate counts for objects crossing in opposite directions (in and out), and outputting both count values and the actual detection objects that crossed the line for traffic analysis, people counting, entry/exit monitoring, and directional flow measurement workflows.

## How This Block Works

This block counts objects that cross a line segment by tracking their movement across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts video_identifier to maintain separate counting state for different videos
   - Uses video metadata to initialize and manage line zone state per video
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Initializes or retrieves a line zone for the video:
   - Creates a LineZone from two coordinate points defining the line segment
   - Configures triggering anchor point if specified (optional - if not specified, uses default anchor behavior)
   - Stores line zone configuration per video using video_identifier
   - Maintains separate counting state for each video
5. Monitors object positions across frames:
   - Tracks each object's position using its unique tracker_id
   - Detects when an object's triggering anchor point (if specified) or default anchor crosses the line
   - Determines crossing direction based on which side of the line the object approaches from
6. Counts line crossings:
   - **In Direction**: Objects crossing the line in one direction increment the count_in counter
   - **Out Direction**: Objects crossing the line in the opposite direction increment the count_out counter
   - Each unique tracker_id is counted only once per crossing (prevents duplicate counting if object oscillates near line)
7. Identifies crossing detections:
   - Creates masks identifying which detections crossed in each direction in the current frame
   - Filters detections to separate those that crossed "in" from those that crossed "out"
   - Returns the actual detection objects (not just counts) for further processing
8. Maintains persistent counting state:
   - Counts accumulate across frames for the entire video
   - State persists for each video until workflow execution completes
   - Separate counters for each unique video_identifier
9. Returns four outputs:
   - **count_in**: Total number of objects that crossed the line in the "in" direction (cumulative across video)
   - **count_out**: Total number of objects that crossed the line in the "out" direction (cumulative across video)
   - **detections_in**: Detection objects that crossed the line in the "in" direction (current frame crossings)
   - **detections_out**: Detection objects that crossed the line in the "out" direction (current frame crossings)

The line segment defines a virtual boundary in the video frame. The direction (in/out) is determined by which side of the line objects approach from - for a horizontal line, objects coming from above might count as "in" while objects from below count as "out" (or vice versa, depending on line orientation). The triggering anchor (if specified) determines which point on the bounding box must cross the line for the crossing to be counted - if not specified, the line zone uses its default anchor behavior. The count outputs provide cumulative totals across the video, while the detection outputs provide the actual objects that crossed in the current frame, enabling further analysis or visualization of crossing events.

## Common Use Cases

- **People Counting**: Count people entering and exiting buildings, stores, or events (e.g., count visitors entering store, track people entering/exiting building, monitor event attendance), enabling entry/exit counting workflows
- **Traffic Analysis**: Count vehicles passing through intersections or road segments (e.g., count vehicles crossing intersection, track traffic flow in specific directions, monitor vehicle passage at checkpoints), enabling traffic flow analysis workflows
- **Retail Analytics**: Track customer movement and foot traffic in retail spaces (e.g., count customers entering store sections, track movement between departments, monitor shopping flow patterns), enabling retail foot traffic analytics workflows
- **Security Monitoring**: Monitor entry and exit at secure areas or checkpoints (e.g., track entries to restricted areas, count people at access points, monitor checkpoint crossings), enabling security access monitoring workflows
- **Occupancy Management**: Track occupancy changes by counting objects entering and leaving spaces (e.g., count entries/exits to manage room capacity, track vehicle arrivals/departures in parking, monitor space occupancy changes), enabling occupancy tracking workflows
- **Wildlife Monitoring**: Count animals crossing defined paths or boundaries (e.g., track animal migration patterns, count wildlife crossing roads, monitor animal movement in habitats), enabling wildlife behavior analysis workflows

## Connecting to Other Blocks

This block receives tracked detections and an image with embedded video metadata, and produces count_in, count_out, detections_in, and detections_out:

- **After Byte Tracker blocks** to count tracked objects crossing lines (e.g., count tracked people crossing line, track vehicle crossings with consistent IDs, monitor tracked object movements), enabling tracking-to-counting workflows
- **After object detection or instance segmentation blocks** with tracking enabled to count detected objects (e.g., count detected vehicles, track people crossings, monitor object movements), enabling detection-to-counting workflows
- **Using detections_in or detections_out outputs** to process or visualize objects that crossed the line (e.g., visualize objects that crossed, analyze crossing objects, filter for crossing events), enabling crossing object analysis workflows
- **Before visualization blocks** to display line counter information and crossing objects (e.g., visualize line and counts, display crossing statistics, show crossing objects with annotations), enabling counting visualization workflows
- **Before data storage blocks** to record counting data and crossing events (e.g., log entry/exit counts, store traffic statistics, record crossing objects with metadata), enabling counting data logging workflows
- **Before notification blocks** to alert on count thresholds or crossing events (e.g., alert when count exceeds limit, notify on specific object crossings, trigger actions based on counts), enabling count-based notification workflows

## Version Differences

**Enhanced from v1:**

- **Detection Outputs**: Adds two new outputs (`detections_in` and `detections_out`) that provide the actual detection objects that crossed the line in each direction, not just count totals, enabling downstream processing and visualization of crossing objects
- **Simplified Input**: Uses `image` input that contains embedded video metadata instead of requiring a separate `metadata` field, simplifying workflow connections and reducing input complexity
- **Optional Triggering Anchor**: Makes `triggering_anchor` optional (default None) instead of required, allowing the line zone to use its default anchor behavior when no specific anchor is needed
- **Improved Integration**: Better integration with image-based workflows since video metadata is accessed directly from the image object rather than requiring separate metadata input

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The line must be defined as a list of exactly 2 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The image's video_metadata should include video_identifier to maintain separate counting state for different videos. The block maintains persistent counting state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate counting, detections should be provided consistently across frames with valid tracker IDs.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/line_counter@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `line_segment` | `List[Any]` | Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from.. | ✅ |
| `triggering_anchor` | `str` | Optional point on the bounding box that must cross the line for counting. If not specified (None), the line zone uses its default anchor behavior. Options when specified: CENTER, BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. Specifying CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Line Counter` in version `v2`.

    - inputs: [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Path Deviation`](path_deviation.md), [`Email Notification`](email_notification.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`OCR Model`](ocr_model.md), [`Velocity`](velocity.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`EasyOCR`](easy_ocr.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`CSV Formatter`](csv_formatter.md), [`SAM 3`](sam3.md), [`Webhook Sink`](webhook_sink.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Seg Preview`](seg_preview.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Transformation`](detections_transformation.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`Qwen-VL`](qwen_vl.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Florence-2 Model`](florence2_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`Overlap Filter`](overlap_filter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Image Stack`](image_stack.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Trace Visualization`](trace_visualization.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Camera Focus`](camera_focus.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`Distance Measurement`](distance_measurement.md), [`SORT Tracker`](sort_tracker.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter`](line_counter.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Path Deviation`](path_deviation.md), [`Email Notification`](email_notification.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Image Slicer`](image_slicer.md), [`Dominant Color`](dominant_color.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Background Subtraction`](background_subtraction.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Line Counter` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate counting state for different videos. Required for persistent counting across frames..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Objects are counted when their triggering anchor point (if specified) crosses the line segment. The detections_in and detections_out outputs provide the actual detection objects that crossed in each direction..
        - `line_segment` (*[`list_of_values`](../kinds/list_of_values.md)*): Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Optional point on the bounding box that must cross the line for counting. If not specified (None), the line zone uses its default anchor behavior. Options when specified: CENTER, BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. Specifying CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line..

    - output
    
        - `count_in` ([`integer`](../kinds/integer.md)): Integer value.
        - `count_out` ([`integer`](../kinds/integer.md)): Integer value.
        - `detections_in` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.
        - `detections_out` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Line Counter` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/line_counter@v2",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "line_segment": [
	        [
	            0,
	            50
	        ],
	        [
	            500,
	            50
	        ]
	    ],
	    "triggering_anchor": "CENTER"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `LineCounterBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/line_counter/v1.py">inference.core.workflows.core_steps.analytics.line_counter.v1.LineCounterBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Count objects crossing a defined line segment in video using tracked detections, maintaining separate counts for objects crossing in opposite directions (in and out) for traffic analysis, people counting, entry/exit monitoring, and directional flow measurement workflows.

## How This Block Works

This block counts objects that cross a line segment by tracking their movement across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and video metadata
2. Validates that detections have tracker IDs (required for tracking object movement across frames)
3. Initializes or retrieves a line zone for the video:
   - Creates a LineZone from two coordinate points defining the line segment
   - Stores line zone configuration per video using video_identifier
   - Maintains separate counting state for each video
4. Monitors object positions across frames:
   - Tracks each object's position using its unique tracker_id
   - Detects when an object's triggering anchor point (default: CENTER of bounding box) crosses the line
   - Determines crossing direction based on which side of the line the object approaches from
5. Counts line crossings:
   - **In Direction**: Objects crossing the line in one direction increment the count_in counter
   - **Out Direction**: Objects crossing the line in the opposite direction increment the count_out counter
   - Each unique tracker_id is counted only once per crossing (prevents duplicate counting if object oscillates near line)
6. Maintains persistent counting state:
   - Counts accumulate across frames for the entire video
   - State persists for each video until workflow execution completes
   - Separate counters for each unique video_identifier
7. Returns two count values:
   - **count_in**: Total number of objects that crossed the line in the "in" direction
   - **count_out**: Total number of objects that crossed the line in the "out" direction

The line segment defines a virtual boundary in the video frame. The direction (in/out) is determined by which side of the line objects approach from - for a horizontal line, objects coming from above might count as "in" while objects from below count as "out" (or vice versa, depending on line orientation). The triggering anchor determines which point on the bounding box must cross the line for the crossing to be counted - using CENTER ensures the object is substantially across the line before counting.

## Common Use Cases

- **People Counting**: Count people entering and exiting buildings, stores, or events (e.g., count visitors entering store, track people entering/exiting building, monitor event attendance), enabling entry/exit counting workflows
- **Traffic Analysis**: Count vehicles passing through intersections or road segments (e.g., count vehicles crossing intersection, track traffic flow in specific directions, monitor vehicle passage at checkpoints), enabling traffic flow analysis workflows
- **Retail Analytics**: Track customer movement and foot traffic in retail spaces (e.g., count customers entering store sections, track movement between departments, monitor shopping flow patterns), enabling retail foot traffic analytics workflows
- **Security Monitoring**: Monitor entry and exit at secure areas or checkpoints (e.g., track entries to restricted areas, count people at access points, monitor checkpoint crossings), enabling security access monitoring workflows
- **Occupancy Management**: Track occupancy changes by counting objects entering and leaving spaces (e.g., count entries/exits to manage room capacity, track vehicle arrivals/departures in parking, monitor space occupancy changes), enabling occupancy tracking workflows
- **Wildlife Monitoring**: Count animals crossing defined paths or boundaries (e.g., track animal migration patterns, count wildlife crossing roads, monitor animal movement in habitats), enabling wildlife behavior analysis workflows

## Connecting to Other Blocks

This block receives tracked detections and video metadata, and produces count_in and count_out values:

- **After Byte Tracker blocks** to count tracked objects crossing lines (e.g., count tracked people crossing line, track vehicle crossings with consistent IDs, monitor tracked object movements), enabling tracking-to-counting workflows
- **After object detection or instance segmentation blocks** with tracking enabled to count detected objects (e.g., count detected vehicles, track people crossings, monitor object movements), enabling detection-to-counting workflows
- **Before visualization blocks** to display line counter information (e.g., visualize line and counts, display crossing statistics, show counting results), enabling counting visualization workflows
- **Before data storage blocks** to record counting data (e.g., log entry/exit counts, store traffic statistics, record occupancy metrics), enabling counting data logging workflows
- **Before notification blocks** to alert on count thresholds or events (e.g., alert when count exceeds limit, notify on occupancy changes, trigger actions based on counts), enabling count-based notification workflows
- **Before analysis blocks** to process counting metrics (e.g., analyze traffic patterns, process occupancy data, work with counting statistics), enabling counting analysis workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The line must be defined as a list of exactly 2 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The block requires video metadata with video_identifier to maintain separate counting state for different videos. The block maintains persistent counting state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate counting, detections should be provided consistently across frames with valid tracker IDs.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/line_counter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `line_segment` | `List[Any]` | Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from.. | ✅ |
| `triggering_anchor` | `str` | Point on the bounding box that must cross the line for counting. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Line Counter` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Path Deviation`](path_deviation.md), [`Email Notification`](email_notification.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`OCR Model`](ocr_model.md), [`Velocity`](velocity.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`EasyOCR`](easy_ocr.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`CSV Formatter`](csv_formatter.md), [`SAM 3`](sam3.md), [`Webhook Sink`](webhook_sink.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Seg Preview`](seg_preview.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Transformation`](detections_transformation.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`Qwen-VL`](qwen_vl.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Florence-2 Model`](florence2_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Image Blur`](image_blur.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Image Stack`](image_stack.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Trace Visualization`](trace_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Image Slicer`](image_slicer.md), [`Dominant Color`](dominant_color.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Background Subtraction`](background_subtraction.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Slicer`](image_slicer.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Line Counter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `metadata` (*[`video_metadata`](../kinds/video_metadata.md)*): Video metadata containing video_identifier to maintain separate counting state for different videos. Required for persistent counting across frames..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Objects are counted when their triggering anchor point crosses the line segment..
        - `line_segment` (*[`list_of_values`](../kinds/list_of_values.md)*): Line segment defined by exactly two points, each with [x, y] coordinates. Objects crossing from one side count as 'in', objects crossing from the other side count as 'out'. Example: [[0, 100], [500, 100]] creates a horizontal line at y=100. Crossing direction depends on which side objects approach from..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the bounding box that must cross the line for counting. Options: CENTER (default), BOTTOM_CENTER, TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, etc. CENTER ensures the object is substantially across the line before counting, reducing false positives from objects near but not fully crossing the line..

    - output
    
        - `count_in` ([`integer`](../kinds/integer.md)): Integer value.
        - `count_out` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Line Counter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/line_counter@v1",
	    "metadata": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "line_segment": [
	        [
	            0,
	            50
	        ],
	        [
	            500,
	            50
	        ]
	    ],
	    "triggering_anchor": "CENTER"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

