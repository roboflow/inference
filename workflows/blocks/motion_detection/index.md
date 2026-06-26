
# Motion Detection



??? "Class: `MotionDetectionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/motion_detection/v1.py">inference.core.workflows.core_steps.classical_cv.motion_detection.v1.MotionDetectionBlockV1</a>
    



Detect motion in video streams using OpenCV's background subtraction algorithm.

## How This Block Works

This block uses background subtraction (specifically the MOG2 algorithm) to detect motion in video frames. The block maintains state across frames to build a background model and track motion patterns:

1. **Initializes background model** - on the first frame, creates a background subtractor using the specified history and threshold parameters
2. **Processes each frame** - applies background subtraction to identify pixels that differ from the learned background model
3. **Filters noise** - applies morphological operations to remove noise and combine nearby motion regions into coherent contours
4. **Extracts motion regions** - finds contours representing motion areas, filters them by minimum size, and optionally clips them to a detection zone
5. **Simplifies contours** - reduces contour complexity to keep detection data manageable
6. **Generates outputs** - creates object detection predictions with bounding boxes, determines motion status, triggers alarms when motion starts, and provides motion zone polygons

The block tracks motion state across frames - the **alarm** output becomes true only when motion transitions from not detected to detected, making it useful for triggering actions when motion first appears.

## Common Use Cases

- **Security Monitoring**: Detect motion in surveillance cameras to trigger alerts, recordings, or notifications when activity is detected
- **Resource Optimization**: Conditionally run expensive inference operations (e.g., object detection, classification) only when motion is detected to save computational resources
- **Activity Detection**: Monitor areas for movement to track occupancy, identify entry/exit events, or detect unauthorized access
- **Video Analytics**: Analyze video streams to identify motion patterns, track activity levels, or detect anomalies in monitored areas
- **Smart Recording**: Trigger video recording or snapshot capture when motion is detected, reducing storage requirements compared to continuous recording
- **Zone Monitoring**: Monitor specific areas within a frame using detection zones to focus motion detection on relevant regions while ignoring busy but irrelevant areas

## Connecting to Other Blocks

The motion detection outputs from this block can be connected to:

- **Conditional logic blocks** (e.g., Continue If) to execute workflow steps only when motion is detected or when alarms trigger
- **Object detection blocks** to run detection models only on frames with motion, saving computational resources
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send alerts when motion is detected or alarms trigger
- **Data storage blocks** (e.g., Roboflow Dataset Upload, CSV Formatter) to log motion events, timestamps, and detection data for analytics
- **Visualization blocks** to draw motion zones, bounding boxes, or annotations on frames showing detected motion
- **Filter blocks** to filter images or data based on motion status before passing to downstream processing


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/motion_detection@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `minimum_contour_area` | `int` | Minimum area in square pixels for a motion region to be detected. Contours smaller than this threshold are filtered out to ignore noise, small shadows, or minor pixel variations. Lower values increase sensitivity but may detect more false positives (e.g., 100 for very sensitive detection, 500 for only large objects). Default is 200 square pixels.. | ✅ |
| `morphological_kernel_size` | `int` | Size of the morphological kernel in pixels used to combine nearby motion regions and filter noise. Larger values merge more distant motion regions into single contours but may also merge separate objects. Smaller values preserve more detail but may leave fragmented detections. The kernel uses an elliptical shape. Default is 3 pixels.. | ✅ |
| `threshold` | `int` | Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16.. | ✅ |
| `history` | `int` | Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames.. | ✅ |
| `detection_zone` | `Union[List[Any], str]` | Optional polygon zone to limit motion detection to a specific area of the frame. Motion is only detected within this zone, ignoring activity outside. Format: [[x1, y1], [x2, y2], [x3, y3], ...] where coordinates are in pixels. The polygon must have more than 3 points. Can be provided as a list, JSON string, or selector referencing zone outputs from other blocks. Useful for focusing on specific regions (e.g., doorways, windows, restricted areas) while ignoring busy but irrelevant areas. If not provided, motion is detected across the entire frame.. | ✅ |
| `suppress_first_detections` | `bool` | If true, suppresses motion detections until the background model has been initialized with enough frames (specified by the history parameter). This prevents false positives from early frames where the background model hasn't learned the scene yet. When false, the block attempts to detect motion immediately, which may produce unreliable results during initialization. Default is true (recommended for most use cases).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Motion Detection` in version `v1`.

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Slack Notification`](slack_notification.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`JSON Parser`](json_parser.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Threshold`](image_threshold.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Blur`](image_blur.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Depth Estimation`](depth_estimation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Crop Visualization`](crop_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Buffer`](buffer.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`Image Slicer`](image_slicer.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Identify Changes`](identify_changes.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`PLC Reader`](plc_reader.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Webhook Sink`](webhook_sink.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Qwen-VL`](qwen_vl.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Track Class Lock`](track_class_lock.md), [`Text Display`](text_display.md), [`Buffer`](buffer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Triangle Visualization`](triangle_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Motion Detection` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The input image or video frame to analyze for motion. The block processes frames sequentially to build a background model - each frame updates the background model and detects motion relative to learned background patterns. Can be connected from workflow inputs or previous steps..
        - `minimum_contour_area` (*[`integer`](../kinds/integer.md)*): Minimum area in square pixels for a motion region to be detected. Contours smaller than this threshold are filtered out to ignore noise, small shadows, or minor pixel variations. Lower values increase sensitivity but may detect more false positives (e.g., 100 for very sensitive detection, 500 for only large objects). Default is 200 square pixels..
        - `morphological_kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the morphological kernel in pixels used to combine nearby motion regions and filter noise. Larger values merge more distant motion regions into single contours but may also merge separate objects. Smaller values preserve more detail but may leave fragmented detections. The kernel uses an elliptical shape. Default is 3 pixels..
        - `threshold` (*[`integer`](../kinds/integer.md)*): Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16..
        - `history` (*[`integer`](../kinds/integer.md)*): Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames..
        - `detection_zone` (*Union[[`zone`](../kinds/zone.md), [`list_of_values`](../kinds/list_of_values.md)]*): Optional polygon zone to limit motion detection to a specific area of the frame. Motion is only detected within this zone, ignoring activity outside. Format: [[x1, y1], [x2, y2], [x3, y3], ...] where coordinates are in pixels. The polygon must have more than 3 points. Can be provided as a list, JSON string, or selector referencing zone outputs from other blocks. Useful for focusing on specific regions (e.g., doorways, windows, restricted areas) while ignoring busy but irrelevant areas. If not provided, motion is detected across the entire frame..
        - `suppress_first_detections` (*[`boolean`](../kinds/boolean.md)*): If true, suppresses motion detections until the background model has been initialized with enough frames (specified by the history parameter). This prevents false positives from early frames where the background model hasn't learned the scene yet. When false, the block attempts to detect motion immediately, which may produce unreliable results during initialization. Default is true (recommended for most use cases)..

    - output
    
        - `motion` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `alarm` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `detections` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `motion_zones` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Motion Detection` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/motion_detection@v1",
	    "image": "$inputs.image",
	    "minimum_contour_area": 200,
	    "morphological_kernel_size": 3,
	    "threshold": 16,
	    "history": 30,
	    "detection_zone": "<block_does_not_provide_example>",
	    "suppress_first_detections": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

