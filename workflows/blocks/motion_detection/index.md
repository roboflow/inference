
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Motion Detection` in version `v1`.

    - inputs: [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Slack Notification`](slack_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Focus`](camera_focus.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Trace Visualization`](trace_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Anthropic Claude`](anthropic_claude.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Google Gemini`](google_gemini.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Line Counter`](line_counter.md), [`Buffer`](buffer.md), [`Image Preprocessing`](image_preprocessing.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Clip Comparison`](clip_comparison.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Template Matching`](template_matching.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Size Measurement`](size_measurement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md)
    - outputs: [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Cache Set`](cache_set.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Overlap Analysis`](overlap_analysis.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Seg Preview`](seg_preview.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Detections Transformation`](detections_transformation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SIFT Comparison`](sift_comparison.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
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

