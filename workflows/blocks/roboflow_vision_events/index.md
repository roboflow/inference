
# Roboflow Vision Events



??? "Class: `RoboflowVisionEventsBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/vision_events/v1.py">inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1.RoboflowVisionEventsBlockV1</a>
    



Send images, model predictions, and event metadata to the Roboflow Vision Events API for
monitoring, quality control, safety alerting, and custom event tracking.

## How This Block Works

This block uploads workflow images and model predictions to the Roboflow Vision Events API,
creating structured events that can be queried, filtered, and visualized in the Roboflow
dashboard.

1. Optionally uploads an input image and/or output image (visualization) to the Vision Events
   image storage via the public API
2. Converts model predictions (object detection, classification, instance segmentation, or
   keypoint detection) into the Vision Events annotation format and attaches them to the
   input image
3. Creates a vision event with the specified event type, use case, event data,
   and custom metadata
4. Supports fire-and-forget mode for non-blocking execution

## Event Types

- **quality_check**: Manufacturing/inspection QA with pass/fail result and optional confidence
- **inventory_count**: Inventory tracking with location, item count, and item type
- **safety_alert**: Safety violations with alert type, severity (low/medium/high), and description
- **custom**: User-defined events with a free-form value string
- **operator_feedback**: Operator review/correction of previous events (correct/incorrect/inconclusive)

## Requirements

**API Key Required**: This block requires a valid Roboflow API key with `vision-events:write`
scope. The API key must be configured in your environment or workflow configuration.

## Common Use Cases

- **Quality Control**: Automatically log inspection results with images and detection overlays
- **Safety Monitoring**: Send safety alerts when violations are detected in video streams
- **Production Analytics**: Track inventory counts and production metrics with visual evidence
- **Active Monitoring**: Fire-and-forget event logging from real-time video processing workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_vision_events@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `event_type` | `str` | The type of vision event to create.. | ✅ |
| `solution` | `str` | The use case to associate the event with. Events are namespaced by use case within a workspace.. | ✅ |
| `external_id` | `str` | External identifier for correlation with other systems (max 1000 chars).. | ✅ |
| `qc_result` | `str` | Quality check result: pass or fail.. | ✅ |
| `location` | `str` | Location identifier for inventory count.. | ✅ |
| `item_count` | `int` | Number of items counted.. | ✅ |
| `item_type` | `str` | Type of item being counted.. | ✅ |
| `alert_type` | `str` | Alert type identifier (e.g. no_hardhat, spill_detected).. | ✅ |
| `severity` | `str` | Severity level for the safety alert.. | ✅ |
| `alert_description` | `str` | Description of the safety alert.. | ✅ |
| `custom_value` | `str` | Arbitrary value for custom events.. | ✅ |
| `related_event_id` | `str` | The event ID of the event being reviewed.. | ✅ |
| `feedback` | `str` | Operator feedback on the related event.. | ✅ |
| `custom_metadata` | `Dict[str, Union[bool, float, int, str]]` | Flat key-value metadata to attach to the event. Keys must match pattern [a-zA-Z0-9_ -]+ (max 100 chars). String values max 1000 chars.. | ✅ |
| `fire_and_forget` | `bool` | If True, the event is sent asynchronously and the workflow continues without waiting. If False, the block waits for the API response.. | ✅ |
| `disable_sink` | `bool` | If True, the block is disabled and no events are sent.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Vision Events` in version `v1`.

    - inputs: [`Distance Measurement`](distance_measurement.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Cache Get`](cache_get.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CogVLM`](cog_vlm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenRouter`](open_router.md), [`Cache Set`](cache_set.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Rate Limiter`](rate_limiter.md), [`Detections Combine`](detections_combine.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`LMM`](lmm.md), [`Google Gemini`](google_gemini.md), [`Cosine Similarity`](cosine_similarity.md), [`QR Code Detection`](qr_code_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`SmolVLM2`](smol_vlm2.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Gaze Detection`](gaze_detection.md), [`Dominant Color`](dominant_color.md), [`Florence-2 Model`](florence2_model.md), [`Identify Changes`](identify_changes.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Seg Preview`](seg_preview.md), [`Moondream2`](moondream2.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen-VL`](qwen_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stitch`](detections_stitch.md), [`OpenAI`](open_ai.md), [`Detections Merge`](detections_merge.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Filter`](detections_filter.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Stitch Images`](stitch_images.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`SAM 3`](sam3.md), [`Delta Filter`](delta_filter.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Continue If`](continue_if.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detection Event Log`](detection_event_log.md), [`Inner Workflow`](inner_workflow.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Preprocessing`](image_preprocessing.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Qwen3.5`](qwen3.5.md), [`Time in Zone`](timein_zone.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Label Visualization`](label_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Property Definition`](property_definition.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Expression`](expression.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Data Aggregator`](data_aggregator.md), [`Camera Focus`](camera_focus.md), [`Dynamic Crop`](dynamic_crop.md), [`Relative Static Crop`](relative_static_crop.md), [`OCR Model`](ocr_model.md), [`Image Stack`](image_stack.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen3-VL`](qwen3_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Path Deviation`](path_deviation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT`](sift.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md)
    - outputs: [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Cache Get`](cache_get.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`CogVLM`](cog_vlm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Cache Set`](cache_set.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Moondream2`](moondream2.md), [`Seg Preview`](seg_preview.md), [`QR Code Generator`](qr_code_generator.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Threshold`](image_threshold.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Vision Events` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `input_image` (*[`image`](../kinds/image.md)*): The original input image. Uploaded to the Vision Events API and used as the base image for detection annotations..
        - `output_image` (*[`image`](../kinds/image.md)*): An optional output/visualized image (e.g., from a visualization block). Displayed as the primary image in the Vision Events dashboard..
        - `predictions` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Optional model predictions to include as detection annotations on the input image. Supports object detection, instance segmentation, keypoint detection, and classification predictions..
        - `event_type` (*[`string`](../kinds/string.md)*): The type of vision event to create..
        - `solution` (*Union[[`roboflow_solution`](../kinds/roboflow_solution.md), [`string`](../kinds/string.md)]*): The use case to associate the event with. Events are namespaced by use case within a workspace..
        - `external_id` (*[`string`](../kinds/string.md)*): External identifier for correlation with other systems (max 1000 chars)..
        - `qc_result` (*[`string`](../kinds/string.md)*): Quality check result: pass or fail..
        - `location` (*[`string`](../kinds/string.md)*): Location identifier for inventory count..
        - `item_count` (*[`integer`](../kinds/integer.md)*): Number of items counted..
        - `item_type` (*[`string`](../kinds/string.md)*): Type of item being counted..
        - `alert_type` (*[`string`](../kinds/string.md)*): Alert type identifier (e.g. no_hardhat, spill_detected)..
        - `severity` (*[`string`](../kinds/string.md)*): Severity level for the safety alert..
        - `alert_description` (*[`string`](../kinds/string.md)*): Description of the safety alert..
        - `custom_value` (*[`string`](../kinds/string.md)*): Arbitrary value for custom events..
        - `related_event_id` (*[`string`](../kinds/string.md)*): The event ID of the event being reviewed..
        - `feedback` (*[`string`](../kinds/string.md)*): Operator feedback on the related event..
        - `custom_metadata` (*[`*`](../kinds/wildcard.md)*): Flat key-value metadata to attach to the event. Keys must match pattern [a-zA-Z0-9_ -]+ (max 100 chars). String values max 1000 chars..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): If True, the event is sent asynchronously and the workflow continues without waiting. If False, the block waits for the API response..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, the block is disabled and no events are sent..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Vision Events` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_vision_events@v1",
	    "input_image": "$inputs.image",
	    "output_image": "$steps.visualization.image",
	    "predictions": "$steps.object_detection_model.predictions",
	    "event_type": "quality_check",
	    "solution": "my-use-case",
	    "external_id": "batch-2025-001",
	    "qc_result": "pass",
	    "location": "warehouse-A",
	    "item_count": 42,
	    "item_type": "widget",
	    "alert_type": "no_hardhat",
	    "severity": "high",
	    "alert_description": "Worker detected without hardhat in zone B",
	    "custom_value": "anomaly detected at 14:32",
	    "related_event_id": "evt_abc123",
	    "feedback": "correct",
	    "custom_metadata": {
	        "camera_id": "cam_01",
	        "location": "$inputs.location"
	    },
	    "fire_and_forget": true,
	    "disable_sink": false
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

