
# Google Vision OCR



??? "Class: `GoogleVisionOCRBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/google_vision_ocr/v1.py">inference.core.workflows.core_steps.models.foundation.google_vision_ocr.v1.GoogleVisionOCRBlockV1</a>
    



Detect text in images using Google Vision OCR.

Supported types of text detection:

- `text_detection`: optimized for areas of text within a larger image.
- `ocr_text_detection`: optimized for dense text documents.

Provide your Google Vision API key or set the value to ``rf_key:account`` (or
``rf_key:user:<id>``) to proxy requests through Roboflow's API.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/google_vision_ocr@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `ocr_type` | `str` | Type of OCR to use. | ❌ |
| `api_key` | `str` | Your Google Vision API key. | ✅ |
| `language_hints` | `List[str]` | Optional list of language codes to pass to the OCR API. If not provided, the API will attempt to detect the language automatically.If provided, language codes must be supported by the OCR API, visit https://cloud.google.com/vision/docs/languages for list of supported language codes.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Google Vision OCR` in version `v1`.

    - inputs: [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Preprocessing`](image_preprocessing.md), [`Florence-2 Model`](florence2_model.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenRouter`](open_router.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Webhook Sink`](webhook_sink.md), [`OCR Model`](ocr_model.md), [`Color Visualization`](color_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Relative Static Crop`](relative_static_crop.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Current Time`](current_time.md), [`OpenAI`](open_ai.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Trace Visualization`](trace_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma`](google_gemma.md), [`Google Gemma API`](google_gemma_api.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Slack Notification`](slack_notification.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Pixel Color Count`](pixel_color_count.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Merge`](detections_merge.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Filter`](detections_filter.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Slack Notification`](slack_notification.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`Overlap Filter`](overlap_filter.md), [`YOLO-World Model`](yolo_world_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Velocity`](velocity.md), [`Detections Combine`](detections_combine.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Google Vision OCR` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image to run OCR.
        - `api_key` (*Union[[`string`](../kinds/string.md), [`ROBOFLOW_MANAGED_KEY`](../kinds/roboflow_managed_key.md), [`secret`](../kinds/secret.md)]*): Your Google Vision API key.

    - output
    
        - `text` ([`string`](../kinds/string.md)): String value.
        - `language` ([`string`](../kinds/string.md)): String value.
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Google Vision OCR` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/google_vision_ocr@v1",
	    "image": "$inputs.image",
	    "ocr_type": "<block_does_not_provide_example>",
	    "api_key": "xxx-xxx",
	    "language_hints": [
	        "en",
	        "fr"
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

