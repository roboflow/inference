
# LMM

!!! warning "Deprecated"

    This block is deprecated and may be removed in a future release.



??? "Class: `LMMBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/lmm/v1.py">inference.core.workflows.core_steps.models.foundation.lmm.v1.LMMBlockV1</a>
    



Ask a question to a Large Multimodal Model (LMM) with an image and text.

You can specify arbitrary text prompts to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision;

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 

_If you want to classify an image into one or more categories, we recommend using the 
dedicated LMMForClassificationBlock._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/lmm@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Holds unconstrained text prompt to LMM mode. | ✅ |
| `lmm_type` | `str` | Type of LMM to be used. | ✅ |
| `lmm_config` | `LMMConfig` | Configuration of LMM. | ❌ |
| `remote_api_key` | `str` | Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`.. | ✅ |
| `json_output` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `hosted_serverless`; execution `remote`
:   LMM_ENABLED=False on Roboflow Hosted Serverless: the /llm_v1 endpoint is not registered, so run_remotely() returns 404.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `LMM` in version `v1`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Cache Set`](cache_set.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Overlap Filter`](overlap_filter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Camera Focus`](camera_focus.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`Dimension Collapse`](dimension_collapse.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Expression`](expression.md), [`Distance Measurement`](distance_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Switch Case`](switch_case.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Current Time`](current_time.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Perspective Correction`](perspective_correction.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Camera Calibration`](camera_calibration.md), [`Line Counter`](line_counter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`SIFT Comparison`](sift_comparison.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Qwen3.5`](qwen3.5.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Delta Filter`](delta_filter.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Barcode Detection`](barcode_detection.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Blur Visualization`](blur_visualization.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Depth Estimation`](depth_estimation.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`LMM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Holds unconstrained text prompt to LMM mode.
        - `lmm_type` (*[`string`](../kinds/string.md)*): Type of LMM to be used.
        - `remote_api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`..

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `LMM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/lmm@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "lmm_type": "gpt_4v",
	    "lmm_config": {
	        "gpt_image_detail": "low",
	        "gpt_model_version": "gpt-4o",
	        "max_tokens": 200
	    },
	    "remote_api_key": "xxx-xxx",
	    "json_output": {
	        "count": "number of cats in the picture"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

