
# CogVLM



??? "Class: `CogVLMBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/cog_vlm/v1.py">inference.core.workflows.core_steps.models.foundation.cog_vlm.v1.CogVLMBlockV1</a>
    




!!! Warning "CogVLM reached **End Of Life**"

    Due to dependencies conflicts with newer models and security vulnerabilities discovered in `transformers`
    library patched in the versions of library incompatible with the model we announced End Of Life for CogVLM
    support in `inference`, effective since release `0.38.0`.
    
    We are leaving this block in ecosystem until release `0.42.0` for clients to get informed about change that 
    was introduced.
    
    Starting as of now, all Workflows using the block stop being functional (runtime error will be raised), 
    after inference release `0.42.0` - this block will be removed and Execution Engine will raise compilation 
    error seeing the block in Workflow definition. 


Ask a question to CogVLM, an open source vision-language model.

This model requires a GPU and can only be run on self-hosted devices, and is not available on the Roboflow Hosted API.

_This model was previously part of the LMM block._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/cog_vlm@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Text prompt to the CogVLM model. | ✅ |
| `json_output_format` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `hosted_serverless`; execution `remote`
:   LMM_ENABLED=False on Roboflow Hosted Serverless: the /llm_v1 and /infer/cog_vlm endpoints are not registered, so run_remotely() returns 404.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `CogVLM` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Track Class Lock`](track_class_lock.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT`](sift.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Delta Filter`](delta_filter.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Property Definition`](property_definition.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Camera Calibration`](camera_calibration.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Filter`](detections_filter.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Continue If`](continue_if.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Motion Detection`](motion_detection.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Switch Case`](switch_case.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Local File Sink`](local_file_sink.md), [`Rate Limiter`](rate_limiter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CogVLM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the CogVLM model.

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `CogVLM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/cog_vlm@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "json_output_format": {
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

