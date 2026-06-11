
# LMM For Classification

!!! warning "Deprecated"

    This block is deprecated and may be removed in a future release.



??? "Class: `LMMForClassificationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/lmm_classifier/v1.py">inference.core.workflows.core_steps.models.foundation.lmm_classifier.v1.LMMForClassificationBlockV1</a>
    



Classify an image into one or more categories using a Large Multimodal Model (LMM).

You can specify arbitrary classes to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/lmm_for_classification@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `lmm_type` | `str` | Type of LMM to be used. | ✅ |
| `classes` | `List[str]` | List of classes that LMM shall classify against. | ✅ |
| `lmm_config` | `LMMConfig` | Configuration of LMM. | ❌ |
| `remote_api_key` | `str` | Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `hosted_serverless`; execution `remote`
:   LMM_ENABLED=False on Roboflow Hosted Serverless: the /llm_v1 endpoint is not registered, so run_remotely() returns 404.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `LMM For Classification` in version `v1`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Google Gemini`](google_gemini.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Cache Get`](cache_get.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Line Counter`](line_counter.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Time in Zone`](timein_zone.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Size Measurement`](size_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`LMM For Classification` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `lmm_type` (*[`string`](../kinds/string.md)*): Type of LMM to be used.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes that LMM shall classify against.
        - `remote_api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`..

    - output
    
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `top` ([`top_class`](../kinds/top_class.md)): String value representing top class predicted by classification model.
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `prediction_type` ([`prediction_type`](../kinds/prediction_type.md)): String value with type of prediction.



??? tip "Example JSON definition of step `LMM For Classification` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/lmm_for_classification@v1",
	    "images": "$inputs.image",
	    "lmm_type": "gpt_4v",
	    "classes": [
	        "a",
	        "b"
	    ],
	    "lmm_config": {
	        "gpt_image_detail": "low",
	        "gpt_model_version": "gpt-4o",
	        "max_tokens": 200
	    },
	    "remote_api_key": "xxx-xxx"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

