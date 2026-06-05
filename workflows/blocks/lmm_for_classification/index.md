
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

    - inputs: [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`OCR Model`](ocr_model.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Blur Visualization`](blur_visualization.md), [`Current Time`](current_time.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Size Measurement`](size_measurement.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Buffer`](buffer.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Threshold`](image_threshold.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenRouter`](open_router.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Slack Notification`](slack_notification.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Cache Set`](cache_set.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`LMM For Classification` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `lmm_type` (*[`string`](../kinds/string.md)*): Type of LMM to be used.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes that LMM shall classify against.
        - `remote_api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`..

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

