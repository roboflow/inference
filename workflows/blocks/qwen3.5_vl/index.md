
# Qwen3.5-VL

!!! warning "Deprecated"

    Use the unified Qwen-VL block (`roboflow_core/qwen_vlm@v1`), which exposes Qwen 3.5 VL alongside other Qwen variants and the OpenRouter passthrough.



??? "Class: `Qwen35VLBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/qwen3_5vl/v1.py">inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v1.Qwen35VLBlockV1</a>
    


This workflow block runs Qwen3.5-VL—a vision language model that accepts an image and an optional text prompt—and returns a text answer based on a conversation template.

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/qwen3_5vl@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Optional text prompt to provide additional context to Qwen3.5-VL. Otherwise it will just be a default one, which may affect the desired model behavior.. | ❌ |
| `model_version` | `str` | The Qwen3.5-VL model to be used for inference.. | ✅ |
| `system_prompt` | `str` | Optional system prompt to provide additional context to Qwen3.5-VL.. | ❌ |
| `enable_thinking` | `bool` | If true, enables Qwen3.5-VL's thinking mode, which allows the model to generate reasoning tokens before answering. The thinking output will be returned in the 'thinking' field.. | ❌ |
| `max_new_tokens` | `int` | Maximum number of tokens to generate. If not set, the model's default will be used. Consider increasing for thinking mode.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Qwen3.5-VL` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Image Threshold`](image_threshold.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Relative Static Crop`](relative_static_crop.md), [`Circle Visualization`](circle_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Halo Visualization`](halo_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Cache Set`](cache_set.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`SAM 3`](sam3.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Google Gemma`](google_gemma.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Equalization`](contrast_equalization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Cache Get`](cache_get.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Qwen3.5-VL` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_version` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): The Qwen3.5-VL model to be used for inference..

    - output
    
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `thinking` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Qwen3.5-VL` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/qwen3_5vl@v1",
	    "images": "$inputs.image",
	    "prompt": "What is in this image?",
	    "model_version": "qwen3_5-0.8b",
	    "system_prompt": "You are a helpful assistant.",
	    "enable_thinking": "<block_does_not_provide_example>",
	    "max_new_tokens": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

