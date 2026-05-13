
# Qwen3.5-VL



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

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Color Visualization`](color_visualization.md), [`Label Visualization`](label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Grid Visualization`](grid_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT`](sift.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Depth Estimation`](depth_estimation.md)
    - outputs: [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`QR Code Generator`](qr_code_generator.md), [`YOLO-World Model`](yolo_world_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Distance Measurement`](distance_measurement.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Time in Zone`](timein_zone.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Cache Set`](cache_set.md), [`Color Visualization`](color_visualization.md), [`Email Notification`](email_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Circle Visualization`](circle_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Threshold`](image_threshold.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Text Display`](text_display.md), [`Google Gemini`](google_gemini.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Cache Get`](cache_get.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Depth Estimation`](depth_estimation.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md)

    
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

