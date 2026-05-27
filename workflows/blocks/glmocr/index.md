
# GLM-OCR



??? "Class: `GLMOCRBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/glm_ocr/v1.py">inference.core.workflows.core_steps.models.foundation.glm_ocr.v1.GLMOCRBlockV1</a>
    



Recognize text in images using GLM-OCR, a vision language model by Zhipu AI specialized
for optical character recognition.

GLM-OCR supports three built-in recognition modes:

- **Text Recognition** — General-purpose text recognition for
  serial numbers, labels, scene text, and documents.
- **Formula Recognition** — Recognizes mathematical formulas
  and equations.
- **Table Recognition** — Recognizes table structures and content.

You can also select **Custom Prompt** to provide your own prompt for specialized
recognition tasks, or **Structured Output** to extract values from the image
into a JSON document with a user-defined schema (pair with the JSON Parser
block to materialize the keys as workflow outputs).

This block pairs well with detection models and DynamicCropBlock to isolate regions of
interest before running OCR. For example, use an object detection model to find labels
or text regions, crop them, then pass the crops to GLM-OCR.

Note: GLM-OCR requires a GPU for inference.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/glm_ocr@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Recognition task to perform. Determines the prompt sent to GLM-OCR.. | ❌ |
| `prompt` | `str` | Custom text prompt for GLM-OCR. Only used when task_type is 'custom'.. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary describing the structure of the expected JSON response. Keys are the JSON field names; values describe what the model should put in each field.. | ❌ |
| `max_new_tokens` | `int` | Maximum number of tokens to generate. If not set, the model default will be used.. | ❌ |
| `model_version` | `str` | The GLM-OCR model to be used for inference.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `GLM-OCR` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`OCR Model`](ocr_model.md), [`Email Notification`](email_notification.md), [`Object Detection Model`](object_detection_model.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`JSON Parser`](json_parser.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Cache Set`](cache_set.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Florence-2 Model`](florence2_model.md), [`SAM 3`](sam3.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Cache Get`](cache_get.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`GLM-OCR` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Custom text prompt for GLM-OCR. Only used when task_type is 'custom'..
        - `model_version` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): The GLM-OCR model to be used for inference..

    - output
    
        - `parsed_output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.



??? tip "Example JSON definition of step `GLM-OCR` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/glm_ocr@v1",
	    "images": "$inputs.image",
	    "task_type": "<block_does_not_provide_example>",
	    "prompt": "Describe the text in the image.",
	    "output_structure": {
	        "my_key": "description"
	    },
	    "max_new_tokens": "<block_does_not_provide_example>",
	    "model_version": "glm-ocr"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

