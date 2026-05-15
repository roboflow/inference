
# CLIP Embedding Model



??? "Class: `ClipModelBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/clip/v1.py">inference.core.workflows.core_steps.models.foundation.clip.v1.ClipModelBlockV1</a>
    



Use a CLIP model to create semantic embeddings of text and images.

This block accepts an image or string and returns an embedding.
The embedding can be used to compare the similarity between different
images or between images and text.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/clip@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `data` | `str` | The string or image to generate an embedding for.. | ✅ |
| `version` | `str` | Variant of CLIP model. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `CLIP Embedding Model` in version `v1`.

    - inputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Blur Visualization`](blur_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Cosine Similarity`](cosine_similarity.md), [`Identify Outliers`](identify_outliers.md), [`Identify Changes`](identify_changes.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CLIP Embedding Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`string`](../kinds/string.md), [`image`](../kinds/image.md)]*): The string or image to generate an embedding for..
        - `version` (*[`string`](../kinds/string.md)*): Variant of CLIP model.

    - output
    
        - `embedding` ([`embedding`](../kinds/embedding.md)): A list of floating point numbers representing a vector embedding..



??? tip "Example JSON definition of step `CLIP Embedding Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/clip@v1",
	    "data": "$inputs.image",
	    "version": "ViT-B-16"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

