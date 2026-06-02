
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

    - inputs: [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Heatmap Visualization`](heatmap_visualization.md), [`CogVLM`](cog_vlm.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenRouter`](open_router.md), [`Color Visualization`](color_visualization.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`GLM-OCR`](glmocr.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Camera Calibration`](camera_calibration.md), [`Corner Visualization`](corner_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Relative Static Crop`](relative_static_crop.md), [`OCR Model`](ocr_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen-VL`](qwen_vl.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`Identify Outliers`](identify_outliers.md), [`Identify Changes`](identify_changes.md), [`Cosine Similarity`](cosine_similarity.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CLIP Embedding Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`image`](../kinds/image.md), [`string`](../kinds/string.md)]*): The string or image to generate an embedding for..
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

