
# Perception Encoder Embedding Model



??? "Class: `PerceptionEncoderModelBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/perception_encoder/v1.py">inference.core.workflows.core_steps.models.foundation.perception_encoder.v1.PerceptionEncoderModelBlockV1</a>
    



Use the Meta Perception Encoder model to create semantic embeddings of text and images.

This block accepts an image or string and returns an embedding. The embedding can be used to compare
similarity between different images or between images and text.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/perception_encoder@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `data` | `str` | The string or image to generate an embedding for.. | ✅ |
| `version` | `str` | Variant of Perception Encoder model. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Perception Encoder Embedding Model` in version `v1`.

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`S3 Sink`](s3_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md)
    - outputs: [`Identify Outliers`](identify_outliers.md), [`Cosine Similarity`](cosine_similarity.md), [`Identify Changes`](identify_changes.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Perception Encoder Embedding Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`image`](../kinds/image.md), [`string`](../kinds/string.md)]*): The string or image to generate an embedding for..
        - `version` (*[`string`](../kinds/string.md)*): Variant of Perception Encoder model.

    - output
    
        - `embedding` ([`embedding`](../kinds/embedding.md)): A list of floating point numbers representing a vector embedding..



??? tip "Example JSON definition of step `Perception Encoder Embedding Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/perception_encoder@v1",
	    "data": "$inputs.image",
	    "version": "PE-Core-B16-224"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

