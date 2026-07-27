
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
| `task_type` | `str` | Recognition task to perform. Determines the prompt sent to GLM-OCR. Accepts a selector (e.g. $inputs.task_type) so the mode can be set dynamically.. | ✅ |
| `prompt` | `str` | Custom text prompt for GLM-OCR. Only used when task_type is 'custom'.. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary describing the structure of the expected JSON response. Keys are the JSON field names; values describe what the model should put in each field.. | ❌ |
| `max_new_tokens` | `int` | Maximum number of tokens to generate. If not set, the model default will be used.. | ❌ |
| `model_version` | `str` | The GLM-OCR model to be used for inference.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `GLM-OCR` in version `v1`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Current Time`](current_time.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Grid Visualization`](grid_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Path Deviation`](path_deviation.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`GLM-OCR` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `task_type` (*[`string`](../kinds/string.md)*): Recognition task to perform. Determines the prompt sent to GLM-OCR. Accepts a selector (e.g. $inputs.task_type) so the mode can be set dynamically..
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

