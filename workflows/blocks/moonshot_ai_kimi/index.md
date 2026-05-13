
# MoonshotAI Kimi



??? "Class: `KimiOpenRouterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/kimi_openrouter/v1.py">inference.core.workflows.core_steps.models.foundation.kimi_openrouter.v1.KimiOpenRouterBlockV1</a>
    



Ask a question to Moonshot AI Kimi vision-language models served via OpenRouter.

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

* **Open Prompt** (`unconstrained`) - Use any prompt to generate a raw response

* **Text Recognition (OCR)** (`ocr`) - Model recognizes text in the image

* **Visual Question Answering** (`visual-question-answering`) - Model answers the question you submit in the prompt

* **Captioning (short)** (`caption`) - Model provides a short description of the image

* **Captioning** (`detailed-caption`) - Model provides a long description of the image

* **Single-Label Classification** (`classification`) - Model classifies the image content as one of the provided classes

* **Multi-Label Classification** (`multi-label-classification`) - Model classifies the image content as one or more of the provided classes

* **Unprompted Object Detection** (`object-detection`) - Model detects and returns the bounding boxes for prominent objects in the image

* **Structured Output Generation** (`structured-answering`) - Model returns a JSON response with the specified fields

#### 🛠️ API providers and model variants

Kimi is exposed via [OpenRouter API](https://openrouter.ai/) and we require
passing an [OpenRouter API Key](https://openrouter.ai/docs/api-keys) to run.

Pick a specific model version from the `model_version` dropdown - new Kimi releases
will be added to this list as they become available on OpenRouter.

!!! warning "API Usage Charges"

    OpenRouter is an external third party providing access to the model and incurring charges on the usage.
    Please check pricing on [openrouter.ai](https://openrouter.ai/) before use.

#### 💡 Further reading and Acceptable Use Policy

!!! warning "Model license"

    Check the [Moonshot AI Kimi license terms](https://huggingface.co/moonshotai) before use.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/kimi_openrouter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Kimi model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your OpenRouter API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response.. | ❌ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `MoonshotAI Kimi` in version `v1`.

    - inputs: [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`CSV Formatter`](csv_formatter.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Image Threshold`](image_threshold.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`Gaze Detection`](gaze_detection.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)
    - outputs: [`Slack Notification`](slack_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Cache Set`](cache_set.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Distance Measurement`](distance_measurement.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Time in Zone`](timein_zone.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Google Gemini`](google_gemini.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Cache Get`](cache_get.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Blur`](image_blur.md), [`Contrast Equalization`](contrast_equalization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`MoonshotAI Kimi` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Kimi model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Your OpenRouter API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `MoonshotAI Kimi` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/kimi_openrouter@v1",
	    "images": "$inputs.image",
	    "task_type": "<block_does_not_provide_example>",
	    "prompt": "my prompt",
	    "output_structure": {
	        "my_key": "description"
	    },
	    "classes": [
	        "class-a",
	        "class-b"
	    ],
	    "api_key": "xxx-xxx",
	    "model_version": "Kimi K2.6 - OpenRouter",
	    "max_tokens": "<block_does_not_provide_example>",
	    "temperature": "<block_does_not_provide_example>",
	    "max_concurrent_requests": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

