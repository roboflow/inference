
# Google Gemini



## v3

??? "Class: `GoogleGeminiBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/google_gemini/v3.py">inference.core.workflows.core_steps.models.foundation.google_gemini.v3.GoogleGeminiBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to Google's Gemini model with vision capabilities.

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

### API Key Options

This block supports two API key modes:

1. **Roboflow Managed API Key (Default)** - Use `rf_key:account` to proxy requests through Roboflow's API:
   * **Simplified setup** - no Google AI API key required
   * **Secure** - your workflow API key is used for authentication
   * **Usage-based billing** - charged per token based on the model used

2. **Custom Google AI API Key** - Provide your own Google AI API key:
   * Full control over API usage
   * You pay Google directly

**WARNING!**

This block makes use of `/v1beta` API of Google Gemini model - the implementation may change
in the future, without guarantee of backward compatibility.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/google_gemini@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Gemini model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your Google AI API key or 'rf_key:account' to use Roboflow's managed API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `thinking_level` | `str` | Controls the depth of internal reasoning for Gemini 3+ models. 'low' minimizes latency and cost (best for simple tasks), 'high' maximizes reasoning depth (default). Only supported by Gemini 3 and newer models.. | ✅ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response. If not specified, the model will use its default limit.. | ❌ |
| `google_code_execution` | `bool` | Enable native code execution for the Gemini model.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit Google Gemini API limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Google Gemini` in version `v3`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Identify Outliers`](identify_outliers.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Google Gemini` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Gemini model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md), [`ROBOFLOW_MANAGED_KEY`](../kinds/roboflow_managed_key.md)]*): Your Google AI API key or 'rf_key:account' to use Roboflow's managed API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `thinking_level` (*[`string`](../kinds/string.md)*): Controls the depth of internal reasoning for Gemini 3+ models. 'low' minimizes latency and cost (best for simple tasks), 'high' maximizes reasoning depth (default). Only supported by Gemini 3 and newer models..
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..
        - `google_code_execution` (*[`boolean`](../kinds/boolean.md)*): Enable native code execution for the Gemini model..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Google Gemini` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/google_gemini@v3",
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
	    "api_key": "rf_key:account",
	    "model_version": "gemini-3-pro-preview",
	    "thinking_level": "<block_does_not_provide_example>",
	    "temperature": "<block_does_not_provide_example>",
	    "max_tokens": "<block_does_not_provide_example>",
	    "google_code_execution": "<block_does_not_provide_example>",
	    "max_concurrent_requests": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `GoogleGeminiBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/google_gemini/v2.py">inference.core.workflows.core_steps.models.foundation.google_gemini.v2.GoogleGeminiBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to Google's Gemini model with vision capabilities.

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

You need to provide your Google AI API key to use the Gemini model.

**WARNING!**

This block makes use of `/v1beta` API of Google Gemini model - the implementation may change
in the future, without guarantee of backward compatibility.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/google_gemini@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Gemini model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your Google AI API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `thinking_level` | `str` | Controls the depth of internal reasoning for Gemini 3+ models. 'low' minimizes latency and cost (best for simple tasks), 'high' maximizes reasoning depth (default). Only supported by Gemini 3 and newer models.. | ✅ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response. If not specified, the model will use its default limit.. | ❌ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit Google Gemini API limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Google Gemini` in version `v2`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Google Gemini` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Gemini model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Your Google AI API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `thinking_level` (*[`string`](../kinds/string.md)*): Controls the depth of internal reasoning for Gemini 3+ models. 'low' minimizes latency and cost (best for simple tasks), 'high' maximizes reasoning depth (default). Only supported by Gemini 3 and newer models..
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Google Gemini` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/google_gemini@v2",
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
	    "model_version": "gemini-3-pro-preview",
	    "thinking_level": "<block_does_not_provide_example>",
	    "temperature": "<block_does_not_provide_example>",
	    "max_tokens": "<block_does_not_provide_example>",
	    "max_concurrent_requests": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `GoogleGeminiBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/google_gemini/v1.py">inference.core.workflows.core_steps.models.foundation.google_gemini.v1.GoogleGeminiBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to Google's Gemini model with vision capabilities.

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

You need to provide your Google AI API key to use the Gemini model. 

**WARNING!**

This block makes use of `/v1beta` API of Google Gemini model - the implementation may change 
in the future, without guarantee of backward compatibility.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/google_gemini@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Gemini model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your Google AI API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response.. | ❌ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit Google Gemini API limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Google Gemini` in version `v1`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Google Gemini` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Gemini model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Your Google AI API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Google Gemini` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/google_gemini@v1",
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
	    "model_version": "gemini-2.5-pro",
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

