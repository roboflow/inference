
# OpenAI



## v4

??? "Class: `OpenAIBlockV4`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/openai/v4.py">inference.core.workflows.core_steps.models.foundation.openai.v4.OpenAIBlockV4</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to OpenAI's GPT models with vision capabilities (including GPT-5 and GPT-4o).

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

Provide your OpenAI API key or set the value to ``rf_key:account`` (or
``rf_key:user:<id>``) to proxy requests through Roboflow's API.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/open_ai@v4`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the OpenAI model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your OpenAI API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `reasoning_effort` | `str` | Controls reasoning. Reducing can result in faster responses and fewer tokens. GPT-5.1 and higher models default to 'none' (no reasoning) and support 'none', 'low', 'medium', 'high'. GPT-5.2 also supports 'xhigh'. GPT-5 models default to 'medium' and support 'minimal', 'low', 'medium', 'high'.. | ✅ |
| `image_detail` | `str` | Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in its response. If not specified, the model will use its default limit. Minimum value is 16.. | ❌ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit OpenAI limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OpenAI` in version `v4`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Current Time`](current_time.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Path Deviation`](path_deviation.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI` in version `v4`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the OpenAI model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md), [`ROBOFLOW_MANAGED_KEY`](../kinds/roboflow_managed_key.md)]*): Your OpenAI API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `reasoning_effort` (*[`string`](../kinds/string.md)*): Controls reasoning. Reducing can result in faster responses and fewer tokens. GPT-5.1 and higher models default to 'none' (no reasoning) and support 'none', 'low', 'medium', 'high'. GPT-5.2 also supports 'xhigh'. GPT-5 models default to 'medium' and support 'minimal', 'low', 'medium', 'high'..
        - `image_detail` (*[`string`](../kinds/string.md)*): Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity..
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `OpenAI` in version `v4`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/open_ai@v4",
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
	    "model_version": "gpt-5.1",
	    "reasoning_effort": "<block_does_not_provide_example>",
	    "image_detail": "auto",
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




## v3

??? "Class: `OpenAIBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/openai/v3.py">inference.core.workflows.core_steps.models.foundation.openai.v3.OpenAIBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to OpenAI's GPT models with vision capabilities (including GPT-5 and GPT-4o).

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

* **Open Prompt** (`unconstrained`) - Use any prompt to generate a raw response

* **Text Recognition (OCR)** (`ocr`) - Model recognizes text in the image

* **Visual Question Answering** (`visual-question-answering`) - Model answers the question you submit in the prompt

* **Captioning (short)** (`caption`) - Model provides a short description of the image

* **Captioning** (`detailed-caption`) - Model provides a long description of the image

* **Single-Label Classification** (`classification`) - Model classifies the image content as one of the provided classes

* **Multi-Label Classification** (`multi-label-classification`) - Model classifies the image content as one or more of the provided classes

* **Structured Output Generation** (`structured-answering`) - Model returns a JSON response with the specified fields

Provide your OpenAI API key or set the value to ``rf_key:account`` (or
``rf_key:user:<id>``) to proxy requests through Roboflow's API.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/open_ai@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the OpenAI model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your OpenAI API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `image_detail` | `str` | Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response.. | ❌ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit OpenAI limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OpenAI` in version `v3`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Current Time`](current_time.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Path Deviation`](path_deviation.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the OpenAI model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md), [`ROBOFLOW_MANAGED_KEY`](../kinds/roboflow_managed_key.md)]*): Your OpenAI API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `image_detail` (*[`string`](../kinds/string.md)*): Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity..
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `OpenAI` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/open_ai@v3",
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
	    "model_version": "gpt-5",
	    "image_detail": "auto",
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




## v2

??? "Class: `OpenAIBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/openai/v2.py">inference.core.workflows.core_steps.models.foundation.openai.v2.OpenAIBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to OpenAI's GPT models with vision capabilities (including GPT-4o and GPT-5).

You can specify arbitrary text prompts or predefined ones, the block supports the following types of prompt:

* **Open Prompt** (`unconstrained`) - Use any prompt to generate a raw response

* **Text Recognition (OCR)** (`ocr`) - Model recognizes text in the image

* **Visual Question Answering** (`visual-question-answering`) - Model answers the question you submit in the prompt

* **Captioning (short)** (`caption`) - Model provides a short description of the image

* **Captioning** (`detailed-caption`) - Model provides a long description of the image

* **Single-Label Classification** (`classification`) - Model classifies the image content as one of the provided classes

* **Multi-Label Classification** (`multi-label-classification`) - Model classifies the image content as one or more of the provided classes

* **Structured Output Generation** (`structured-answering`) - Model returns a JSON response with the specified fields

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/open_ai@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the OpenAI model. | ✅ |
| `output_structure` | `Dict[str, str]` | Dictionary with structure of expected JSON response. | ❌ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `api_key` | `str` | Your OpenAI API key. | ✅ |
| `model_version` | `str` | Model to be used. | ✅ |
| `image_detail` | `str` | Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response.. | ❌ |
| `temperature` | `float` | Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are.. | ✅ |
| `max_concurrent_requests` | `int` | Number of concurrent requests that can be executed by block when batch of input images provided. If not given - block defaults to value configured globally in Workflows Execution Engine. Please restrict if you hit OpenAI limits.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OpenAI` in version `v2`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Current Time`](current_time.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Path Deviation`](path_deviation.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the OpenAI model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Your OpenAI API key.
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.
        - `image_detail` (*[`string`](../kinds/string.md)*): Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity..
        - `temperature` (*[`float`](../kinds/float.md)*): Temperature to sample from the model - value in range 0.0-2.0, the higher - the more random / "creative" the generations are..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `OpenAI` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/open_ai@v2",
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
	    "model_version": "gpt-4o",
	    "image_detail": "auto",
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




## v1

??? "Class: `OpenAIBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/openai/v1.py">inference.core.workflows.core_steps.models.foundation.openai.v1.OpenAIBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Ask a question to OpenAI's GPT-4 with Vision model.

You can specify arbitrary text prompts to the OpenAIBlock.

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 

_This model was previously part of the LMM block._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/open_ai@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Text prompt to the OpenAI model. | ✅ |
| `openai_api_key` | `str` | Your OpenAI API key. | ✅ |
| `openai_model` | `str` | Model to be used. | ✅ |
| `json_output_format` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |
| `image_detail` | `str` | Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity.. | ✅ |
| `max_tokens` | `int` | Maximum number of tokens the model can generate in it's response.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OpenAI` in version `v1`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Current Time`](current_time.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`Detection Offset`](detection_offset.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`SIFT`](sift.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Property Definition`](property_definition.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Data Aggregator`](data_aggregator.md), [`Path Deviation`](path_deviation.md), [`Blur Visualization`](blur_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`GLM-OCR`](glmocr.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Label Visualization`](label_visualization.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Merge`](detections_merge.md), [`Line Counter`](line_counter.md), [`Detections Filter`](detections_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Overlap Analysis`](overlap_analysis.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Expression`](expression.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Corner Visualization`](corner_visualization.md), [`Continue If`](continue_if.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dominant Color`](dominant_color.md), [`Delta Filter`](delta_filter.md), [`Mask Visualization`](mask_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Switch Case`](switch_case.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Combine`](detections_combine.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Identify Changes`](identify_changes.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the OpenAI model.
        - `openai_api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Your OpenAI API key.
        - `openai_model` (*[`string`](../kinds/string.md)*): Model to be used.
        - `image_detail` (*[`string`](../kinds/string.md)*): Indicates the image's quality, with 'high' suggesting it is of high resolution and should be processed or displayed with high fidelity..

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `OpenAI` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/open_ai@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "openai_api_key": "xxx-xxx",
	    "openai_model": "gpt-4o",
	    "json_output_format": {
	        "count": "number of cats in the picture"
	    },
	    "image_detail": "auto",
	    "max_tokens": 450
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

