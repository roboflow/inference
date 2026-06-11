
# OpenAI-Compatible LLM



??? "Class: `OpenAICompatibleBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/openai_compatible/v1.py">inference.core.workflows.core_steps.models.foundation.openai_compatible.v1.OpenAICompatibleBlockV1</a>
    



Send a prompt to any OpenAI-compatible API endpoint (e.g. local Qwen, vLLM, Ollama,
LM Studio, or any service that implements the OpenAI chat completions API).

## How this block works

1. You provide a **Base URL** (e.g. `http://localhost:8000/v1`) and a **Model Name**.
2. Write an **Instruction** — the text the model receives.
3. Add rows under **Inputs** to feed step outputs (images, detections, text) into
   the request. Image inputs are base64-encoded and sent as vision content parts.
   A list of images becomes one vision part per image.
4. Non-image inputs are converted to strings. To splice them into the instruction
   text, reference the input by name with the placeholder syntax shown in the
   Instruction field's help text.
5. Optionally apply **UQL operations** to transform input values before insertion.

## Image handling

- A `WorkflowImageData` value is JPEG-encoded and sent as an `image_url` part.
- Raw JPEG `bytes` (e.g. from the Image Stack block) are sent directly.
- A list of either is fanned out into multiple `image_url` parts.

If an image input is also referenced in the instruction text by name, the
placeholder is removed from the text — the image only travels as a vision part.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/openai_compatible@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `base_url` | `str` | URL of the OpenAI-compatible server, including /v1.. | ✅ |
| `model_name` | `str` | Model identifier sent to the server.. | ✅ |
| `api_key` | `str` | API key, if the endpoint requires one.. | ✅ |
| `system_prompt` | `str` | Optional system message that sets model behavior.. | ✅ |
| `prompt` | `str` | Text sent to the model.. | ✅ |
| `prompt_parameters` | `Dict[str, Union[bool, float, int, str]]` | Step outputs to include in the request (images or text).. | ✅ |
| `prompt_parameters_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Optional UQL operations applied to inputs before use.. | ❌ |
| `max_tokens` | `int` | Maximum tokens the model may generate.. | ❌ |
| `temperature` | `float` | Sampling temperature, 0.0 to 2.0.. | ✅ |
| `extra_body` | `Dict[Any, Any]` | Extra JSON forwarded as the OpenAI SDK extra_body argument.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OpenAI-Compatible LLM` in version `v1`.

    - inputs: [`Camera Focus`](camera_focus.md), [`Current Time`](current_time.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`LMM For Classification`](lmm_for_classification.md), [`Barcode Detection`](barcode_detection.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SIFT`](sift.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Distance Measurement`](distance_measurement.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Filter`](detections_filter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`EasyOCR`](easy_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Overlap Analysis`](overlap_analysis.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`SORT Tracker`](sort_tracker.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dynamic Zone`](dynamic_zone.md), [`Continue If`](continue_if.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Property Definition`](property_definition.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`Anthropic Claude`](anthropic_claude.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Detections Combine`](detections_combine.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OCR Model`](ocr_model.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI-Compatible LLM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `base_url` (*[`string`](../kinds/string.md)*): URL of the OpenAI-compatible server, including /v1..
        - `model_name` (*[`string`](../kinds/string.md)*): Model identifier sent to the server..
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): API key, if the endpoint requires one..
        - `system_prompt` (*[`string`](../kinds/string.md)*): Optional system message that sets model behavior..
        - `prompt` (*[`string`](../kinds/string.md)*): Text sent to the model..
        - `prompt_parameters` (*[`*`](../kinds/wildcard.md)*): Step outputs to include in the request (images or text)..
        - `temperature` (*[`float`](../kinds/float.md)*): Sampling temperature, 0.0 to 2.0..

    - output
    
        - `output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `error_status` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `OpenAI-Compatible LLM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/openai_compatible@v1",
	    "base_url": "http://localhost:8000/v1",
	    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
	    "api_key": "xxx-xxx",
	    "system_prompt": "You are a helpful assistant.",
	    "prompt": "Describe what you see in the image.",
	    "prompt_parameters": {
	        "detections": "$steps.model.predictions",
	        "frames": "$steps.image_stack.frames"
	    },
	    "prompt_parameters_operations": {
	        "detections": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "max_tokens": "<block_does_not_provide_example>",
	    "temperature": "<block_does_not_provide_example>",
	    "extra_body": {
	        "chat_template_kwargs": {
	            "enable_thinking": false
	        },
	        "guided_choice": [
	            "A",
	            "B",
	            "C",
	            "D"
	        ]
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

