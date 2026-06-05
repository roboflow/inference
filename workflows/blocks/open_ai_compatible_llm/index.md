
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

    - inputs: [`Image Preprocessing`](image_preprocessing.md), [`Detections Transformation`](detections_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Image Threshold`](image_threshold.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Crop Visualization`](crop_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Absolute Static Crop`](absolute_static_crop.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OCR Model`](ocr_model.md), [`Color Visualization`](color_visualization.md), [`Gaze Detection`](gaze_detection.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Blur Visualization`](blur_visualization.md), [`SAM 3`](sam3.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Detections Consensus`](detections_consensus.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Delta Filter`](delta_filter.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Seg Preview`](seg_preview.md), [`Overlap Filter`](overlap_filter.md), [`Buffer`](buffer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dominant Color`](dominant_color.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Qwen3-VL`](qwen3_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Background Subtraction`](background_subtraction.md), [`GLM-OCR`](glmocr.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Property Definition`](property_definition.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Data Aggregator`](data_aggregator.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Pixel Color Count`](pixel_color_count.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen-VL`](qwen_vl.md), [`CogVLM`](cog_vlm.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Merge`](detections_merge.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dimension Collapse`](dimension_collapse.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Environment Secrets Store`](environment_secrets_store.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Identify Changes`](identify_changes.md), [`Depth Estimation`](depth_estimation.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Identify Outliers`](identify_outliers.md), [`Inner Workflow`](inner_workflow.md), [`Time in Zone`](timein_zone.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Size Measurement`](size_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Continue If`](continue_if.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Line Counter`](line_counter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Barcode Detection`](barcode_detection.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Google Gemma`](google_gemma.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`QR Code Detection`](qr_code_detection.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Image Contours`](image_contours.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OpenAI-Compatible LLM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `base_url` (*[`string`](../kinds/string.md)*): URL of the OpenAI-compatible server, including /v1..
        - `model_name` (*[`string`](../kinds/string.md)*): Model identifier sent to the server..
        - `api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): API key, if the endpoint requires one..
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

