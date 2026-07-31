
# CogVLM



??? "Class: `CogVLMBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/cog_vlm/v1.py">inference.core.workflows.core_steps.models.foundation.cog_vlm.v1.CogVLMBlockV1</a>
    




!!! Warning "CogVLM reached **End Of Life**"

    Due to dependencies conflicts with newer models and security vulnerabilities discovered in `transformers`
    library patched in the versions of library incompatible with the model we announced End Of Life for CogVLM
    support in `inference`, effective since release `0.38.0`.
    
    We are leaving this block in ecosystem until release `0.42.0` for clients to get informed about change that 
    was introduced.
    
    Starting as of now, all Workflows using the block stop being functional (runtime error will be raised), 
    after inference release `0.42.0` - this block will be removed and Execution Engine will raise compilation 
    error seeing the block in Workflow definition. 


Ask a question to CogVLM, an open source vision-language model.

This model requires a GPU and can only be run on self-hosted devices, and is not available on the Roboflow Hosted API.

_This model was previously part of the LMM block._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/cog_vlm@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Text prompt to the CogVLM model. | ✅ |
| `json_output_format` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `hosted_serverless`; execution `remote`
:   LMM_ENABLED=False on Roboflow Hosted Serverless: the /llm_v1 and /infer/cog_vlm endpoints are not registered, so run_remotely() returns 404.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `CogVLM` in version `v1`.

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CogVLM`](cog_vlm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Email Notification`](email_notification.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Current Time`](current_time.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`SIFT`](sift.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`OCR Model`](ocr_model.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Relative Static Crop`](relative_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Qwen-VL`](qwen_vl.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PP-OCR`](ppocr.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Local File Sink`](local_file_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemma`](google_gemma.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Polygon Visualization`](polygon_visualization.md), [`Cosmos 3`](cosmos3.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Icon Visualization`](icon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Background Subtraction`](background_subtraction.md), [`CSV Formatter`](csv_formatter.md), [`Corner Visualization`](corner_visualization.md), [`Detection Offset`](detection_offset.md), [`JSON Parser`](json_parser.md), [`Property Definition`](property_definition.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Inner Workflow`](inner_workflow.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Icon Visualization`](icon_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Changes`](identify_changes.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Expression`](expression.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Overlap Filter`](overlap_filter.md), [`Cache Get`](cache_get.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SAM 3 Interactive`](sam3_interactive.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Pixel Color Count`](pixel_color_count.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Stitch Images`](stitch_images.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Delta Filter`](delta_filter.md), [`Camera Focus`](camera_focus.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Dominant Color`](dominant_color.md), [`Continue If`](continue_if.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Switch Case`](switch_case.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Detections Combine`](detections_combine.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`EasyOCR`](easy_ocr.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`Data Aggregator`](data_aggregator.md), [`Image Contours`](image_contours.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CogVLM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the CogVLM model.

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `CogVLM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/cog_vlm@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "json_output_format": {
	        "count": "number of cats in the picture"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

