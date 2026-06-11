
# PLC EthernetIP



??? "Class: `PLCBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/PLCethernetIP/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.PLCethernetIP.v1.PLCBlockV1</a>
    



This **PLC Communication** block integrates a Roboflow Workflow with a PLC using Ethernet/IP communication.
It can:
- Read tags from a PLC if `mode='read'`.
- Write tags to a PLC if `mode='write'`.
- Perform both read and write in a single run if `mode='read_and_write'`.

**Parameters depending on mode:**
- If `mode='read'` or `mode='read_and_write'`, `tags_to_read` must be provided.
- If `mode='write'` or `mode='read_and_write'`, `tags_to_write` must be provided.

If a read or write operation fails, an error message is printed to the terminal, 
and the corresponding entry in the output dictionary is set to a generic "ReadFailure" or "WriteFailure" message.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sinks@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `plc_ip` | `str` | IP address of the target PLC.. | ✅ |
| `mode` | `str` | Mode of operation: 'read', 'write', or 'read_and_write'.. | ❌ |
| `tags_to_read` | `List[str]` | List of PLC tag names to read. Applicable if mode='read' or mode='read_and_write'.. | ✅ |
| `tags_to_write` | `Dict[str, Union[float, int, str]]` | Dictionary of tags and the values to write. Applicable if mode='write' or mode='read_and_write'.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `PLC EthernetIP` in version `v1`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`Detections Stitch`](detections_stitch.md), [`Overlap Analysis`](overlap_analysis.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Continue If`](continue_if.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Detections Filter`](detections_filter.md), [`Property Definition`](property_definition.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SmolVLM2`](smol_vlm2.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Byte Tracker`](byte_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`CSV Formatter`](csv_formatter.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Detections Merge`](detections_merge.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Color Visualization`](color_visualization.md), [`Path Deviation`](path_deviation.md), [`Seg Preview`](seg_preview.md), [`Cache Set`](cache_set.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Size Measurement`](size_measurement.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`PLC EthernetIP` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `plc_ip` (*[`string`](../kinds/string.md)*): IP address of the target PLC..
        - `tags_to_read` (*[`list_of_values`](../kinds/list_of_values.md)*): List of PLC tag names to read. Applicable if mode='read' or mode='read_and_write'..
        - `tags_to_write` (*[`dictionary`](../kinds/dictionary.md)*): Dictionary of tags and the values to write. Applicable if mode='write' or mode='read_and_write'..
        - `depends_on` (*[`*`](../kinds/wildcard.md)*): Reference to the step output this block depends on..

    - output
    
        - `plc_results` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `PLC EthernetIP` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sinks@v1",
	    "plc_ip": "192.168.1.10",
	    "mode": "read",
	    "tags_to_read": [
	        "camera_msg",
	        "sku_number"
	    ],
	    "tags_to_write": {
	        "camera_fault": true,
	        "defect_count": 5
	    },
	    "depends_on": "$steps.some_previous_step"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

