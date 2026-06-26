
# PLC EthernetIP

!!! warning "Deprecated"

    This block is deprecated. Use the PLC Reader / PLC Writer blocks (set Connection mode to 'Direct - EtherNet/IP') instead. Note the outputs differ: instead of a single `plc_results` list, the PLC Reader returns `tag_values` (a tag->value dict) and the PLC Writer returns `write_result`, each alongside an `error_status` flag.



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

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Slack Notification`](slack_notification.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Rate Limiter`](rate_limiter.md), [`Local File Sink`](local_file_sink.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Camera Calibration`](camera_calibration.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Property Definition`](property_definition.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`PLC Reader`](plc_reader.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Continue If`](continue_if.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Dominant Color`](dominant_color.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Switch Case`](switch_case.md), [`Expression`](expression.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OpenAI`](open_ai.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Buffer`](buffer.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md)

    
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

