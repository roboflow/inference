
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

    - inputs: [`Absolute Static Crop`](absolute_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Distance Measurement`](distance_measurement.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Delta Filter`](delta_filter.md), [`Blur Visualization`](blur_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`QR Code Detection`](qr_code_detection.md), [`SAM 3`](sam3.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Property Definition`](property_definition.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Mask Visualization`](mask_visualization.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Cache Get`](cache_get.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Stitch Images`](stitch_images.md), [`Google Gemini`](google_gemini.md), [`QR Code Generator`](qr_code_generator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Line Counter`](line_counter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Rate Limiter`](rate_limiter.md), [`Switch Case`](switch_case.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Inner Workflow`](inner_workflow.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Expression`](expression.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Local File Sink`](local_file_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SORT Tracker`](sort_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Google Gemini`](google_gemini.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenRouter`](open_router.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`SAM 3`](sam3.md), [`Cache Set`](cache_set.md), [`PLC Reader`](plc_reader.md), [`Size Measurement`](size_measurement.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
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

