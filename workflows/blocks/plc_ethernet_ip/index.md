
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

    - inputs: [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PLC Writer`](plc_writer.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Seg Preview`](seg_preview.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Cache Set`](cache_set.md), [`OpenAI`](open_ai.md), [`Data Aggregator`](data_aggregator.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Buffer`](buffer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Google Gemini`](google_gemini.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Delta Filter`](delta_filter.md), [`Object Detection Model`](object_detection_model.md), [`Expression`](expression.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PP-OCR`](ppocr.md), [`Clip Comparison`](clip_comparison.md), [`Blur Visualization`](blur_visualization.md), [`EasyOCR`](easy_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CSV Formatter`](csv_formatter.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Motion Detection`](motion_detection.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Rate Limiter`](rate_limiter.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Current Time`](current_time.md), [`Trace Visualization`](trace_visualization.md), [`Cache Get`](cache_get.md), [`PLC Reader`](plc_reader.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Image Stack`](image_stack.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Continue If`](continue_if.md), [`Webhook Sink`](webhook_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`SmolVLM2`](smol_vlm2.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Relative Static Crop`](relative_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`SIFT`](sift.md), [`Dominant Color`](dominant_color.md), [`Moondream2`](moondream2.md), [`Property Definition`](property_definition.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Identify Changes`](identify_changes.md), [`Overlap Analysis`](overlap_analysis.md), [`Email Notification`](email_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Velocity`](velocity.md), [`Image Threshold`](image_threshold.md), [`CogVLM`](cog_vlm.md), [`MQTT Writer`](mqtt_writer.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Event Writer`](event_writer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Slack Notification`](slack_notification.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections Transformation`](detections_transformation.md), [`Size Measurement`](size_measurement.md), [`Local File Sink`](local_file_sink.md), [`Inner Workflow`](inner_workflow.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Outliers`](identify_outliers.md), [`Detection Offset`](detection_offset.md), [`Switch Case`](switch_case.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`S3 Sink`](s3_sink.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detection Event Log`](detection_event_log.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Corner Visualization`](corner_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Bounding Rectangle`](bounding_rectangle.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)
    - outputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`PLC Reader`](plc_reader.md), [`Google Gemini`](google_gemini.md), [`Google Gemini`](google_gemini.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Seg Preview`](seg_preview.md), [`SAM 3`](sam3.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Cache Set`](cache_set.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Path Deviation`](path_deviation.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`OpenRouter`](open_router.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Size Measurement`](size_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md)

    
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

