
# PLC ModbusTCP

!!! warning "Deprecated"

    This block is deprecated. Use the PLC Reader / PLC Writer blocks (set Connection mode to 'Direct - Modbus') instead. Note the outputs differ: instead of a single `modbus_results` list, the PLC Reader returns `tag_values` (a tag->value dict) and the PLC Writer returns `write_result`, each alongside an `error_status` flag.



??? "Class: `ModbusTCPBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/PLC_modbus/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.PLC_modbus.v1.ModbusTCPBlockV1</a>
    



This **Modbus TCP** block integrates a Roboflow Workflow with a PLC using Modbus TCP.
It can:
- Read registers from a PLC if `mode='read'`.
- Write registers to a PLC if `mode='write'`.
- Perform both read and write in a single run if `mode='read_and_write'`.

**Parameters depending on mode:**
- If `mode='read'` or `mode='read_and_write'`, `registers_to_read` must be provided as a list of register addresses.
- If `mode='write'` or `mode='read_and_write'`, `registers_to_write` must be provided as a dictionary mapping register addresses to values.

If a read or write operation fails, an error message is printed to the terminal, 
and the corresponding entry in the output dictionary is set to "ReadFailure" or "WriteFailure".


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/modbus_tcp@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `plc_ip` | `str` | IP address of the target PLC.. | ✅ |
| `plc_port` | `int` | Port number for Modbus TCP communication.. | ❌ |
| `mode` | `str` | Mode of operation: 'read', 'write', or 'read_and_write'.. | ❌ |
| `registers_to_read` | `List[int]` | List of register addresses to read. Applicable if mode='read' or 'read_and_write'.. | ✅ |
| `registers_to_write` | `Dict[str, int]` | Dictionary mapping register addresses to values to write. Applicable if mode='write' or 'read_and_write'.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `PLC ModbusTCP` in version `v1`.

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Cosine Similarity`](cosine_similarity.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Property Definition`](property_definition.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`QR Code Detection`](qr_code_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen3.5`](qwen3.5.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Continue If`](continue_if.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Dynamic Crop`](dynamic_crop.md), [`Overlap Analysis`](overlap_analysis.md), [`Dominant Color`](dominant_color.md), [`Halo Visualization`](halo_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Track Class Lock`](track_class_lock.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`Google Gemma API`](google_gemma_api.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Switch Case`](switch_case.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Expression`](expression.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`Current Time`](current_time.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Buffer`](buffer.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`PLC ModbusTCP` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `plc_ip` (*[`string`](../kinds/string.md)*): IP address of the target PLC..
        - `registers_to_read` (*[`list_of_values`](../kinds/list_of_values.md)*): List of register addresses to read. Applicable if mode='read' or 'read_and_write'..
        - `registers_to_write` (*[`list_of_values`](../kinds/list_of_values.md)*): Dictionary mapping register addresses to values to write. Applicable if mode='write' or 'read_and_write'..
        - `depends_on` (*[`*`](../kinds/wildcard.md)*): Reference to the step output this block depends on..

    - output
    
        - `modbus_results` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `PLC ModbusTCP` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/modbus_tcp@v1",
	    "plc_ip": "10.0.1.31",
	    "plc_port": 502,
	    "mode": "read",
	    "registers_to_read": [
	        1000,
	        1001
	    ],
	    "registers_to_write": {
	        "1005": 25
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

