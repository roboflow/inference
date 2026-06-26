
# PLC Writer



??? "Class: `PLCWriterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/plc/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1.PLCWriterBlockV1</a>
    



The **PLC Writer** block writes a single tag value to a PLC. To write several tags, add one
PLC Writer block per tag (each is its own request to the PLC).

This block can reach the PLC three ways, selected by **Connection mode** in the advanced
section:

- **Roboflow PLC Relay** (default): sends tags to the on-device **PLC Relay** service over
  HTTP. The relay owns the protocol (Allen-Bradley, Modbus, or Siemens S7), the device IP,
  and the tag schema, so the same Workflow runs unchanged across devices. Tags are sent in
  a single batch request per frame over a persistent keep-alive connection (high FPS).
- **Direct (EtherNet/IP)**: connects straight to the PLC with `pylogix`. Tags are
  addressed by name (e.g. `Program:MainProgram.Tag1`).
- **Direct (Modbus TCP)**: connects straight to the PLC with `pymodbus`. Tags are
  addressed as `area:address` (`holding:100`, `coil:0`, `input:5`, `discrete:2`); a bare
  number defaults to a holding register.

**Address** is the relay host in relay mode, or the PLC's IP address in either direct
mode. The advanced section exposes the relevant extras per mode (relay port, processor
slot, Modbus port / unit id).

On any failure the error is logged and that tag's entry in the output is set to
`"ReadFailure"` / `"WriteFailure"`; `error_status` is `True` if any tag failed.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/plc_writer@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `tag` | `str` | The single PLC tag to write. Relay and Direct (EtherNet/IP) modes use a tag name (e.g. `camera_fault`); Direct (Modbus TCP) mode uses `area:address` (`holding:100`, `coil:0`; a bare number is a holding register, and only `holding` registers and `coil`s are writable, not the read-only `input` / `discrete` areas). To write several tags, add one PLC Writer block per tag.. | ✅ |
| `value` | `Union[bool, float, int, str]` | The value to write to the tag. May be a fixed value or a reference to a workflow input or a previous step's output. Must be a boolean, integer, or float, except Direct (EtherNet/IP) mode, which also accepts strings (for Logix STRING tags).. | ✅ |
| `ip_address` | `str` | Address of the PLC Relay (relay mode) or of the PLC itself (direct modes). A bare host/IP is accepted; in relay mode a full URL may also be given.. | ✅ |
| `connection_mode` | `str` | How to reach the PLC: through the on-device PLC Relay, or directly over EtherNet/IP or Modbus TCP.. | ❌ |
| `relay_port` | `int` | Port of the PLC Relay service (relay mode).. | ✅ |
| `request_timeout` | `int` | Read timeout in seconds for each request to the PLC Relay service (relay mode). This must cover the relay's synchronous PLC batch transaction, which can run for seconds against a slow or disconnected PLC (especially Modbus / S7); if it is exceeded the request is abandoned and every tag in the batch is reported as a failure. Raise it for slow or flaky PLCs. (Connecting to the relay itself uses a separate short timeout, so a down relay still fails fast.). | ✅ |
| `processor_slot` | `int` | EtherNet/IP processor slot of the PLC (direct EtherNet/IP mode).. | ✅ |
| `modbus_port` | `int` | Modbus TCP port of the PLC (direct Modbus mode).. | ✅ |
| `modbus_unit_id` | `int` | Modbus unit / slave id of the PLC (direct Modbus mode).. | ✅ |
| `disable_sink` | `bool` | If True, skip the write to the PLC and return an empty result.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `PLC Writer` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Track Class Lock`](track_class_lock.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`QR Code Generator`](qr_code_generator.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Delta Filter`](delta_filter.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Property Definition`](property_definition.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Camera Calibration`](camera_calibration.md), [`Dominant Color`](dominant_color.md), [`Detections Filter`](detections_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Continue If`](continue_if.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Byte Tracker`](byte_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Overlap Analysis`](overlap_analysis.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Switch Case`](switch_case.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`PLC Writer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `tag` (*[`string`](../kinds/string.md)*): The single PLC tag to write. Relay and Direct (EtherNet/IP) modes use a tag name (e.g. `camera_fault`); Direct (Modbus TCP) mode uses `area:address` (`holding:100`, `coil:0`; a bare number is a holding register, and only `holding` registers and `coil`s are writable, not the read-only `input` / `discrete` areas). To write several tags, add one PLC Writer block per tag..
        - `value` (*[`*`](../kinds/wildcard.md)*): The value to write to the tag. May be a fixed value or a reference to a workflow input or a previous step's output. Must be a boolean, integer, or float, except Direct (EtherNet/IP) mode, which also accepts strings (for Logix STRING tags)..
        - `depends_on` (*[`*`](../kinds/wildcard.md)*): Optional reference to a step this write should run after, for when the write order matters but the tag/value are not themselves derived from that step. Dependencies are otherwise inferred from selector-valued `tag` / `value`, so this is not needed for input- or step-driven writes..
        - `ip_address` (*[`string`](../kinds/string.md)*): Address of the PLC Relay (relay mode) or of the PLC itself (direct modes). A bare host/IP is accepted; in relay mode a full URL may also be given..
        - `relay_port` (*[`integer`](../kinds/integer.md)*): Port of the PLC Relay service (relay mode)..
        - `request_timeout` (*[`integer`](../kinds/integer.md)*): Read timeout in seconds for each request to the PLC Relay service (relay mode). This must cover the relay's synchronous PLC batch transaction, which can run for seconds against a slow or disconnected PLC (especially Modbus / S7); if it is exceeded the request is abandoned and every tag in the batch is reported as a failure. Raise it for slow or flaky PLCs. (Connecting to the relay itself uses a separate short timeout, so a down relay still fails fast.).
        - `processor_slot` (*[`integer`](../kinds/integer.md)*): EtherNet/IP processor slot of the PLC (direct EtherNet/IP mode)..
        - `modbus_port` (*[`integer`](../kinds/integer.md)*): Modbus TCP port of the PLC (direct Modbus mode)..
        - `modbus_unit_id` (*[`integer`](../kinds/integer.md)*): Modbus unit / slave id of the PLC (direct Modbus mode)..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, skip the write to the PLC and return an empty result..

    - output
    
        - `write_result` ([`string`](../kinds/string.md)): String value.
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `PLC Writer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/plc_writer@v1",
	    "tag": "camera_fault",
	    "value": true,
	    "depends_on": "$steps.some_previous_step",
	    "ip_address": "127.0.0.1",
	    "connection_mode": "relay",
	    "relay_port": 8007,
	    "request_timeout": 10,
	    "processor_slot": 0,
	    "modbus_port": 502,
	    "modbus_unit_id": 1,
	    "disable_sink": false
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

