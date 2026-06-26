
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

    - inputs: [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Color Visualization`](color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Text Display`](text_display.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Blur Visualization`](blur_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Byte Tracker`](byte_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Expression`](expression.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Combine`](detections_combine.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Delta Filter`](delta_filter.md), [`Size Measurement`](size_measurement.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Byte Tracker`](byte_tracker.md), [`Switch Case`](switch_case.md), [`Dominant Color`](dominant_color.md), [`Rate Limiter`](rate_limiter.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Image Slicer`](image_slicer.md), [`Byte Tracker`](byte_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Overlap Analysis`](overlap_analysis.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen3.5`](qwen3.5.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Seg Preview`](seg_preview.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Path Deviation`](path_deviation.md), [`Trace Visualization`](trace_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Clip Comparison`](clip_comparison.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Google Gemma`](google_gemma.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Filter`](detections_filter.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Barcode Detection`](barcode_detection.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cosine Similarity`](cosine_similarity.md), [`Image Contours`](image_contours.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Image Slicer`](image_slicer.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`Detections Transformation`](detections_transformation.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Property Definition`](property_definition.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`CogVLM`](cog_vlm.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`LMM`](lmm.md), [`Continue If`](continue_if.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`EasyOCR`](easy_ocr.md), [`Cache Get`](cache_get.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)
    - outputs: [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Calibration`](camera_calibration.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Label Visualization`](label_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
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

