
# PLC Reader



??? "Class: `PLCReaderBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/plc/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.plc.v1.PLCReaderBlockV1</a>
    



The **PLC Reader** block reads tag values from a PLC and makes them available to the rest
of the Workflow.

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

Use the following identifier in step `"type"` field: `roboflow_core/plc_reader@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `tags_to_read` | `List[str]` | PLC tags to read, entered comma-separated (e.g. `camera_msg, sku_number`). Relay and Direct (EtherNet/IP) modes use tag names. Direct (Modbus TCP) mode uses `area:address`, where area is `holding`, `input` (read-only), `coil`, or `discrete` (read-only); a bare number means a holding register (`100` = `holding:100`). Example for Modbus: `holding:100, coil:0`.. | ✅ |
| `ip_address` | `str` | Address of the PLC Relay (relay mode) or of the PLC itself (direct modes). A bare host/IP is accepted; in relay mode a full URL may also be given.. | ✅ |
| `connection_mode` | `str` | How to reach the PLC: through the on-device PLC Relay, or directly over EtherNet/IP or Modbus TCP.. | ❌ |
| `relay_port` | `int` | Port of the PLC Relay service (relay mode).. | ✅ |
| `request_timeout` | `int` | Read timeout in seconds for each request to the PLC Relay service (relay mode). This must cover the relay's synchronous PLC batch transaction, which can run for seconds against a slow or disconnected PLC (especially Modbus / S7); if it is exceeded the request is abandoned and every tag in the batch is reported as a failure. Raise it for slow or flaky PLCs. (Connecting to the relay itself uses a separate short timeout, so a down relay still fails fast.). | ✅ |
| `processor_slot` | `int` | EtherNet/IP processor slot of the PLC (direct EtherNet/IP mode).. | ✅ |
| `modbus_port` | `int` | Modbus TCP port of the PLC (direct Modbus mode).. | ✅ |
| `modbus_unit_id` | `int` | Modbus unit / slave id of the PLC (direct Modbus mode).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `PLC Reader` in version `v1`.

    - inputs: [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Qwen-VL`](qwen_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Stack`](image_stack.md), [`OpenRouter`](open_router.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dynamic Zone`](dynamic_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Dimension Collapse`](dimension_collapse.md), [`Size Measurement`](size_measurement.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`PLC Reader` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `tags_to_read` (*[`list_of_values`](../kinds/list_of_values.md)*): PLC tags to read, entered comma-separated (e.g. `camera_msg, sku_number`). Relay and Direct (EtherNet/IP) modes use tag names. Direct (Modbus TCP) mode uses `area:address`, where area is `holding`, `input` (read-only), `coil`, or `discrete` (read-only); a bare number means a holding register (`100` = `holding:100`). Example for Modbus: `holding:100, coil:0`..
        - `ip_address` (*[`string`](../kinds/string.md)*): Address of the PLC Relay (relay mode) or of the PLC itself (direct modes). A bare host/IP is accepted; in relay mode a full URL may also be given..
        - `relay_port` (*[`integer`](../kinds/integer.md)*): Port of the PLC Relay service (relay mode)..
        - `request_timeout` (*[`integer`](../kinds/integer.md)*): Read timeout in seconds for each request to the PLC Relay service (relay mode). This must cover the relay's synchronous PLC batch transaction, which can run for seconds against a slow or disconnected PLC (especially Modbus / S7); if it is exceeded the request is abandoned and every tag in the batch is reported as a failure. Raise it for slow or flaky PLCs. (Connecting to the relay itself uses a separate short timeout, so a down relay still fails fast.).
        - `processor_slot` (*[`integer`](../kinds/integer.md)*): EtherNet/IP processor slot of the PLC (direct EtherNet/IP mode)..
        - `modbus_port` (*[`integer`](../kinds/integer.md)*): Modbus TCP port of the PLC (direct Modbus mode)..
        - `modbus_unit_id` (*[`integer`](../kinds/integer.md)*): Modbus unit / slave id of the PLC (direct Modbus mode)..

    - output
    
        - `tag_values` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `PLC Reader` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/plc_reader@v1",
	    "tags_to_read": [
	        "camera_msg",
	        "sku_number"
	    ],
	    "ip_address": "127.0.0.1",
	    "connection_mode": "relay",
	    "relay_port": 8007,
	    "request_timeout": 10,
	    "processor_slot": 0,
	    "modbus_port": 502,
	    "modbus_unit_id": 1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

