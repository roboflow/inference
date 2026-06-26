
# OPC UA Writer Sink



??? "Class: `OPCWriterSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/opc_writer/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.opc_writer.v1.OPCWriterSinkBlockV1</a>
    



The **OPC UA Writer** block enables you to write data to a variable on an OPC UA server, leveraging the 
[asyncua](https://github.com/FreeOpcUa/opcua-asyncio) library for seamless communication.

### Supported Data Types
This block supports writing the following data types to OPC UA server variables:
- Numbers (integers, floats)
- Booleans
- Strings

**Note:** The data type you send must match the expected type of the target OPC UA variable.

### Node Lookup Mode
The block supports two methods for locating OPC UA nodes via the `node_lookup_mode` parameter:

- **`hierarchical` (default)**: Uses standard OPC UA hierarchical path navigation. The block navigates
  through the address space using `get_child()`. Each component in the `object_name` path is
  automatically prefixed with the namespace index.
  - **Example**: `object_name="Roboflow/Crane_11"` → path `0:Objects/2:Roboflow/2:Crane_11/2:Variable`
  - **Best for**: Traditional OPC UA servers with hierarchical address spaces

- **`direct`**: Uses direct NodeId string access. The block constructs a NodeId as
  `ns={namespace};s={object_name}/{variable_name}` and accesses it directly via `get_node()`.
  - **Example**: `object_name="[Sample_Tags]/Ramp"` → NodeId `ns=2;s=[Sample_Tags]/Ramp/South_Person_Count`
  - **Best for**: Ignition SCADA systems and other servers using string-based NodeId identifiers

### Cooldown
To prevent excessive traffic to the OPC UA server, the block includes a `cooldown_seconds` parameter, 
which defaults to **5 seconds**. During the cooldown period:
- Consecutive executions of the block will set the `throttling_status` output to `True`.
- No data will be sent to the server.

You can customize the `cooldown_seconds` parameter based on your needs. Setting it to `0` disables 
the cooldown entirely.

### Asynchronous Execution
The block provides a `fire_and_forget` property for asynchronous execution:
- **When `fire_and_forget=True`**: The block sends data in the background, allowing the Workflow to 
  proceed immediately. However, the `error_status` output will always be set to `False`, so we do not 
  recommend this mode for debugging.
- **When `fire_and_forget=False`**: The block waits for confirmation before proceeding, ensuring errors 
  are captured in the `error_status` output.

### Disabling the Block Dynamically
You can disable the **OPC UA Writer** block during execution by linking the `disable_sink` parameter 
to a Workflow input. By providing a specific input value, you can dynamically prevent the block from 
executing.

### Connection Pooling
The block uses a connection pool to efficiently manage OPC UA connections. Instead of creating a new
connection for each write operation, connections are reused across multiple writes to the same server.
This significantly reduces latency and resource usage for high-frequency write scenarios.

- Connections are automatically pooled per server URL and username combination
- If a connection fails during a write operation, it is automatically invalidated and a new connection
  is established on the next write attempt

### Retry Logic
The block includes configurable retry logic with exponential backoff for handling transient connection failures:

- `max_retries`: Number of connection attempts before giving up (default: 3)
- `retry_backoff_seconds`: Base delay between retries in seconds (default: 1.0). The delay doubles
  after each failed attempt (exponential backoff).

**Note:** Authentication errors (wrong username/password) are not retried as they will continue to fail.

### Cooldown Limitations
!!! warning "Cooldown Limitations"
    The cooldown feature is optimized for workflows involving video processing.
    - In other contexts, such as Workflows triggered by HTTP services (e.g., Roboflow Hosted API,
      Dedicated Deployment, or self-hosted `Inference` server), the cooldown timer will not be applied effectively.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_enterprise/opc_writer_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `url` | `str` | URL of the OPC UA server to which data will be written.. | ✅ |
| `namespace` | `str` | The OPC UA namespace URI or index used to locate objects and variables.. | ✅ |
| `user_name` | `str` | Optional username for authentication when connecting to the OPC UA server.. | ✅ |
| `password` | `str` | Optional password for authentication when connecting to the OPC UA server.. | ✅ |
| `object_name` | `str` | The name of the target object in the namespace to search for.. | ✅ |
| `variable_name` | `str` | The name of the variable within the target object to be updated.. | ✅ |
| `value` | `Union[bool, float, int, str]` | The value to be written to the target variable on the OPC UA server.. | ✅ |
| `value_type` | `str` | The type of the value to be written to the target variable on the OPC UA server. Supported types: Boolean, Double, Float, Int16, Int32, Int64, Integer (Int64 alias), SByte, String, UInt16, UInt32, UInt64.. | ✅ |
| `timeout` | `int` | The number of seconds to wait for a response from the OPC UA server before timing out.. | ✅ |
| `fire_and_forget` | `bool` | Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling.. | ✅ |
| `disable_sink` | `bool` | Boolean flag to disable block execution.. | ✅ |
| `cooldown_seconds` | `int` | The minimum number of seconds to wait between consecutive updates to the OPC UA server.. | ✅ |
| `node_lookup_mode` | `str` | Method to locate the OPC UA node: 'hierarchical' uses path navigation, 'direct' uses NodeId strings (for Ignition-style string-based tags).. | ✅ |
| `max_retries` | `int` | Maximum number of connection attempts before giving up. Default is 3 with exponential backoff starting at 15ms.. | ✅ |
| `retry_backoff_seconds` | `float` | Base delay between retry attempts in seconds (doubles each retry). Default is 0.015 (15ms) for fast exponential backoff.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OPC UA Writer Sink` in version `v1`.

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Distance Measurement`](distance_measurement.md), [`Identify Changes`](identify_changes.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`CogVLM`](cog_vlm.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`Image Stack`](image_stack.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`S3 Sink`](s3_sink.md), [`Pixel Color Count`](pixel_color_count.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Path Deviation`](path_deviation.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OPC UA Writer Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `url` (*[`string`](../kinds/string.md)*): URL of the OPC UA server to which data will be written..
        - `namespace` (*[`string`](../kinds/string.md)*): The OPC UA namespace URI or index used to locate objects and variables..
        - `user_name` (*[`string`](../kinds/string.md)*): Optional username for authentication when connecting to the OPC UA server..
        - `password` (*[`string`](../kinds/string.md)*): Optional password for authentication when connecting to the OPC UA server..
        - `object_name` (*[`string`](../kinds/string.md)*): The name of the target object in the namespace to search for..
        - `variable_name` (*[`string`](../kinds/string.md)*): The name of the variable within the target object to be updated..
        - `value` (*Union[[`float`](../kinds/float.md), [`string`](../kinds/string.md), [`boolean`](../kinds/boolean.md), [`integer`](../kinds/integer.md)]*): The value to be written to the target variable on the OPC UA server..
        - `value_type` (*[`string`](../kinds/string.md)*): The type of the value to be written to the target variable on the OPC UA server. Supported types: Boolean, Double, Float, Int16, Int32, Int64, Integer (Int64 alias), SByte, String, UInt16, UInt32, UInt64..
        - `timeout` (*[`integer`](../kinds/integer.md)*): The number of seconds to wait for a response from the OPC UA server before timing out..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable block execution..
        - `cooldown_seconds` (*[`integer`](../kinds/integer.md)*): The minimum number of seconds to wait between consecutive updates to the OPC UA server..
        - `node_lookup_mode` (*[`string`](../kinds/string.md)*): Method to locate the OPC UA node: 'hierarchical' uses path navigation, 'direct' uses NodeId strings (for Ignition-style string-based tags)..
        - `max_retries` (*[`integer`](../kinds/integer.md)*): Maximum number of connection attempts before giving up. Default is 3 with exponential backoff starting at 15ms..
        - `retry_backoff_seconds` (*[`float`](../kinds/float.md)*): Base delay between retry attempts in seconds (doubles each retry). Default is 0.015 (15ms) for fast exponential backoff..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `disabled` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `OPC UA Writer Sink` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_enterprise/opc_writer_sink@v1",
	    "url": "opc.tcp://localhost:4840/freeopcua/server/",
	    "namespace": "http://examples.freeopcua.github.io",
	    "user_name": "John",
	    "password": "secret",
	    "object_name": "Line1",
	    "variable_name": "InspectionSuccess",
	    "value": "running",
	    "value_type": "Boolean",
	    "timeout": 10,
	    "fire_and_forget": true,
	    "disable_sink": false,
	    "cooldown_seconds": 10,
	    "node_lookup_mode": "hierarchical",
	    "max_retries": 1,
	    "retry_backoff_seconds": 0.015
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

