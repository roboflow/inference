
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

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`EasyOCR`](easy_ocr.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Identify Changes`](identify_changes.md), [`CSV Formatter`](csv_formatter.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Google Gemini`](google_gemini.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Cache Get`](cache_get.md), [`Triangle Visualization`](triangle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SAM 3`](sam3.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Path Deviation`](path_deviation.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Current Time`](current_time.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Mask Visualization`](mask_visualization.md)

    
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
        - `value` (*Union[[`boolean`](../kinds/boolean.md), [`integer`](../kinds/integer.md), [`float`](../kinds/float.md), [`string`](../kinds/string.md)]*): The value to be written to the target variable on the OPC UA server..
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

