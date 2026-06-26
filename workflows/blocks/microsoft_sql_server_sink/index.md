
# Microsoft SQL Server Sink



??? "Class: `MicrosoftSQLServerSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/microsoft_sql_server/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.microsoft_sql_server.v1.MicrosoftSQLServerSinkBlockV1</a>
    



The **Microsoft SQL Server Sink** block enables users to send data from a Roboflow workflow directly to a Microsoft SQL Server 
database. This block allows seamless integration of inference results, metadata, and processed data into structured SQL 
databases for further analysis, reporting, or automation.

### Database Connection Setup

The block supports two authentication methods:

1. **Windows Authentication (Default)**: Uses the current Windows credentials
2. **SQL Server Authentication**: Uses username and password

Required connection parameters:
* **Host**: The IP address or hostname of the Microsoft SQL Server instance
* **Port**: The port number for SQL Server (default: 1433)
* **Database**: The target database where data will be inserted
* **Table Name**: The name of the table where the data will be inserted

Optional authentication parameters (for SQL Server Authentication):
* **Username**: The SQL Server username for authentication
* **Password**: The password associated with the username

If username and password are not provided, the block will use Windows Authentication (trusted connection).

### Data Input Format

The block expects data in a dictionary format or list of dictionaries that map to the target table columns:

```python
# Single row
{
    "timestamp": "2025-02-12T10:30:00Z",
    "part_detected": "Defective Part",
    "confidence": 0.92,
    "camera_id": "CAM_001"
}

# Multiple rows
[
    {
        "timestamp": "2025-02-12T10:30:00Z",
        "part_detected": "Defective Part",
        "confidence": 0.92,
        "camera_id": "CAM_001"
    },
    {
        "timestamp": "2025-02-12T10:31:00Z",
        "part_detected": "Good Part",
        "confidence": 0.95,
        "camera_id": "CAM_002"
    }
]
```

### Important Notes

* The specified table must already exist in the database
* The authenticated user must have INSERT permissions
* Column names in the data must match the table schema
* When using Windows Authentication, ensure the service account has proper permissions
* The pyodbc package must be installed


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/microsoft_sql_server_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `host` | `str` | SQL Server host address. | ✅ |
| `port` | `int` | SQL Server port. | ✅ |
| `database` | `str` | Target database name. | ✅ |
| `username` | `str` | SQL Server username. | ✅ |
| `password` | `str` | SQL Server password. | ✅ |
| `table_name` | `str` | Target table name. | ✅ |
| `data` | `Union[Dict[Any, Any], List[Dict[Any, Any]]]` | Data to insert into the database. Can be a single dictionary or list of dictionaries.. | ✅ |
| `fire_and_forget` | `bool` | Run in asynchronous mode for faster processing. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Microsoft SQL Server Sink` in version `v1`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma API`](google_gemma_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`SIFT Comparison`](sift_comparison.md), [`S3 Sink`](s3_sink.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Google Vision OCR`](google_vision_ocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma`](google_gemma.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Identify Changes`](identify_changes.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Motion Detection`](motion_detection.md), [`Qwen3.5`](qwen3.5.md), [`CogVLM`](cog_vlm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`PLC Reader`](plc_reader.md), [`Current Time`](current_time.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Slack Notification`](slack_notification.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md)
    - outputs: [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Time in Zone`](timein_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Label Visualization`](label_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Microsoft SQL Server Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `host` (*[`string`](../kinds/string.md)*): SQL Server host address.
        - `port` (*[`string`](../kinds/string.md)*): SQL Server port.
        - `database` (*[`string`](../kinds/string.md)*): Target database name.
        - `username` (*[`string`](../kinds/string.md)*): SQL Server username.
        - `password` (*[`secret`](../kinds/secret.md)*): SQL Server password.
        - `table_name` (*[`string`](../kinds/string.md)*): Target table name.
        - `data` (*[`dictionary`](../kinds/dictionary.md)*): Data to insert into the database. Can be a single dictionary or list of dictionaries..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Run in asynchronous mode for faster processing.

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Microsoft SQL Server Sink` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/microsoft_sql_server_sink@v1",
	    "host": "localhost",
	    "port": 1433,
	    "database": "production_db",
	    "username": "db_user",
	    "password": "$inputs.sql_password",
	    "table_name": "detections",
	    "data": {
	        "object_detected": "Defective Part",
	        "timestamp": "2025-02-12T10:30:00Z"
	    },
	    "fire_and_forget": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

