
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

    - inputs: [`Google Vision OCR`](google_vision_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`SmolVLM2`](smol_vlm2.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PLC Reader`](plc_reader.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Current Time`](current_time.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OCR Model`](ocr_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Identify Changes`](identify_changes.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`LMM`](lmm.md), [`Detections Consensus`](detections_consensus.md), [`Event Writer`](event_writer.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
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

