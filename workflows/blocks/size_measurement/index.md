
# Size Measurement



??? "Class: `SizeMeasurementBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/size_measurement/v1.py">inference.core.workflows.core_steps.classical_cv.size_measurement.v1.SizeMeasurementBlockV1</a>
    



The [**Size Measurement Block**](https://www.

## How This Block Works

youtube.com/watch?v=FQY7TSHfZeI) calculates the dimensions of objects relative to a reference object. It uses one model to detect the reference object and another to detect the objects to measure. The block outputs the dimensions of the objects in terms of the reference object.

- **Reference Object**: This is the known object used as a baseline for measurements. Its dimensions are known and used to scale the measurements of other objects.
- **Object to Measure**: This is the object whose dimensions are being calculated. The block measures these dimensions relative to the reference object.

### Block Usage

To use the Size Measurement Block, follow these steps:

1. **Select Models**: Choose a model to detect the reference object and another model to detect the objects you want to measure.
2. **Configure Inputs**: Provide the predictions from both models as inputs to the block.
3. **Set Reference Dimensions**: Specify the known dimensions of the reference object in the format 'width,height' or as a tuple (width, height).
4. **Run the Block**: Execute the block to calculate the dimensions of the detected objects relative to the reference object.

### Example

Imagine you have a scene with a calibration card and several packages. The calibration card has known dimensions of 5.0 inches by 3.0 inches. You want to measure the dimensions of packages in the scene.

- **Reference Object**: Calibration card with dimensions 5.0 inches (width) by 3.0 inches (height).
- **Objects to Measure**: Packages detected in the scene.

The block will use the known dimensions of the calibration card to calculate the dimensions of each package. For example, if a package is detected with a width of 100 pixels and a height of 60 pixels, and the calibration card is detected with a width of 50 pixels and a height of 30 pixels, the block will calculate the package's dimensions as:

- **Width**: (100 pixels / 50 pixels) * 5.0 inches = 10.0 inches
- **Height**: (60 pixels / 30 pixels) * 3.0 inches = 6.0 inches

This allows you to obtain the real-world dimensions of the packages based on the reference object's known size.

[Watch the video tutorial](https://www.youtube.com/watch?v=FQY7TSHfZeI)


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/size_measurement@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `reference_predictions` | `List[Any]` | Reference object used to calculate the dimensions of the specified objects. If multiple objects are provided, the highest confidence prediction will be used.. | ✅ |
| `reference_dimensions` | `Union[List[float], Tuple[float, float], str]` | Dimensions of the reference object in desired units, (e.g. inches). Will be used to convert the pixel dimensions of the other objects to real-world units.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Size Measurement` in version `v1`.

    - inputs: [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Detections Combine`](detections_combine.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Byte Tracker`](byte_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Local File Sink`](local_file_sink.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Detections Filter`](detections_filter.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Dimension Collapse`](dimension_collapse.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Halo Visualization`](halo_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Email Notification`](email_notification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Cache Set`](cache_set.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Path Deviation`](path_deviation.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Size Measurement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `object_predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions to measure the dimensions of..
        - `reference_predictions` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Reference object used to calculate the dimensions of the specified objects. If multiple objects are provided, the highest confidence prediction will be used..
        - `reference_dimensions` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): Dimensions of the reference object in desired units, (e.g. inches). Will be used to convert the pixel dimensions of the other objects to real-world units..

    - output
    
        - `dimensions` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Size Measurement` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/size_measurement@v1",
	    "object_predictions": "$segmentation.object_predictions",
	    "reference_predictions": "$segmentation.reference_predictions",
	    "reference_dimensions": [
	        4.5,
	        3.0
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

