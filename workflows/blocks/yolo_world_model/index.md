
# YOLO-World Model



??? "Class: `YoloWorldModelBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/yolo_world/v1.py">inference.core.workflows.core_steps.models.foundation.yolo_world.v1.YoloWorldModelBlockV1</a>
    



Run YOLO-World, a zero-shot object detection model, on an image.

YOLO-World accepts one or more text classes you want to identify in an image. The model 
returns the location of objects that meet the specified class, if YOLO-World is able to 
identify objects of that class.

We recommend experimenting with YOLO-World to evaluate the model on your use case 
before using this block in production. For example on how to effectively prompt 
YOLO-World, refer to the [Roboflow YOLO-World prompting 
guide](https://blog.roboflow.com/yolo-world-prompting-tips/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/yolo_world_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `class_names` | `List[str]` | One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects.. | ✅ |
| `version` | `str` | Variant of YoloWorld model. | ✅ |
| `confidence` | `float` | Confidence threshold for detections. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `YOLO-World Model` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Dimension Collapse`](dimension_collapse.md), [`Event Writer`](event_writer.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Local File Sink`](local_file_sink.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`EasyOCR`](easy_ocr.md), [`LMM`](lmm.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemma`](google_gemma.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Google Gemini`](google_gemini.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Size Measurement`](size_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Image Slicer`](image_slicer.md)
    - outputs: [`Event Writer`](event_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Camera Focus`](camera_focus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Path Deviation`](path_deviation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Overlap Filter`](overlap_filter.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Label Visualization`](label_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Transformation`](detections_transformation.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Line Counter`](line_counter.md), [`Detection Offset`](detection_offset.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Detections Combine`](detections_combine.md), [`Dot Visualization`](dot_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Path Deviation`](path_deviation.md), [`Detection Event Log`](detection_event_log.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`YOLO-World Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `class_names` (*[`list_of_values`](../kinds/list_of_values.md)*): One or more classes that you want YOLO-World to detect. The model accepts any string as an input, though does best with short descriptions of common objects..
        - `version` (*[`string`](../kinds/string.md)*): Variant of YoloWorld model.
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for detections.

    - output
    
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `YOLO-World Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/yolo_world_model@v1",
	    "images": "$inputs.image",
	    "class_names": [
	        "person",
	        "car",
	        "license plate"
	    ],
	    "version": "v2-s",
	    "confidence": 0.005
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

