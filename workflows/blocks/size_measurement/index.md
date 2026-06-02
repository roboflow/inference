
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

    - inputs: [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SAM 3`](sam3.md), [`Byte Tracker`](byte_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`CogVLM`](cog_vlm.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenRouter`](open_router.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Gemma API`](google_gemma_api.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Path Deviation`](path_deviation.md), [`Google Gemini`](google_gemini.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`EasyOCR`](easy_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`LMM`](lmm.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`OCR Model`](ocr_model.md), [`Seg Preview`](seg_preview.md), [`Moondream2`](moondream2.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Detections Merge`](detections_merge.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemma`](google_gemma.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Florence-2 Model`](florence2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Seg Preview`](seg_preview.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Path Deviation`](path_deviation.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Time in Zone`](timein_zone.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Size Measurement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `object_predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Model predictions to measure the dimensions of..
        - `reference_predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`list_of_values`](../kinds/list_of_values.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Reference object used to calculate the dimensions of the specified objects. If multiple objects are provided, the highest confidence prediction will be used..
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

