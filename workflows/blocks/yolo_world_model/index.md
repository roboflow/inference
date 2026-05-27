
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

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Qwen-VL`](qwen_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Identify Outliers`](identify_outliers.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`OCR Model`](ocr_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Line Counter`](line_counter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Overlap Analysis`](overlap_analysis.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections Combine`](detections_combine.md), [`Detections Merge`](detections_merge.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Circle Visualization`](circle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Byte Tracker`](byte_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Path Deviation`](path_deviation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Florence-2 Model`](florence2_model.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Byte Tracker`](byte_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Overlap Filter`](overlap_filter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Triangle Visualization`](triangle_visualization.md)

    
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

