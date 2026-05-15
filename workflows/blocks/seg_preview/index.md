
# Seg Preview



??? "Class: `SegPreviewBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/seg_preview/v1.py">inference.core.workflows.core_steps.models.foundation.seg_preview.v1.SegPreviewBlockV1</a>
    


Seg Preview

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/seg-preview@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `class_names` | `Union[List[str], str]` | List of classes to recognise. | ✅ |
| `threshold` | `float` | Threshold for predicted mask scores. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Seg Preview` in version `v1`.

    - inputs: [`Perspective Correction`](perspective_correction.md), [`S3 Sink`](s3_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen-VL`](qwen_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Clip Comparison`](clip_comparison.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Local File Sink`](local_file_sink.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OCR Model`](ocr_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Cosine Similarity`](cosine_similarity.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Color Visualization`](color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Text Display`](text_display.md), [`Dimension Collapse`](dimension_collapse.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Perspective Correction`](perspective_correction.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Path Deviation`](path_deviation.md), [`Line Counter`](line_counter.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Detection Offset`](detection_offset.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Polygon Visualization`](polygon_visualization.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Zone`](dynamic_zone.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Corner Visualization`](corner_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Seg Preview` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `class_names` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): List of classes to recognise.
        - `threshold` (*[`float`](../kinds/float.md)*): Threshold for predicted mask scores.

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Seg Preview` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/seg-preview@v1",
	    "images": "$inputs.image",
	    "class_names": [
	        "car",
	        "person"
	    ],
	    "threshold": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

