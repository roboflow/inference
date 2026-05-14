
# EasyOCR



??? "Class: `EasyOCRBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/easy_ocr/v1.py">inference.core.workflows.core_steps.models.foundation.easy_ocr.v1.EasyOCRBlockV1</a>
    



 Retrieve the characters in an image using EasyOCR Optical Character Recognition (OCR).

This block returns the text within an image.

You may want to use this block in combination with a detections-based block (i.e.
ObjectDetectionBlock). An object detection model could isolate specific regions from an
image (i.e. a shipping container ID in a logistics use case) for further processing.
You can then use a DynamicCropBlock to crop the region of interest before running OCR.

Using a detections model then cropping detections allows you to isolate your analysis
on particular regions of an image.

Note that EasyOCR has limitations running within containers on Apple Silicon.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/easy_ocr@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `language` | `str` | Language model to use for OCR. | ❌ |
| `quantize` | `bool` | Quantized models are smaller and faster, but may be less accurate and won't work correctly on all hardware.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `EasyOCR` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SIFT`](sift.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Dot Visualization`](dot_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`OpenAI`](open_ai.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemma API`](google_gemma_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`EasyOCR` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..

    - output
    
        - `result` ([`string`](../kinds/string.md)): String value.
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `prediction_type` ([`prediction_type`](../kinds/prediction_type.md)): String value with type of prediction.



??? tip "Example JSON definition of step `EasyOCR` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/easy_ocr@v1",
	    "images": "$inputs.image",
	    "language": "<block_does_not_provide_example>",
	    "quantize": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

