
# Blur Visualization



??? "Class: `BlurVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/blur/v1.py">inference.core.workflows.core_steps.visualizations.blur.v1.BlurVisualizationBlockV1</a>
    



Apply blur effects to detected objects in an image, obscuring their details while preserving the background, useful for privacy protection, content filtering, or visual emphasis.

## How This Block Works

This block takes an image and detection predictions and applies a blur effect to the detected objects, leaving the background unchanged. The block:

1. Takes an image and predictions as input
2. Identifies detected regions from bounding boxes or segmentation masks
3. Applies a blur effect (using average pooling) to the detected object regions
4. Preserves the background and areas outside detected objects unchanged
5. Returns an annotated image where detected objects are blurred, while the rest of the image remains sharp

The block works with both object detection predictions (using bounding boxes) and instance segmentation predictions (using masks). When masks are available, it blurs the exact shape of detected objects; otherwise, it blurs rectangular bounding box regions. The blur intensity is controlled by the kernel size parameter, where larger kernel sizes create stronger blur effects. This creates a visual effect that obscures or anonymizes detected objects while maintaining context from the surrounding image, making it ideal for privacy protection, content filtering, or focusing attention on the background.

## Common Use Cases

- **Privacy Protection and Anonymization**: Blur faces, people, license plates, or other sensitive information in images or videos to protect privacy, comply with data protection regulations, or anonymize content before sharing or publishing
- **Content Filtering and Moderation**: Obscure inappropriate or sensitive content in images or videos for content moderation workflows, safe content previews, or user-generated content filtering
- **Visual Emphasis and Focus**: Blur detected objects to draw attention to other parts of the image, create visual contrast between blurred foreground objects and sharp backgrounds, or emphasize specific elements in composition
- **Product Photography and E-commerce**: Blur detected distracting elements or secondary products in images to keep the main subject sharp and prominent for product photography, catalog creation, or e-commerce image preparation
- **Security and Surveillance**: Anonymize people, vehicles, or other identifiable elements in security footage or surveillance images while preserving scene context for analysis, reporting, or public sharing
- **Documentation and Reporting**: Create anonymized or censored versions of images for reports, documentation, or case studies where sensitive information needs to be obscured but overall context should remain visible

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to add additional annotations on top of blurred objects for comprehensive visualization or to indicate what was blurred
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save blurred images for documentation, reporting, or archiving privacy-protected content
- **Webhook blocks** to send blurred images to external systems, APIs, or web applications for content moderation, privacy-compliant sharing, or anonymized analysis
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send blurred images as privacy-protected visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with blurred objects for live monitoring, privacy-compliant video processing, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/blur_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `kernel_size` | `int` | Size of the blur kernel used for average pooling. Larger values create stronger blur effects, making objects more obscured. Smaller values create subtle blur effects. Typical values range from 5 (light blur) to 51 (strong blur). Must be an odd number for optimal blurring performance.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Blur Visualization` in version `v1`.

    - inputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Slack Notification`](slack_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Offset`](detection_offset.md), [`Velocity`](velocity.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Overlap Filter`](overlap_filter.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Merge`](detections_merge.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Morphological Transformation`](morphological_transformation.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Relative Static Crop`](relative_static_crop.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Moondream2`](moondream2.md), [`Dynamic Zone`](dynamic_zone.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Camera Focus`](camera_focus.md), [`Grid Visualization`](grid_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Gaze Detection`](gaze_detection.md), [`Camera Calibration`](camera_calibration.md), [`SIFT`](sift.md), [`Email Notification`](email_notification.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Preprocessing`](image_preprocessing.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Seg Preview`](seg_preview.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`JSON Parser`](json_parser.md), [`Detections Stitch`](detections_stitch.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`Motion Detection`](motion_detection.md), [`Crop Visualization`](crop_visualization.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Perspective Correction`](perspective_correction.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Image Slicer`](image_slicer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Camera Focus`](camera_focus.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Trace Visualization`](trace_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`QR Code Detection`](qr_code_detection.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Halo Visualization`](halo_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Blur Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Model predictions to visualize..
        - `kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the blur kernel used for average pooling. Larger values create stronger blur effects, making objects more obscured. Smaller values create subtle blur effects. Typical values range from 5 (light blur) to 51 (strong blur). Must be an odd number for optimal blurring performance..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Blur Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/blur_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.object_detection_model.predictions",
	    "kernel_size": 15
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

