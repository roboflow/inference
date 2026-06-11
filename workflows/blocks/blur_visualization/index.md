
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

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Event Writer`](event_writer.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Local File Sink`](local_file_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Path Deviation`](path_deviation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`JSON Parser`](json_parser.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Detections Transformation`](detections_transformation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Stitch`](detections_stitch.md), [`Gaze Detection`](gaze_detection.md), [`Byte Tracker`](byte_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Detection Offset`](detection_offset.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Merge`](detections_merge.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`SORT Tracker`](sort_tracker.md), [`Image Stack`](image_stack.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Detections Combine`](detections_combine.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`Detection Event Log`](detection_event_log.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Blur Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Model predictions to visualize..
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

