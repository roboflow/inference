
# Model Comparison Visualization



??? "Class: `ModelComparisonVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/model_comparison/v1.py">inference.core.workflows.core_steps.visualizations.model_comparison.v1.ModelComparisonVisualizationBlockV1</a>
    



Compare predictions from two different models by color-coding areas where only one model detected objects, highlighting model differences while leaving overlapping predictions unchanged to visualize model agreement and disagreement.

## How This Block Works

This block takes an image and predictions from two models (Model A and Model B) and creates a visual comparison overlay that highlights differences between the models. The block:

1. Takes an image and two sets of predictions (predictions_a and predictions_b) as input
2. Creates masks for areas predicted by each model (using bounding boxes or segmentation masks if available)
3. Identifies four distinct regions:
   - Areas predicted only by Model A (colored with color_a, default green)
   - Areas predicted only by Model B (colored with color_b, default red)
   - Areas predicted by both models (left unchanged, allowing the original image to show through)
   - Areas predicted by neither model (colored with background_color, default black)
4. Applies colored overlays to the identified regions using the specified opacity
5. Returns an annotated image where model differences are visually distinguished with color coding

The block creates a side-by-side comparison visualization that makes it easy to see where models agree (unchanged areas) and where they disagree (color-coded areas). Areas where both models made predictions are left unchanged, allowing the original image to "shine through" and clearly showing model consensus. This visualization helps identify model strengths, weaknesses, and differences in detection behavior. The block works with object detection predictions (using bounding boxes) or instance segmentation predictions (using masks), making it versatile for comparing different model types.

## Common Use Cases

- **Model Evaluation and Comparison**: Compare two models' detection performance side-by-side to identify where models agree, disagree, or have different detection behaviors for model evaluation, benchmarking, or selection workflows
- **Model Development and Debugging**: Visualize differences between model versions, architectures, or configurations to understand how changes affect detection behavior, identify improvement opportunities, or debug model performance issues
- **Ensemble Model Analysis**: Compare predictions from different models in ensemble workflows to understand model agreement patterns, identify complementary strengths, or analyze consensus areas for ensemble decision-making
- **Training Data Analysis**: Compare model predictions to ground truth annotations or between training runs to identify patterns in detection differences, validate training improvements, or analyze model behavior across datasets
- **A/B Testing and Model Selection**: Visually compare candidate models to evaluate relative performance, identify detection differences, or make informed model selection decisions for deployment
- **Quality Assurance and Validation**: Validate model consistency, compare model performance on edge cases, or identify systematic differences between models for quality assurance, validation, or compliance workflows

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions_a and predictions_b from different models for comparison
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save comparison visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send comparison visualizations to external systems, APIs, or web applications for display in dashboards, model monitoring tools, or evaluation interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send comparison visualizations as visual evidence in alerts or reports for model performance monitoring
- **Video output blocks** to create annotated video streams or recordings with model comparison visualizations for live model evaluation, performance monitoring, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/model_comparison_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `color_a` | `str` | Color used to highlight areas predicted only by Model A (that Model B did not predict). Can be specified as a color name (e.g., 'GREEN', 'BLUE'), hex color code (e.g., '#00FF00', '#FFFFFF'), or RGB format (e.g., 'rgb(0, 255, 0)'). Default is GREEN to indicate Model A's unique predictions.. | ✅ |
| `color_b` | `str` | Color used to highlight areas predicted only by Model B (that Model A did not predict). Can be specified as a color name (e.g., 'RED', 'BLUE'), hex color code (e.g., '#FF0000', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 0, 0)'). Default is RED to indicate Model B's unique predictions.. | ✅ |
| `background_color` | `str` | Color used for areas predicted by neither model. Can be specified as a color name (e.g., 'BLACK', 'GRAY'), hex color code (e.g., '#000000', '#808080'), or RGB format (e.g., 'rgb(0, 0, 0)'). Default is BLACK to indicate areas where both models missed detections.. | ✅ |
| `opacity` | `float` | Opacity of the comparison overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the color-coded overlays appear over the original image. Lower values create more transparent overlays where original image details remain more visible, while higher values create more opaque overlays with stronger color emphasis. Typical values range from 0.5 to 0.8 for balanced visibility.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Model Comparison Visualization` in version `v1`.

    - inputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`JSON Parser`](json_parser.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Identify Outliers`](identify_outliers.md), [`Icon Visualization`](icon_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Image Blur`](image_blur.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT`](sift.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Model Comparison Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions_a` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Predictions from Model A (the first model being compared). Can be object detection, instance segmentation, or keypoint detection predictions. Areas predicted only by Model A (and not by Model B) will be colored with color_a. Works with bounding boxes or masks depending on prediction type..
        - `color_a` (*[`string`](../kinds/string.md)*): Color used to highlight areas predicted only by Model A (that Model B did not predict). Can be specified as a color name (e.g., 'GREEN', 'BLUE'), hex color code (e.g., '#00FF00', '#FFFFFF'), or RGB format (e.g., 'rgb(0, 255, 0)'). Default is GREEN to indicate Model A's unique predictions..
        - `predictions_b` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Predictions from Model B (the second model being compared). Can be object detection, instance segmentation, or keypoint detection predictions. Areas predicted only by Model B (and not by Model A) will be colored with color_b. Works with bounding boxes or masks depending on prediction type..
        - `color_b` (*[`string`](../kinds/string.md)*): Color used to highlight areas predicted only by Model B (that Model A did not predict). Can be specified as a color name (e.g., 'RED', 'BLUE'), hex color code (e.g., '#FF0000', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 0, 0)'). Default is RED to indicate Model B's unique predictions..
        - `background_color` (*[`string`](../kinds/string.md)*): Color used for areas predicted by neither model. Can be specified as a color name (e.g., 'BLACK', 'GRAY'), hex color code (e.g., '#000000', '#808080'), or RGB format (e.g., 'rgb(0, 0, 0)'). Default is BLACK to indicate areas where both models missed detections..
        - `opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the comparison overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the color-coded overlays appear over the original image. Lower values create more transparent overlays where original image details remain more visible, while higher values create more opaque overlays with stronger color emphasis. Typical values range from 0.5 to 0.8 for balanced visibility..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Model Comparison Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/model_comparison_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions_a": "$steps.object_detection_model.predictions",
	    "color_a": "GREEN",
	    "predictions_b": "$steps.object_detection_model.predictions",
	    "color_b": "RED",
	    "background_color": "BLACK",
	    "opacity": 0.7
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

