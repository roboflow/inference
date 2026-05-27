
# Triangle Visualization



??? "Class: `TriangleVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/triangle/v1.py">inference.core.workflows.core_steps.visualizations.triangle.v1.TriangleVisualizationBlockV1</a>
    



Draw triangular markers on an image to mark specific points on detected objects, with customizable position, size, color, and outline styling, providing directional indicators and geometric markers for visual annotations.

## How This Block Works

This block takes an image and detection predictions and draws triangular markers at specified anchor positions on each detected object. The block:

1. Takes an image and predictions as input
2. Determines the triangle position for each detection based on the selected anchor point (center, corners, edges, or center of mass)
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws triangular markers with the specified base width and height dimensions, with optional outline thickness using Supervision's TriangleAnnotator
5. Returns an annotated image with triangular markers overlaid on the original image

The block supports various position options including the center of the bounding box, any of the four corners, edge midpoints, or the center of mass (useful for objects with irregular shapes). Triangles can be customized with different sizes (base width and height), optional outlines for better visibility, and various color palettes. Triangular markers provide a distinctive geometric shape that can serve as directional indicators (pointing in a specific direction) or geometric markers, offering an alternative to circular dots or square markers. This provides a clean visualization style that marks detection locations with directional or geometric emphasis, making it ideal for applications requiring directional indicators, geometric markers, or distinctive point markers.

## Common Use Cases

- **Directional Object Marking**: Mark detected objects with triangular markers that can indicate direction or orientation, useful for tracking applications, motion analysis, or directional workflows where the triangle's pointed shape provides directional information
- **Geometric Marker Visualization**: Use triangular markers as distinctive geometric shapes to mark detection locations, providing visual variety compared to circular dots or rectangular bounding boxes for design purposes or geometric emphasis
- **Minimal Object Marking**: Mark detected objects with small triangular markers instead of bounding boxes for cleaner, less cluttered visualizations when working with dense scenes or many detections, with the triangular shape providing visual distinction
- **Tracking Visualization**: Use triangular markers to visualize object trajectories or tracking IDs over time, creating a cleaner alternative to bounding boxes for tracking workflows, with triangles potentially indicating movement direction
- **Point of Interest Highlighting**: Mark specific anchor points (corners, center, center of mass) on detected objects with triangular markers for applications like object tracking, spatial analysis, or geometric annotation workflows
- **Design and Aesthetic Applications**: Create triangular markers for design purposes, user interfaces, dashboards, or artistic visualizations where geometric shapes provide distinctive visual style or aesthetic appeal

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Trace Visualization) to combine triangular markers with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with triangular markers for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with triangular markers to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with triangular markers as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with triangular markers for live monitoring, tracking visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/triangle_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `color_palette` | `str` | Select a color palette for the visualised elements.. | ✅ |
| `palette_size` | `int` | Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes.. | ✅ |
| `custom_colors` | `List[str]` | Define a list of custom colors for bounding boxes in HEX format.. | ✅ |
| `color_axis` | `str` | Choose how bounding box colors are assigned.. | ✅ |
| `position` | `str` | Anchor position for placing the triangle marker relative to the detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object, useful for irregular shapes).. | ✅ |
| `base` | `int` | Base width of the triangle in pixels. Controls the horizontal width of the triangular marker at its base. Larger values create wider triangles, while smaller values create narrower triangles. Works together with height to control the overall triangle size and shape.. | ✅ |
| `height` | `int` | Height of the triangle in pixels. Controls the vertical height of the triangular marker from base to tip. Larger values create taller triangles, while smaller values create shorter triangles. Works together with base to control the overall triangle size and shape.. | ✅ |
| `outline_thickness` | `int` | Thickness of the triangle outline in pixels. A value of 0 creates a filled triangle with no outline. Higher values create thicker outlines around the triangle border, improving visibility and contrast. Useful for making triangular markers more visible against complex backgrounds.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Triangle Visualization` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`GLM-OCR`](glmocr.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`JSON Parser`](json_parser.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Detections Merge`](detections_merge.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`SAM 3`](sam3.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Offset`](detection_offset.md), [`Identify Changes`](identify_changes.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Slack Notification`](slack_notification.md), [`OpenAI`](open_ai.md), [`Mask Area Measurement`](mask_area_measurement.md), [`OCR Model`](ocr_model.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Velocity`](velocity.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Object Detection Model`](object_detection_model.md), [`Image Blur`](image_blur.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Dominant Color`](dominant_color.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Buffer`](buffer.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Triangle Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Model predictions to visualize..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `position` (*[`string`](../kinds/string.md)*): Anchor position for placing the triangle marker relative to the detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object, useful for irregular shapes)..
        - `base` (*[`integer`](../kinds/integer.md)*): Base width of the triangle in pixels. Controls the horizontal width of the triangular marker at its base. Larger values create wider triangles, while smaller values create narrower triangles. Works together with height to control the overall triangle size and shape..
        - `height` (*[`integer`](../kinds/integer.md)*): Height of the triangle in pixels. Controls the vertical height of the triangular marker from base to tip. Larger values create taller triangles, while smaller values create shorter triangles. Works together with base to control the overall triangle size and shape..
        - `outline_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the triangle outline in pixels. A value of 0 creates a filled triangle with no outline. Higher values create thicker outlines around the triangle border, improving visibility and contrast. Useful for making triangular markers more visible against complex backgrounds..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Triangle Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/triangle_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.object_detection_model.predictions",
	    "color_palette": "DEFAULT",
	    "palette_size": 10,
	    "custom_colors": [
	        "#FF0000",
	        "#00FF00",
	        "#0000FF"
	    ],
	    "color_axis": "CLASS",
	    "position": "CENTER",
	    "base": 10,
	    "height": 10,
	    "outline_thickness": 2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

