
# Trace Visualization



??? "Class: `TraceVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/trace/v1.py">inference.core.workflows.core_steps.visualizations.trace.v1.TraceVisualizationBlockV1</a>
    



Draw trajectory paths for tracked objects, visualizing their movement history by connecting recent positions with colored lines to show object movement patterns, paths, and tracking behavior over time.

## How This Block Works

This block takes an image and tracked predictions (with tracker IDs) and draws trajectory paths showing the recent movement history of each tracked object. The block:

1. Takes an image and tracked predictions as input (predictions must include tracker_id data from a tracking block)
2. Extracts tracking IDs and position history for each tracked object
3. Determines the reference point for drawing traces based on the selected position anchor (center, corners, edges, or center of mass)
4. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Draws trajectory lines connecting the recent positions (up to trace_length positions) for each tracked object using Supervision's TraceAnnotator
6. Connects historical positions sequentially, creating path traces that show object movement direction and patterns
7. Returns an annotated image with trajectory paths overlaid on the original image

The block visualizes object tracking by drawing the path that each tracked object has taken over recent frames. Each tracked object gets a unique trace line (colored by track ID, class, or index) that connects its recent positions, creating a visual trail that shows movement direction, speed, and trajectory patterns. The trace_length parameter controls how many historical positions are included in each trace (longer traces show more movement history, shorter traces show recent movement only). This visualization requires predictions with tracker IDs from tracking blocks (like Byte Tracker), as it needs the tracking information to connect positions across frames. The traces help visualize object movement, identify tracking patterns, and understand object behavior over time.

## Common Use Cases

- **Object Trajectory Visualization**: Visualize movement paths and trajectories of tracked objects to understand object behavior, movement patterns, or navigation routes for applications like vehicle tracking, pedestrian flow analysis, or object movement monitoring
- **Tracking Performance Validation**: Validate tracking performance by visualizing object paths to ensure tracking consistency, identify tracking errors or ID switches, or verify that objects maintain consistent trajectories
- **Movement Pattern Analysis**: Analyze movement patterns, speeds, or direction changes by visualizing trajectory traces to understand object behavior, detect anomalies, or identify movement trends in surveillance, security, or traffic monitoring workflows
- **Path Deviation Detection**: Visualize object paths to detect deviations from expected routes, identify unusual movement patterns, or monitor object trajectories for safety, security, or compliance workflows
- **Real-Time Tracking Monitoring**: Display trajectory traces in real-time monitoring interfaces, dashboards, or live video feeds to visualize object movement and tracking behavior as it happens
- **Video Analysis and Post-Processing**: Create trajectory visualizations for video analysis, post-processing workflows, or forensic analysis where understanding object movement paths and patterns is critical

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Tracking blocks** (e.g., Byte Tracker) to receive tracked predictions with tracker IDs that are required for trace visualization
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Dot Visualization) to combine trajectory traces with additional annotations for comprehensive tracking visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with trajectory traces for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with trajectory traces to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with trajectory traces as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with trajectory traces for live monitoring, tracking visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/trace_visualization@v1`to add the block as
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
| `position` | `str` | Anchor position for drawing trajectory traces relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object). The trace path is drawn connecting positions at this anchor point across recent frames.. | ✅ |
| `trace_length` | `int` | Maximum number of historical tracked object positions to include in each trajectory trace. Controls how long the movement trail appears. Higher values create longer traces showing more movement history, while lower values create shorter traces showing only recent movement. Must be at least 1. Typical values range from 10 to 50 frames depending on the desired trail length and frame rate.. | ✅ |
| `thickness` | `int` | Thickness of the trajectory trace lines in pixels. Controls how thick the path lines appear. Higher values create thicker, more visible traces, while lower values create thinner, more subtle traces. Must be at least 1. Typical values range from 1 to 5 pixels.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Trace Visualization` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Velocity`](velocity.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Buffer`](buffer.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`EasyOCR`](easy_ocr.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Depth Estimation`](depth_estimation.md), [`CogVLM`](cog_vlm.md), [`Detections Filter`](detections_filter.md), [`LMM`](lmm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`EasyOCR`](easy_ocr.md), [`SAM 3`](sam3.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`CogVLM`](cog_vlm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter Visualization`](line_counter_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Trace Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Model predictions to visualize..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `position` (*[`string`](../kinds/string.md)*): Anchor position for drawing trajectory traces relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object). The trace path is drawn connecting positions at this anchor point across recent frames..
        - `trace_length` (*[`integer`](../kinds/integer.md)*): Maximum number of historical tracked object positions to include in each trajectory trace. Controls how long the movement trail appears. Higher values create longer traces showing more movement history, while lower values create shorter traces showing only recent movement. Must be at least 1. Typical values range from 10 to 50 frames depending on the desired trail length and frame rate..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the trajectory trace lines in pixels. Controls how thick the path lines appear. Higher values create thicker, more visible traces, while lower values create thinner, more subtle traces. Must be at least 1. Typical values range from 1 to 5 pixels..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Trace Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/trace_visualization@v1",
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
	    "trace_length": 30,
	    "thickness": 1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

