
# Label Visualization



??? "Class: `LabelVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/label/v1.py">inference.core.workflows.core_steps.visualizations.label.v1.LabelVisualizationBlockV1</a>
    



Draw text labels on detected objects with customizable content, position, styling, and background colors to display information like class names, confidence scores, tracking IDs, or other detection metadata.

## How This Block Works

This block takes an image and detection predictions and draws text labels on each detected object. The block:

1. Takes an image and predictions as input
2. Extracts label text for each detection based on the selected text option (class name, confidence, tracker ID, dimensions, area, time in zone, or index)
3. Determines label position based on the selected anchor point (center, corners, edges, or center of mass)
4. Applies background color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Renders text labels with customizable text color, scale, thickness, padding, and border radius using Supervision's LabelAnnotator
6. Returns an annotated image with text labels overlaid on the original image

The block supports various text content options including class names, confidence scores, combination of class and confidence, tracker IDs (for tracked objects), time in zone (for zone analysis), object dimensions (center coordinates and width/height), area, or detection index. Labels are rendered with colored backgrounds that match the object's assigned color from the palette, and text styling (color, size, thickness) can be customized for optimal visibility. The labels can be positioned at any anchor point relative to each detection, allowing flexible placement for different visualization needs.

## Common Use Cases

- **Information Display on Detections**: Add informative text labels showing class names, confidence scores, or other metadata directly on detected objects for quick identification and validation
- **Model Performance Visualization**: Display confidence scores or class predictions on detected objects to visualize model certainty, identify low-confidence detections, and validate model performance
- **Object Tracking Visualization**: Show tracker IDs on tracked objects to visualize object tracking across frames, monitor persistent object identities, or debug tracking algorithms
- **Zone Analysis and Monitoring**: Display "Time In Zone" labels on objects to visualize how long objects have been in specific zones for occupancy monitoring, dwell time analysis, or compliance tracking
- **Spatial Information Display**: Show object dimensions (center coordinates, width, height) or area measurements directly on detections for spatial analysis, measurement workflows, or quality control
- **Professional Presentation and Reporting**: Create clean, informative visualizations with labeled detections for reports, dashboards, or presentations that combine visual results with textual information

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Dot Visualization) to combine text labels with geometric annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with labels for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with labels to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with labels as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with labels for live monitoring, tracking visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/label_visualization@v1`to add the block as
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
| `text` | `str` | Content to display in text labels. Options: 'Class' (class name), 'Confidence' (confidence score), 'Class and Confidence' (both), 'Tracker Id' (tracking ID for tracked objects), 'Time In Zone' (time spent in zone), 'Dimensions' (center coordinates and width x height), 'Area' (bounding box area in pixels), 'Area (mask)' (mask area in pixels from Mask Area Measurement block), 'Area (converted)' (mask area in converted units from Mask Area Measurement block), or 'Index' (detection index).. | ✅ |
| `text_position` | `str` | Anchor position for placing labels relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).. | ✅ |
| `text_color` | `str` | Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').. | ✅ |
| `text_scale` | `float` | Scale factor for text size. Higher values create larger text. Default is 1.0.. | ✅ |
| `text_thickness` | `int` | Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility.. | ✅ |
| `text_padding` | `int` | Padding around the text in pixels. Controls the spacing between the text and the label background border.. | ✅ |
| `border_radius` | `int` | Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Label Visualization` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Detections Combine`](detections_combine.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Webhook Sink`](webhook_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`Detections Transformation`](detections_transformation.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Pixel Color Count`](pixel_color_count.md), [`Dimension Collapse`](dimension_collapse.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`OCR Model`](ocr_model.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Label Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Model predictions to visualize..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `text` (*[`string`](../kinds/string.md)*): Content to display in text labels. Options: 'Class' (class name), 'Confidence' (confidence score), 'Class and Confidence' (both), 'Tracker Id' (tracking ID for tracked objects), 'Time In Zone' (time spent in zone), 'Dimensions' (center coordinates and width x height), 'Area' (bounding box area in pixels), 'Area (mask)' (mask area in pixels from Mask Area Measurement block), 'Area (converted)' (mask area in converted units from Mask Area Measurement block), or 'Index' (detection index)..
        - `text_position` (*[`string`](../kinds/string.md)*): Anchor position for placing labels relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object)..
        - `text_color` (*[`string`](../kinds/string.md)*): Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)')..
        - `text_scale` (*[`float`](../kinds/float.md)*): Scale factor for text size. Higher values create larger text. Default is 1.0..
        - `text_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility..
        - `text_padding` (*[`integer`](../kinds/integer.md)*): Padding around the text in pixels. Controls the spacing between the text and the label background border..
        - `border_radius` (*[`integer`](../kinds/integer.md)*): Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Label Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/label_visualization@v1",
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
	    "text": "LABEL",
	    "text_position": "CENTER",
	    "text_color": "WHITE",
	    "text_scale": 1.0,
	    "text_thickness": 1,
	    "text_padding": 10,
	    "border_radius": 0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

