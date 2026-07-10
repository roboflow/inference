
# Classification Label Visualization



??? "Class: `ClassificationLabelVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/classification_label/v1.py">inference.core.workflows.core_steps.visualizations.classification_label.v1.ClassificationLabelVisualizationBlockV1</a>
    



Visualize classification predictions as text labels positioned on images, automatically handling both single-label and multi-label classification formats with customizable styling and positioning.

## How This Block Works

This block takes an image and classification predictions (for entire image classification, not object detection) and displays text labels showing the predicted class names and confidence scores. The block:

1. Takes an image and classification predictions as input
2. Automatically detects whether predictions are single-label (one class per image) or multi-label (multiple classes per image)
3. For single-label predictions: selects the highest confidence prediction to display
4. For multi-label predictions: formats and sorts all predicted classes by confidence score (highest first)
5. Extracts label text based on the selected text option (class name, confidence score, or both)
6. Positions labels on the image at the specified location (top, center, or bottom edges, with left/center/right alignment)
7. Applies background color styling based on the selected color palette, with colors assigned by class
8. Renders text labels with customizable text color, scale, thickness, padding, and border radius
9. Returns an annotated image with classification labels overlaid on the original image

Unlike the regular Label Visualization block (which labels detected objects with bounding boxes), this block is designed for image-level classification where the entire image is classified into one or more categories. Labels are positioned at the edges or center of the image itself, not relative to object locations. For multi-label predictions, multiple labels are stacked vertically at the chosen position, making it easy to see all predicted classes and their confidence scores.

## Common Use Cases

- **Image Classification Results Display**: Visualize the predicted class and confidence score for classified images in applications like content moderation, product categorization, or medical image analysis
- **Multi-Class Probability Visualization**: Display multiple predicted classes with their confidence scores for multi-label classification tasks, such as tagging images with multiple attributes, detecting multiple defects, or identifying multiple objects in scene classification
- **Model Performance Validation**: Show classification predictions directly on images to validate model performance, verify correct classifications, and identify misclassifications during model development or testing
- **User Interface Integration**: Create clean, professional displays of classification results for applications, dashboards, or mobile apps where users need to see what an image was classified as
- **Documentation and Reporting**: Generate annotated images showing classification results for reports, documentation, or training data review to demonstrate model predictions
- **Quality Control Workflows**: Display classification results on production images for quality control, content filtering, or automated categorization workflows where visual confirmation of predictions is needed

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with classification labels for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with classification labels to external systems, APIs, or web applications for display in dashboards or classification monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with classification labels as visual evidence in alerts or reports when specific classes are detected
- **Video output blocks** to create annotated video streams or recordings with classification labels for live monitoring, real-time classification display, or post-processing analysis
- **Conditional logic blocks** (e.g., Continue If) to route workflow execution based on classification results or confidence scores displayed in the labels


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/classification_label_visualization@v1`to add the block as
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
| `text` | `str` | Content to display in text labels. Options: 'Class' (class name only), 'Confidence' (confidence score only, formatted as decimal), or 'Class and Confidence' (both class name and confidence score).. | ✅ |
| `text_position` | `str` | Position for placing labels on the image. Options include: TOP (TOP_LEFT, TOP_CENTER, TOP_RIGHT), CENTER (CENTER_LEFT, CENTER, CENTER_RIGHT), or BOTTOM (BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT). For multi-label predictions, labels are stacked vertically at the chosen position.. | ✅ |
| `text_color` | `str` | Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').. | ✅ |
| `text_scale` | `float` | Scale factor for text size. Higher values create larger text. Default is 1.0.. | ✅ |
| `text_thickness` | `int` | Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility.. | ✅ |
| `text_padding` | `int` | Padding around the text in pixels. Controls the spacing between the text and the label background border, and the spacing between multiple labels in multi-label predictions.. | ✅ |
| `border_radius` | `int` | Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Classification Label Visualization` in version `v1`.

    - inputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`PLC Writer`](plc_writer.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`JSON Parser`](json_parser.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Threshold`](image_threshold.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`CogVLM`](cog_vlm.md), [`MQTT Writer`](mqtt_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Event Writer`](event_writer.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Buffer`](buffer.md), [`Slack Notification`](slack_notification.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`Local File Sink`](local_file_sink.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PP-OCR`](ppocr.md), [`Blur Visualization`](blur_visualization.md), [`EasyOCR`](easy_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`CSV Formatter`](csv_formatter.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Motion Detection`](motion_detection.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Current Time`](current_time.md), [`PLC Reader`](plc_reader.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Outliers`](identify_outliers.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Detection Event Log`](detection_event_log.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Morphological Transformation`](morphological_transformation.md), [`Relative Static Crop`](relative_static_crop.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenRouter`](open_router.md), [`Identify Changes`](identify_changes.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT Comparison`](sift_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)
    - outputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Threshold`](image_threshold.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`CogVLM`](cog_vlm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM`](lmm.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`SAM 3`](sam3.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Event Writer`](event_writer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemma`](google_gemma.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`Crop Visualization`](crop_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Object Detection Model`](object_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`PP-OCR`](ppocr.md), [`Blur Visualization`](blur_visualization.md), [`EasyOCR`](easy_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Motion Detection`](motion_detection.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Image Stack`](image_stack.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Relative Static Crop`](relative_static_crop.md), [`Background Color Visualization`](background_color_visualization.md), [`Clip Comparison`](clip_comparison.md), [`GeoTag Detection`](geo_tag_detection.md), [`Corner Visualization`](corner_visualization.md), [`SIFT`](sift.md), [`Track Class Lock`](track_class_lock.md), [`Dominant Color`](dominant_color.md), [`Moondream2`](moondream2.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT Comparison`](sift_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Classification Label Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*[`classification_prediction`](../kinds/classification_prediction.md)*): Classification predictions from a single-label or multi-label classification model. The block automatically detects the prediction format and handles both types accordingly..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `text` (*[`string`](../kinds/string.md)*): Content to display in text labels. Options: 'Class' (class name only), 'Confidence' (confidence score only, formatted as decimal), or 'Class and Confidence' (both class name and confidence score)..
        - `text_position` (*[`string`](../kinds/string.md)*): Position for placing labels on the image. Options include: TOP (TOP_LEFT, TOP_CENTER, TOP_RIGHT), CENTER (CENTER_LEFT, CENTER, CENTER_RIGHT), or BOTTOM (BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT). For multi-label predictions, labels are stacked vertically at the chosen position..
        - `text_color` (*[`string`](../kinds/string.md)*): Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)')..
        - `text_scale` (*[`float`](../kinds/float.md)*): Scale factor for text size. Higher values create larger text. Default is 1.0..
        - `text_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility..
        - `text_padding` (*[`integer`](../kinds/integer.md)*): Padding around the text in pixels. Controls the spacing between the text and the label background border, and the spacing between multiple labels in multi-label predictions..
        - `border_radius` (*[`integer`](../kinds/integer.md)*): Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Classification Label Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/classification_label_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.classification_model.predictions",
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

