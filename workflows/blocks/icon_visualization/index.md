
# Icon Visualization



??? "Class: `IconVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/icon/v1.py">inference.core.workflows.core_steps.visualizations.icon.v1.IconVisualizationBlockV1</a>
    



Place custom icon images on images either at fixed positions (static mode) or dynamically positioned on detected objects (dynamic mode), useful for watermarks, labels, badges, or visual markers.

## How This Block Works

This block takes an image and optionally detection predictions, then places a custom icon image on the image. The block supports two modes:

**Static Mode** (for watermarks and fixed positioning):
1. Takes an image and an icon image as input
2. Places the icon at fixed x and y coordinates on the image
3. Supports negative coordinates for positioning from the right or bottom edges
4. Returns an annotated image with the icon at the specified static location

**Dynamic Mode** (for detection-based positioning):
1. Takes an image, an icon image, and detection predictions as input
2. Positions the icon on each detected object based on the selected anchor point (center, corners, edges, or center of mass)
3. Places the icon at the same position relative to each detection
4. Returns an annotated image with icons overlaid on detected objects

The block supports PNG images with transparency (alpha channel), allowing icons to blend naturally with the background. Icons can be resized to any width and height, making them suitable for various use cases from small badges to large watermarks. In static mode, icons are placed at fixed coordinates, making it ideal for watermarks or branding. In dynamic mode, icons automatically follow detected objects, making it useful for labeling, categorizing, or marking detected items with custom visual indicators.

## Common Use Cases

- **Watermarks and Branding**: Place logos, watermarks, or branding elements at fixed positions (static mode) on images or videos for content protection, copyright marking, or brand identification
- **Object Labeling with Icons**: Place custom icons on detected objects (dynamic mode) to categorize, label, or mark objects with visual indicators (e.g., warning icons on unsafe objects, category icons for products, status badges)
- **Visual Status Indicators**: Display status icons (e.g., checkmarks, warning signs, information badges) on detected objects based on classification results, confidence levels, or custom logic for quick visual feedback
- **Product Marking and Categorization**: Place category icons, product type indicators, or custom markers on detected products in retail, e-commerce, or inventory management workflows
- **Custom Annotation Systems**: Create custom annotation workflows with specialized icons for quality control, defect marking, or compliance tracking in manufacturing or inspection workflows
- **Interactive UI Elements**: Add icon-based visual elements to images or videos for user interfaces, dashboards, or interactive applications where custom icons provide intuitive visual cues

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to combine icon placement with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with icons for documentation, reporting, or archiving
- **Webhook blocks** to send visualized results with icons to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with icons as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with icons for live monitoring, tracking visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/icon_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `mode` | `str` | Mode for placing icons. 'static' mode places the icon at fixed x,y coordinates (useful for watermarks or fixed-position elements). 'dynamic' mode places icons on detected objects based on their positions (useful for object labeling or categorization).. | ✅ |
| `icon_width` | `int` | Width of the icon in pixels. The icon image will be resized to this width while maintaining aspect ratio if height is also specified.. | ✅ |
| `icon_height` | `int` | Height of the icon in pixels. The icon image will be resized to this height while maintaining aspect ratio if width is also specified.. | ✅ |
| `position` | `str` | Anchor position for placing icons relative to each detection's bounding box (dynamic mode only). Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).. | ✅ |
| `x_position` | `int` | X coordinate for static mode positioning. Positive values position from the left edge of the image. Negative values position from the right edge (e.g., -10 places the icon 10 pixels from the right edge).. | ✅ |
| `y_position` | `int` | Y coordinate for static mode positioning. Positive values position from the top edge of the image. Negative values position from the bottom edge (e.g., -10 places the icon 10 pixels from the bottom edge).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Icon Visualization` in version `v1`.

    - inputs: [`Detections Filter`](detections_filter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Velocity`](velocity.md), [`LMM`](lmm.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Google Gemma API`](google_gemma_api.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Local File Sink`](local_file_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Current Time`](current_time.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Detections Merge`](detections_merge.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`Dominant Color`](dominant_color.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Icon Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `icon` (*[`image`](../kinds/image.md)*): The icon image to place on the input image. PNG format with transparency (alpha channel) is recommended for best results, as it allows the icon to blend naturally with the background. The icon will be resized to the specified width and height..
        - `mode` (*[`string`](../kinds/string.md)*): Mode for placing icons. 'static' mode places the icon at fixed x,y coordinates (useful for watermarks or fixed-position elements). 'dynamic' mode places icons on detected objects based on their positions (useful for object labeling or categorization)..
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions to place icons on (required for dynamic mode). Icons will be positioned on each detected object based on the selected position anchor point..
        - `icon_width` (*[`integer`](../kinds/integer.md)*): Width of the icon in pixels. The icon image will be resized to this width while maintaining aspect ratio if height is also specified..
        - `icon_height` (*[`integer`](../kinds/integer.md)*): Height of the icon in pixels. The icon image will be resized to this height while maintaining aspect ratio if width is also specified..
        - `position` (*[`string`](../kinds/string.md)*): Anchor position for placing icons relative to each detection's bounding box (dynamic mode only). Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object)..
        - `x_position` (*[`integer`](../kinds/integer.md)*): X coordinate for static mode positioning. Positive values position from the left edge of the image. Negative values position from the right edge (e.g., -10 places the icon 10 pixels from the right edge)..
        - `y_position` (*[`integer`](../kinds/integer.md)*): Y coordinate for static mode positioning. Positive values position from the top edge of the image. Negative values position from the bottom edge (e.g., -10 places the icon 10 pixels from the bottom edge)..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Icon Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/icon_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "icon": "$inputs.icon",
	    "mode": "static",
	    "predictions": "$steps.object_detection_model.predictions",
	    "icon_width": 64,
	    "icon_height": 64,
	    "position": "TOP_CENTER",
	    "x_position": 10,
	    "y_position": 10
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

