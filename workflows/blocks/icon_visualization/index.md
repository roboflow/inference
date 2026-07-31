
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

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Corner Visualization`](corner_visualization.md), [`Path Deviation`](path_deviation.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Detection Offset`](detection_offset.md), [`JSON Parser`](json_parser.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Combine`](detections_combine.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Cosmos 3`](cosmos3.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Outliers`](identify_outliers.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)
    - outputs: [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Camera Calibration`](camera_calibration.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Perspective Correction`](perspective_correction.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Icon Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `icon` (*[`image`](../kinds/image.md)*): The icon image to place on the input image. PNG format with transparency (alpha channel) is recommended for best results, as it allows the icon to blend naturally with the background. The icon will be resized to the specified width and height..
        - `mode` (*[`string`](../kinds/string.md)*): Mode for placing icons. 'static' mode places the icon at fixed x,y coordinates (useful for watermarks or fixed-position elements). 'dynamic' mode places icons on detected objects based on their positions (useful for object labeling or categorization)..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions to place icons on (required for dynamic mode). Icons will be positioned on each detected object based on the selected position anchor point..
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

