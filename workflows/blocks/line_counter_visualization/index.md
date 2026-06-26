
# Line Counter Visualization



??? "Class: `LineCounterZoneVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/line_zone/v1.py">inference.core.workflows.core_steps.visualizations.line_zone.v1.LineCounterZoneVisualizationBlockV1</a>
    



Draw a line zone on an image to visualize counting boundaries, displaying a colored line overlay with in/out count labels for line counter workflows that track objects crossing a specified line.

## How This Block Works

This block takes an image and line zone coordinates (two points defining a line) and draws a visual representation of the counting line with count statistics. The block:

1. Takes an image and line zone coordinates (two points: [x1, y1] and [x2, y2]) as input
2. Creates a line mask from the zone coordinates using the specified color and thickness
3. Overlays the line onto the image with the specified opacity, creating a semi-transparent line visualization
4. Displays text labels showing the count_in (objects that crossed into the zone) and count_out (objects that crossed out of the zone) values
5. Positions the count text at the starting point of the line (x1, y1) with customizable text styling
6. Returns an annotated image with the line zone and count statistics overlaid on the original image

The block visualizes line counting zones used to track object movement across a defined boundary line. The line is drawn between the two specified points with customizable color, thickness, and opacity. Count statistics (in and out) are displayed as text labels, typically connected from a Line Counter block that tracks object crossings. The visualization helps users see the counting boundary and monitor counting results in real-time. Note: This block should typically be placed before other visualization blocks in the workflow, as the line zone provides a background reference layer for object detection visualizations.

## Common Use Cases

- **Line Counter Visualization**: Visualize line counting zones for people counting, vehicle counting, or object tracking workflows where objects cross a defined line boundary, displaying the counting line and in/out statistics
- **Traffic and Movement Monitoring**: Display counting lines for traffic monitoring, pedestrian flow analysis, or entry/exit tracking applications where you need to visualize the counting boundary and current counts
- **Checkpoint and Access Control**: Visualize counting lines at checkpoints, gates, or access points to show the monitoring boundary and track entry/exit counts for security or access control workflows
- **Retail and Business Analytics**: Display counting lines for foot traffic analysis, customer flow monitoring, or occupancy tracking in retail, hospitality, or business intelligence applications
- **Crowd Management and Safety**: Visualize counting lines for crowd management, capacity monitoring, or safety workflows where tracking object movement across boundaries is critical
- **Real-Time Counting Dashboards**: Create visual overlays for real-time counting dashboards, monitoring interfaces, or live video feeds where the counting line and statistics need to be clearly visible

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Line Counter blocks** to receive count_in and count_out values that are displayed on the visualization
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to add object detection annotations on top of the line zone visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with line zone visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with line zones to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with line zones as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with line zone visualizations for live monitoring, counting visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/line_counter_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `zone` | `List[Any]` | Line zone coordinates in the format [[x1, y1], [x2, y2]] consisting of exactly two points that define the counting line. The line is drawn between these two points, and objects crossing this line are counted. Typically connected from a Line Counter block's zone output.. | ✅ |
| `color` | `str` | Color of the line zone. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The line is drawn in this color with the specified opacity.. | ✅ |
| `thickness` | `int` | Thickness of the line zone in pixels. Controls how thick the counting line appears. Higher values create thicker, more visible lines, while lower values create thinner lines. Typical values range from 1 to 10 pixels.. | ✅ |
| `text_thickness` | `int` | Thickness of the count text labels in pixels. Controls how bold the text appears (line width of text characters). Higher values create thicker, bolder text, while lower values create thinner text. Typical values range from 1 to 3.. | ✅ |
| `text_scale` | `float` | Scale factor for the count text labels. Controls the size of the text displaying count_in and count_out values. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Typical values range from 0.5 to 2.0.. | ✅ |
| `count_in` | `int` | Number of objects that crossed into the line zone (crossing from one side to the other in the 'in' direction). Typically connected from a Line Counter block's count_in output (e.g., '$steps.line_counter.count_in'). This value is displayed in the visualization text label.. | ✅ |
| `count_out` | `int` | Number of objects that crossed out of the line zone (crossing from one side to the other in the 'out' direction). Typically connected from a Line Counter block's count_out output (e.g., '$steps.line_counter.count_out'). This value is displayed in the visualization text label.. | ✅ |
| `opacity` | `float` | Opacity of the line zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the counting line appears over the image. Lower values create more transparent lines that blend with the background, while higher values create more opaque, visible lines. Typical values range from 0.2 to 0.5 for balanced visibility.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Line Counter Visualization` in version `v1`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Image Contours`](image_contours.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`PLC Reader`](plc_reader.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen3.5`](qwen3.5.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Line Counter Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `zone` (*[`list_of_values`](../kinds/list_of_values.md)*): Line zone coordinates in the format [[x1, y1], [x2, y2]] consisting of exactly two points that define the counting line. The line is drawn between these two points, and objects crossing this line are counted. Typically connected from a Line Counter block's zone output..
        - `color` (*[`string`](../kinds/string.md)*): Color of the line zone. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The line is drawn in this color with the specified opacity..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the line zone in pixels. Controls how thick the counting line appears. Higher values create thicker, more visible lines, while lower values create thinner lines. Typical values range from 1 to 10 pixels..
        - `text_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the count text labels in pixels. Controls how bold the text appears (line width of text characters). Higher values create thicker, bolder text, while lower values create thinner text. Typical values range from 1 to 3..
        - `text_scale` (*[`float`](../kinds/float.md)*): Scale factor for the count text labels. Controls the size of the text displaying count_in and count_out values. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Typical values range from 0.5 to 2.0..
        - `count_in` (*[`integer`](../kinds/integer.md)*): Number of objects that crossed into the line zone (crossing from one side to the other in the 'in' direction). Typically connected from a Line Counter block's count_in output (e.g., '$steps.line_counter.count_in'). This value is displayed in the visualization text label..
        - `count_out` (*[`integer`](../kinds/integer.md)*): Number of objects that crossed out of the line zone (crossing from one side to the other in the 'out' direction). Typically connected from a Line Counter block's count_out output (e.g., '$steps.line_counter.count_out'). This value is displayed in the visualization text label..
        - `opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the line zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the counting line appears over the image. Lower values create more transparent lines that blend with the background, while higher values create more opaque, visible lines. Typical values range from 0.2 to 0.5 for balanced visibility..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Line Counter Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/line_counter_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "zone": [
	        [
	            0,
	            50
	        ],
	        [
	            500,
	            50
	        ]
	    ],
	    "color": "WHITE",
	    "thickness": 2,
	    "text_thickness": 1,
	    "text_scale": 1.0,
	    "count_in": "$steps.line_counter.count_in",
	    "count_out": "$steps.line_counter.count_out",
	    "opacity": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

