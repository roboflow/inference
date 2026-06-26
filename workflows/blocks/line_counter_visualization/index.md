
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

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`Moondream2`](moondream2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Dominant Color`](dominant_color.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Crop Visualization`](crop_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
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

