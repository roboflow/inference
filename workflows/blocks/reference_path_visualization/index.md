
# Reference Path Visualization



??? "Class: `ReferencePathVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/reference_path/v1.py">inference.core.workflows.core_steps.visualizations.reference_path.v1.ReferencePathVisualizationBlockV1</a>
    



Draw a static reference path on an image to visualize an expected or ideal route, displaying a predefined polyline path that can be compared against actual object trajectories for path deviation analysis and route compliance monitoring.

## How This Block Works

This block takes an image and reference path coordinates (a list of points defining a path) and draws a static polyline path representing an expected route or ideal trajectory. The block:

1. Takes an image and reference path coordinates (a list of points: [(x1, y1), (x2, y2), (x3, y3), ...]) as input
2. Converts the coordinate list into a polyline path connecting the points in sequence
3. Draws the reference path as a polyline using the specified color and thickness
4. Returns an annotated image with the reference path overlaid on the original image

The block visualizes a static, predefined reference path that represents where objects should ideally move or what route they should follow. Unlike Trace Visualization (which draws dynamic paths based on actual tracked object movement), Reference Path Visualization draws a fixed path that remains constant. This reference path serves as a baseline for comparison, allowing you to visualize the expected route alongside actual object trajectories. The path is drawn as a continuous line connecting all the specified points, creating a visual guide for route compliance, path deviation analysis, or navigation workflows. This block is commonly used with Path Deviation analytics blocks to visually display the reference path that actual object trajectories will be compared against.

## Common Use Cases

- **Path Deviation Visualization**: Visualize a reference path alongside actual object trajectories to compare expected routes against actual movement for path deviation detection, route compliance monitoring, or navigation validation workflows
- **Route Planning and Navigation**: Display predefined routes, navigation paths, or expected travel routes that objects should follow for route planning, navigation systems, or waypoint visualization applications
- **Compliance and Safety Monitoring**: Visualize expected paths for safety monitoring, compliance workflows, or route validation where objects need to follow specific paths (e.g., vehicles on designated lanes, robots on expected routes)
- **Industrial and Logistics Applications**: Display reference paths for conveyor systems, automated guided vehicles (AGVs), or manufacturing workflows where objects must follow predefined routes for process control or quality assurance
- **Security and Access Control**: Visualize expected movement paths for security monitoring, access control, or surveillance workflows where deviations from expected routes need to be identified
- **Training and Documentation**: Display reference paths in training materials, documentation, or demonstrations to show expected object behavior, routes, or movement patterns for educational or reference purposes

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Path Deviation analytics blocks** to compare tracked object trajectories against the visualized reference path for deviation analysis
- **Other visualization blocks** (e.g., Trace Visualization, Bounding Box Visualization, Label Visualization) to combine reference path visualization with actual object tracking visualizations for comprehensive path comparison
- **Tracking blocks** (e.g., Byte Tracker) where the reference path can serve as a visual baseline for comparing actual tracked object trajectories
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with reference paths for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with reference paths to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with reference paths as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with reference paths for live monitoring, path visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/reference_path_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `reference_path` | `List[Any]` | Reference path coordinates in the format [(x1, y1), (x2, y2), (x3, y3), ...] defining the expected or ideal route. The path is drawn as a polyline connecting these points in sequence, creating a continuous line representing the reference trajectory. Typically connected from Path Deviation analytics blocks or defined manually as an expected route. Must contain at least two points to form a valid path.. | ✅ |
| `color` | `str` | Color of the reference path. Can be specified as a color name (e.g., 'WHITE', 'GREEN', 'BLUE'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(91, 181, 115)'). The reference path is drawn in this color with the specified thickness.. | ✅ |
| `thickness` | `int` | Thickness of the reference path line in pixels. Controls how thick the reference path appears. Higher values create thicker, more visible paths, while lower values create thinner, more subtle paths. Must be greater than or equal to zero. Typical values range from 1 to 5 pixels.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Reference Path Visualization` in version `v1`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Distance Measurement`](distance_measurement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md)
    - outputs: [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`SORT Tracker`](sort_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Blur Visualization`](blur_visualization.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Reference Path Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `reference_path` (*[`list_of_values`](../kinds/list_of_values.md)*): Reference path coordinates in the format [(x1, y1), (x2, y2), (x3, y3), ...] defining the expected or ideal route. The path is drawn as a polyline connecting these points in sequence, creating a continuous line representing the reference trajectory. Typically connected from Path Deviation analytics blocks or defined manually as an expected route. Must contain at least two points to form a valid path..
        - `color` (*[`string`](../kinds/string.md)*): Color of the reference path. Can be specified as a color name (e.g., 'WHITE', 'GREEN', 'BLUE'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(91, 181, 115)'). The reference path is drawn in this color with the specified thickness..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the reference path line in pixels. Controls how thick the reference path appears. Higher values create thicker, more visible paths, while lower values create thinner, more subtle paths. Must be greater than or equal to zero. Typical values range from 1 to 5 pixels..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Reference Path Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/reference_path_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "reference_path": "$inputs.expected_path",
	    "color": "WHITE",
	    "thickness": 2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

