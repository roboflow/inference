
# Keypoint Visualization



??? "Class: `KeypointVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/keypoint/v1.py">inference.core.workflows.core_steps.visualizations.keypoint.v1.KeypointVisualizationBlockV1</a>
    



Visualize keypoints (landmark points) detected on objects by drawing point markers, connecting edges, or labeled vertices, providing pose estimation visualization for anatomical points, structural landmarks, or object key features.

## How This Block Works

This block takes an image and keypoint detection predictions and visualizes the detected keypoints using one of three visualization modes. The block:

1. Takes an image and keypoint detection predictions as input (predictions must include keypoint coordinates, confidence scores, and class names)
2. Extracts keypoint data (coordinates, confidence values, and class names) from the predictions
3. Converts the detection data into a KeyPoints format suitable for visualization
4. Applies one of three visualization modes based on the annotator_type setting:
   - **Edge mode**: Draws connecting lines (edges) between keypoints using specified edge pairs to show keypoint relationships (e.g., skeleton connections in pose estimation)
   - **Vertex mode**: Draws circular markers at each keypoint location without connections, showing individual keypoint positions
   - **Vertex label mode**: Draws circular markers with text labels identifying each keypoint class name, providing labeled keypoint visualization
5. Applies color styling, sizing, and optional text labeling based on the selected parameters
6. Returns an annotated image with keypoints visualized according to the selected mode

The block supports three visualization styles to suit different use cases. Edge mode connects related keypoints with lines (useful for pose estimation skeletons or structural relationships), vertex mode shows individual keypoint locations as circular markers, and vertex label mode adds text labels to identify each keypoint type. This visualization is essential for pose estimation workflows, anatomical point detection, or any application where specific landmark points on objects need to be identified and visualized.

## Common Use Cases

- **Human Pose Estimation**: Visualize human body keypoints (joints, body parts) for pose estimation, activity recognition, or motion analysis applications where anatomical points need to be displayed with skeleton connections or labeled markers
- **Animal Pose Estimation**: Display animal keypoints for behavior analysis, veterinary applications, or wildlife monitoring where anatomical landmarks need to be visualized for pose analysis or movement tracking
- **Structural Landmark Detection**: Visualize keypoints on objects, structures, or machinery for structural analysis, quality control, or measurement workflows where specific landmark points need to be identified and displayed
- **Facial Landmark Detection**: Display facial keypoints (eye corners, nose tip, mouth corners, etc.) for facial recognition, expression analysis, or face alignment applications where facial features need to be visualized
- **Sports and Movement Analysis**: Visualize keypoints for sports analysis, biomechanics, or movement studies where body positions, joint angles, or movement patterns need to be analyzed and displayed
- **Quality Control and Inspection**: Display keypoints for manufacturing, quality assurance, or inspection workflows where specific points on products or components need to be identified, measured, or validated

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Keypoint Detection Model blocks** to receive keypoint predictions that are visualized with point markers, edges, or labeled vertices
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to combine keypoint visualization with additional annotations for comprehensive pose or structure visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with keypoint visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with keypoints to external systems, APIs, or web applications for display in dashboards, pose analysis tools, or monitoring interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with keypoints as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with keypoint visualizations for live pose estimation, movement analysis, or post-processing workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/keypoint_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `annotator_type` | `str` | Type of keypoint visualization mode. Options: 'edge' (draws connecting lines between keypoints using edge pairs, useful for skeleton/pose visualization), 'vertex' (draws circular markers at keypoint locations without connections), 'vertex_label' (draws circular markers with text labels identifying each keypoint class name).. | ❌ |
| `color` | `str` | Color of the keypoint markers, edges, or labels. Can be specified as a color name (e.g., 'green', 'red', 'blue'), hex color code (e.g., '#A351FB', '#FF0000'), or RGB format. Used for keypoint circles (vertex/vertex_label modes) or edge lines (edge mode).. | ✅ |
| `text_color` | `str` | Color of the text labels displayed on keypoints (vertex_label mode only). Can be specified as a color name (e.g., 'black', 'white'), hex color code, or RGB format. Only applies when annotator_type is 'vertex_label'.. | ✅ |
| `text_scale` | `float` | Scale factor for keypoint label text size (vertex_label mode only). Controls the size of text labels displayed on keypoints. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Only applies when annotator_type is 'vertex_label'. Typical values range from 0.3 to 1.0.. | ✅ |
| `text_thickness` | `int` | Thickness of the keypoint label text characters in pixels (vertex_label mode only). Controls how bold the text labels appear. Higher values create thicker, bolder text. Only applies when annotator_type is 'vertex_label'. Typical values range from 1 to 3.. | ✅ |
| `text_padding` | `int` | Padding around keypoint label text in pixels (vertex_label mode only). Controls the spacing between the text label and its background border. Higher values create more space around text. Only applies when annotator_type is 'vertex_label'. Typical values range from 5 to 20 pixels.. | ✅ |
| `thickness` | `int` | Thickness of the edge lines connecting keypoints in pixels (edge mode only). Controls how thick the connecting lines between keypoints appear. Higher values create thicker, more visible edges. Only applies when annotator_type is 'edge'. Typical values range from 1 to 5 pixels.. | ✅ |
| `radius` | `int` | Radius of the circular keypoint markers in pixels (vertex and vertex_label modes only). Controls the size of circular markers drawn at keypoint locations. Higher values create larger, more visible markers. Only applies when annotator_type is 'vertex' or 'vertex_label'. Typical values range from 5 to 20 pixels.. | ✅ |
| `edges` | `List[Any]` | Edge connections between keypoints (edge mode only). List of pairs of keypoint indices (e.g., [(0, 1), (1, 2), ...]) defining which keypoints should be connected with lines. For pose estimation, this typically represents skeleton connections (e.g., connecting joints). Only applies when annotator_type is 'edge'. Required for edge visualization.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Keypoint Visualization` in version `v1`.

    - inputs: [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Cosine Similarity`](cosine_similarity.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Transformation`](detections_transformation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`PLC Reader`](plc_reader.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PP-OCR`](ppocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`GeoTag Detection`](geo_tag_detection.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Identify Outliers`](identify_outliers.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections Filter`](detections_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`JSON Parser`](json_parser.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Keypoint Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)*): Keypoint detection predictions containing keypoint coordinates, confidence scores, and class names. Predictions must include keypoints_xy (keypoint coordinates), keypoints_confidence (confidence values), and keypoints_class_name (keypoint class/type names). Requires outputs from a keypoint detection model block..
        - `color` (*[`string`](../kinds/string.md)*): Color of the keypoint markers, edges, or labels. Can be specified as a color name (e.g., 'green', 'red', 'blue'), hex color code (e.g., '#A351FB', '#FF0000'), or RGB format. Used for keypoint circles (vertex/vertex_label modes) or edge lines (edge mode)..
        - `text_color` (*[`string`](../kinds/string.md)*): Color of the text labels displayed on keypoints (vertex_label mode only). Can be specified as a color name (e.g., 'black', 'white'), hex color code, or RGB format. Only applies when annotator_type is 'vertex_label'..
        - `text_scale` (*[`float`](../kinds/float.md)*): Scale factor for keypoint label text size (vertex_label mode only). Controls the size of text labels displayed on keypoints. Values greater than 1.0 make text larger, values less than 1.0 make text smaller. Only applies when annotator_type is 'vertex_label'. Typical values range from 0.3 to 1.0..
        - `text_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the keypoint label text characters in pixels (vertex_label mode only). Controls how bold the text labels appear. Higher values create thicker, bolder text. Only applies when annotator_type is 'vertex_label'. Typical values range from 1 to 3..
        - `text_padding` (*[`integer`](../kinds/integer.md)*): Padding around keypoint label text in pixels (vertex_label mode only). Controls the spacing between the text label and its background border. Higher values create more space around text. Only applies when annotator_type is 'vertex_label'. Typical values range from 5 to 20 pixels..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the edge lines connecting keypoints in pixels (edge mode only). Controls how thick the connecting lines between keypoints appear. Higher values create thicker, more visible edges. Only applies when annotator_type is 'edge'. Typical values range from 1 to 5 pixels..
        - `radius` (*[`integer`](../kinds/integer.md)*): Radius of the circular keypoint markers in pixels (vertex and vertex_label modes only). Controls the size of circular markers drawn at keypoint locations. Higher values create larger, more visible markers. Only applies when annotator_type is 'vertex' or 'vertex_label'. Typical values range from 5 to 20 pixels..
        - `edges` (*[`list_of_values`](../kinds/list_of_values.md)*): Edge connections between keypoints (edge mode only). List of pairs of keypoint indices (e.g., [(0, 1), (1, 2), ...]) defining which keypoints should be connected with lines. For pose estimation, this typically represents skeleton connections (e.g., connecting joints). Only applies when annotator_type is 'edge'. Required for edge visualization..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Keypoint Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/keypoint_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.keypoint_detection_model.predictions",
	    "annotator_type": "<block_does_not_provide_example>",
	    "color": "#A351FB",
	    "text_color": "black",
	    "text_scale": 0.5,
	    "text_thickness": 1,
	    "text_padding": 10,
	    "thickness": 2,
	    "radius": 10,
	    "edges": "$inputs.edges"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

