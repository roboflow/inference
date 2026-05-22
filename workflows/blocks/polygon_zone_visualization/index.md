
# Polygon Zone Visualization



??? "Class: `PolygonZoneVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/polygon_zone/v1.py">inference.core.workflows.core_steps.visualizations.polygon_zone.v1.PolygonZoneVisualizationBlockV1</a>
    



Draw polygon zones on an image to visualize monitoring areas, displaying colored polygon overlays for zone-based detection and counting workflows that track objects within irregular, custom-defined regions.

## How This Block Works

This block takes an image and polygon zone coordinates (a list of points defining a polygon shape) and draws a filled polygon overlay to visualize the monitoring zone. The block:

1. Takes an image and polygon zone coordinates (a list of points: [(x1, y1), (x2, y2), (x3, y3), ...]) as input
2. Creates a filled polygon mask from the zone coordinates using the specified color
3. Overlays the filled polygon onto the image with the specified opacity, creating a semi-transparent zone visualization
4. Returns an annotated image with the polygon zone overlay on the original image

The block visualizes polygon zones used to define irregular monitoring areas for detection, counting, or tracking workflows. The polygon is drawn as a filled shape between the specified points, creating a closed region that can represent any custom area shape (unlike rectangular bounding boxes). This allows for flexible zone definitions that match real-world boundaries, such as specific floor areas, irregular regions of interest, or complex monitoring zones. The zone overlay is semi-transparent, allowing the underlying image details to remain visible while clearly indicating the monitoring area. Note: This block should typically be placed before other visualization blocks in the workflow, as the polygon zone provides a background reference layer for object detection visualizations.

## Common Use Cases

- **Zone Detection Visualization**: Visualize polygon zones for object detection or counting workflows where objects are tracked within irregular, custom-defined areas, displaying the monitoring boundaries clearly
- **Area-Based Monitoring**: Display polygon zones for area-based monitoring applications such as occupancy tracking, people counting in specific regions, or object presence detection within defined spaces
- **Custom Region Visualization**: Visualize custom monitoring regions that don't fit rectangular boundaries, such as irregular floor areas, complex room layouts, or specific zones within larger spaces
- **Security and Surveillance**: Display polygon zones for security monitoring, access control, or surveillance workflows where specific areas need to be visually marked and monitored
- **Retail and Business Analytics**: Visualize polygon zones for foot traffic analysis, customer movement tracking, or space utilization monitoring in retail, hospitality, or business intelligence applications
- **Real-Time Zone Monitoring**: Create visual overlays for real-time monitoring dashboards, live video feeds, or monitoring interfaces where polygon zones need to be clearly visible to indicate monitored areas

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Zone detection or counting blocks** to receive polygon zone coordinates that are visualized
- **Other visualization blocks** (e.g., Bounding Box Visualization, Label Visualization, Polygon Visualization) to add object detection annotations on top of the polygon zone visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with polygon zone visualizations for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with polygon zones to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with polygon zones as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with polygon zone visualizations for live monitoring, zone visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/polygon_zone_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `zone` | `List[Any]` | Polygon zone coordinates in the format [[(x1, y1), (x2, y2), (x3, y3), ...], ...] defining one or more polygon shapes. Each zone must consist of more than 2 points to form a valid polygon. The polygon is drawn as a filled shape connecting these points in order, creating a closed region. Typically connected from zone detection or counting blocks that define monitoring areas.. | ✅ |
| `color` | `str` | Color of the polygon zone overlay. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The polygon is filled with this color and overlaid with the specified opacity.. | ✅ |
| `opacity` | `float` | Opacity of the polygon zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the polygon zone appears over the image. Lower values create more transparent zones that blend with the background, while higher values create more opaque, visible zones. Typical values range from 0.2 to 0.5 for balanced visibility where both the zone and underlying image are visible.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Polygon Zone Visualization` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`JSON Parser`](json_parser.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Image Blur`](image_blur.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Buffer`](buffer.md), [`Google Gemini`](google_gemini.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Identify Changes`](identify_changes.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CSV Formatter`](csv_formatter.md), [`S3 Sink`](s3_sink.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Outliers`](identify_outliers.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen-VL`](qwen_vl.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Webhook Sink`](webhook_sink.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`CogVLM`](cog_vlm.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Camera Focus`](camera_focus.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Label Visualization`](label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Calibration`](camera_calibration.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)
    - outputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Seg Preview`](seg_preview.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Blur`](image_blur.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Buffer`](buffer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`LMM`](lmm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Object Detection Model`](object_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen3.5`](qwen3.5.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`CogVLM`](cog_vlm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Moondream2`](moondream2.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Template Matching`](template_matching.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`QR Code Detection`](qr_code_detection.md), [`Label Visualization`](label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Circle Visualization`](circle_visualization.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Polygon Zone Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `zone` (*[`list_of_values`](../kinds/list_of_values.md)*): Polygon zone coordinates in the format [[(x1, y1), (x2, y2), (x3, y3), ...], ...] defining one or more polygon shapes. Each zone must consist of more than 2 points to form a valid polygon. The polygon is drawn as a filled shape connecting these points in order, creating a closed region. Typically connected from zone detection or counting blocks that define monitoring areas..
        - `color` (*[`string`](../kinds/string.md)*): Color of the polygon zone overlay. Can be specified as a color name (e.g., 'WHITE', 'RED'), hex color code (e.g., '#5bb573', '#FFFFFF'), or RGB format (e.g., 'rgb(255, 255, 255)'). The polygon is filled with this color and overlaid with the specified opacity..
        - `opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the polygon zone overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls how transparent the polygon zone appears over the image. Lower values create more transparent zones that blend with the background, while higher values create more opaque, visible zones. Typical values range from 0.2 to 0.5 for balanced visibility where both the zone and underlying image are visible..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Polygon Zone Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/polygon_zone_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "zone": "$inputs.zones",
	    "color": "WHITE",
	    "opacity": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

