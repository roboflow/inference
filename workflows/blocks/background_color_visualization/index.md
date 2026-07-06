
# Background Color Visualization



??? "Class: `BackgroundColorVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/background_color/v1.py">inference.core.workflows.core_steps.visualizations.background_color.v1.BackgroundColorVisualizationBlockV1</a>
    



Apply a colored overlay to areas outside detected regions, effectively masking the background while preserving detected objects at their original appearance.

## How This Block Works

This block takes an image and detection predictions and applies a colored overlay to all areas outside of the detected objects, leaving the detected regions unchanged. The block:

1. Takes an image and predictions as input
2. Creates a colored mask layer with the specified background color
3. Identifies detected regions from bounding boxes or segmentation masks (preserves detected objects)
4. Applies the colored overlay to all areas outside the detected regions with the specified opacity
5. Blends the colored overlay with the original image based on the opacity setting
6. Returns an annotated image where detected objects appear unchanged, while the background is filled with the specified color

The block works with both object detection predictions (using bounding boxes) and instance segmentation predictions (using masks). When masks are available, it preserves the exact shape of detected objects; otherwise, it uses bounding box regions. The opacity parameter controls how transparent or opaque the background overlay is, allowing you to create effects ranging from subtle background dimming (low opacity) to complete background replacement (high opacity). This creates a visual focus effect that highlights the detected objects by de-emphasizing or completely hiding the background.

## Common Use Cases

- **Object Focus and Highlighting**: Highlight detected objects by dimming or replacing the background, making objects stand out for presentations, documentation, or user interfaces
- **Background Removal Effects**: Create images where backgrounds are replaced with solid colors or semi-transparent overlays for product photography, content creation, or design workflows
- **Privacy and Anonymization**: Mask backgrounds while preserving detected objects (e.g., people, vehicles) to anonymize images, protect privacy, or comply with data protection requirements
- **Visual Debugging and Validation**: Dim backgrounds to focus attention on detected regions when validating model performance, checking detection accuracy, or debugging detection results
- **Presentation and Documentation**: Create clean, professional visualizations for reports, presentations, or documentation where you want to emphasize detected objects without distracting backgrounds
- **Content Creation and Editing**: Prepare images for further processing, compositing, or editing by isolating detected objects with colored backgrounds for easier manipulation or integration into other workflows

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Bounding Box Visualization, Polygon Visualization) to add additional annotations on top of the background-colored image for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with background coloring for documentation, reporting, or archiving
- **Webhook blocks** to send visualized results with background coloring to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with background coloring as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with background coloring for live monitoring, tracking visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/background_color_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `color` | `str` | Color to use for the background overlay. Areas outside detected regions will be filled with this color. Can be a color name (e.g., 'BLACK', 'WHITE') or color code in HEX format (e.g., '#000000') or RGB format (e.g., 'rgb(0, 0, 0)').. | ✅ |
| `opacity` | `float` | Opacity of the background overlay, ranging from 0.0 (fully transparent, original background visible) to 1.0 (fully opaque, complete background replacement). Values between 0.0 and 1.0 create a blend between the original image and the background color. Lower values create subtle background dimming, while higher values create stronger background replacement effects.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Background Color Visualization` in version `v1`.

    - inputs: [`Detections Filter`](detections_filter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Local File Sink`](local_file_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Current Time`](current_time.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Detections Merge`](detections_merge.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`Dominant Color`](dominant_color.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Background Color Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions to visualize..
        - `color` (*[`string`](../kinds/string.md)*): Color to use for the background overlay. Areas outside detected regions will be filled with this color. Can be a color name (e.g., 'BLACK', 'WHITE') or color code in HEX format (e.g., '#000000') or RGB format (e.g., 'rgb(0, 0, 0)')..
        - `opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the background overlay, ranging from 0.0 (fully transparent, original background visible) to 1.0 (fully opaque, complete background replacement). Values between 0.0 and 1.0 create a blend between the original image and the background color. Lower values create subtle background dimming, while higher values create stronger background replacement effects..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Background Color Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/background_color_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.object_detection_model.predictions",
	    "color": "WHITE",
	    "opacity": 0.5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

