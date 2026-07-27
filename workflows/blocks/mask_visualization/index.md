
# Mask Visualization



??? "Class: `MaskVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/mask/v1.py">inference.core.workflows.core_steps.visualizations.mask.v1.MaskVisualizationBlockV1</a>
    



Fill segmentation masks with semi-transparent color overlays, creating solid color fills that precisely follow the shape of detected objects from instance segmentation predictions.

## How This Block Works

This block takes an image and instance segmentation predictions (with masks) and fills the mask regions with colored overlays. The block:

1. Takes an image and instance segmentation predictions (with masks) as input
2. Extracts segmentation masks for each detected object from the predictions
3. Applies color styling to each mask based on the selected color palette, with colors assigned by class, index, or track ID
4. Fills the mask regions with solid colors using Supervision's MaskAnnotator
5. Blends the colored mask overlays with the original image using the specified opacity level
6. Returns an annotated image where mask regions are filled with semi-transparent colors, while non-masked areas remain unchanged

The block fills the exact shape of each object's segmentation mask with colored overlays, creating solid color fills that precisely follow object boundaries. Unlike polygon visualization (which draws outlines) or bounding box visualizations (which use rectangular regions), mask visualization fills the entire mask area with color, providing clear visual indication of the segmented regions. The opacity parameter controls how transparent the mask overlay is, allowing you to see the original image details through the colored mask (lower opacity) or create more opaque fills (higher opacity) that better obscure background details. This block requires instance segmentation predictions with mask data, as it specifically works with segmentation masks to create precise, shape-following color fills.

## Common Use Cases

- **Instance Segmentation Visualization**: Visualize instance segmentation results by filling mask regions with colors to clearly show segmented objects, validate segmentation quality, or highlight detected regions in analysis workflows
- **Precise Shape-Following Overlays**: Fill objects with colors that exactly match their segmented shapes, useful for applications requiring accurate region visualization such as medical imaging, quality control, or precise object identification
- **Mask-Based Object Highlighting**: Highlight segmented objects with colored overlays that follow exact object boundaries, providing clear visual distinction between different objects or object classes
- **Segmentation Model Validation**: Visualize segmentation predictions with colored mask fills to verify model performance, identify segmentation errors, or validate mask accuracy in model development and debugging workflows
- **Medical and Scientific Imaging**: Display segmented regions in medical imaging, microscopy, or scientific analysis applications where colored mask overlays help visualize tissue boundaries, cell regions, or measured areas
- **Mask Quality Inspection**: Use colored mask fills to inspect segmentation quality, verify mask boundaries, or identify areas where segmentation may need improvement in training data or model outputs

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Polygon Visualization, Bounding Box Visualization) to combine mask fills with additional annotations (labels, outlines) for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with mask overlays for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with mask fills to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with mask overlays as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with mask fills for live monitoring, segmentation visualization, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/mask_visualization@v1`to add the block as
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
| `opacity` | `float` | Opacity of the mask overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls the transparency of the colored mask fill. Lower values (e.g., 0.3-0.5) create semi-transparent overlays that allow original image details to show through, while higher values (e.g., 0.7-1.0) create more opaque fills that better obscure background details. Typical values range from 0.4 to 0.7 for balanced visualization where both the mask and underlying image are visible.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Mask Visualization` in version `v1`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Template Matching`](template_matching.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Motion Detection`](motion_detection.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Detection Offset`](detection_offset.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Identify Outliers`](identify_outliers.md), [`SIFT`](sift.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SIFT Comparison`](sift_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Combine`](detections_combine.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Buffer`](buffer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Relative Static Crop`](relative_static_crop.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`LMM`](lmm.md), [`Corner Visualization`](corner_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Mask Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`semantic_segmentation_prediction`](../kinds/semantic_segmentation_prediction.md)]*): Segmentation predictions containing masks for detected objects. The block uses segmentation masks to create colored fills that precisely follow object or class boundaries. Requires segmentation model outputs with mask data, which may be RLE-encoded..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the mask overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls the transparency of the colored mask fill. Lower values (e.g., 0.3-0.5) create semi-transparent overlays that allow original image details to show through, while higher values (e.g., 0.7-1.0) create more opaque fills that better obscure background details. Typical values range from 0.4 to 0.7 for balanced visualization where both the mask and underlying image are visible..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Mask Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/mask_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.instance_segmentation_model.predictions",
	    "color_palette": "DEFAULT",
	    "palette_size": 10,
	    "custom_colors": [
	        "#FF0000",
	        "#00FF00",
	        "#0000FF"
	    ],
	    "color_axis": "CLASS",
	    "opacity": 0.5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

