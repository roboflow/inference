
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

    - inputs: [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Combine`](detections_combine.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Velocity`](velocity.md), [`Detection Event Log`](detection_event_log.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`CSV Formatter`](csv_formatter.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Identify Outliers`](identify_outliers.md), [`JSON Parser`](json_parser.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Distance Measurement`](distance_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)
    - outputs: [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Barcode Detection`](barcode_detection.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Preprocessing`](image_preprocessing.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dominant Color`](dominant_color.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Mask Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`semantic_segmentation_prediction`](../kinds/semantic_segmentation_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Segmentation predictions containing masks for detected objects. The block uses segmentation masks to create colored fills that precisely follow object or class boundaries. Requires segmentation model outputs with mask data, which may be RLE-encoded..
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

