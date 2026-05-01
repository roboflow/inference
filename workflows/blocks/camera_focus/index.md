
# Camera Focus



## v2

??? "Class: `CameraFocusBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/camera_focus/v2.py">inference.core.workflows.core_steps.classical_cv.camera_focus.v2.CameraFocusBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Calculate focus quality scores using the Tenengrad focus measure (Sobel gradient magnitudes) to assess image sharpness, detect blur, evaluate camera focus performance, enable auto-focus systems, perform image quality assessment, compute per-region focus measures for detected objects, and provide comprehensive visualization overlays including zebra pattern exposure warnings, focus peaking, heads-up display, composition grid, and center markers for professional camera control and image analysis workflows.

## How This Block Works

This block calculates the Tenengrad focus measure, which quantifies image sharpness by measuring gradient magnitudes using Sobel operators. The block:

1. Receives an input image (color or grayscale, automatically converts color to grayscale for processing)
2. Optionally receives detection bounding boxes to compute focus measures within specific regions
3. Converts the image to grayscale if it's in color format (Tenengrad measure works on single-channel images)
4. Calculates horizontal and vertical Sobel gradients:
   - Applies Sobel operator in horizontal direction (gradient X) to detect vertical edges
   - Applies Sobel operator in vertical direction (gradient Y) to detect horizontal edges
   - Uses 3x3 Sobel kernels for gradient computation
   - Computes gradient magnitude using squared values: sqrt(gx² + gy²) approximated as gx² + gy²
5. Calculates the focus measure:
   - Squares the horizontal and vertical gradient components
   - Sums the squared gradients to create a focus measure matrix
   - Higher values indicate stronger edges and finer detail (sharper, more focused regions)
   - Lower values indicate weaker edges and less detail (blurrier, less focused regions)
6. Computes overall focus value:
   - Calculates mean of focus measure matrix across entire image
   - Returns a single numerical focus score for the whole image
7. Computes per-region focus measures (if detections provided):
   - Extracts bounding box coordinates from detection predictions
   - Clips bounding boxes to image boundaries
   - Calculates mean focus measure within each bounding box region
   - Returns a list of focus values, one per detection region
8. Applies optional visualization overlays:
   - **Zebra Pattern Warnings**: Diagonal stripe overlay on under/overexposed regions (blue for underexposed, red for overexposed) to identify exposure issues
   - **Focus Peaking**: Green overlay highlighting in-focus areas (regions above focus threshold) to visualize sharp regions
   - **Heads-Up Display (HUD)**: Semi-transparent overlay showing focus value, brightness histogram (for each color channel and grayscale), and exposure information in top-left corner
   - **Composition Grid**: Overlay grid lines for composition assistance (2x2, 3x3 rule of thirds, 4x4, or 5x5 divisions)
   - **Center Marker**: Crosshair marker at frame center for alignment and framing reference
9. Preserves image structure and metadata
10. Returns the visualization image (if overlays enabled), overall focus measure value, and per-bounding-box focus measures list

The Tenengrad focus measure quantifies image sharpness by analyzing edge strength and gradient magnitudes. In-focus images contain many sharp edges with strong gradients, resulting in high Tenengrad scores. Out-of-focus images have blurred edges with weak gradients, resulting in low Tenengrad scores. The measure uses Sobel operators to compute gradients efficiently and is robust to noise. Higher Tenengrad values indicate better focus, with typical ranges varying based on image content, resolution, and edge density. The visualization overlays provide professional camera control aids, helping identify focus issues, exposure problems, and composition opportunities in real-time or during analysis.

## Common Use Cases

- **Auto-Focus Systems**: Assess focus quality to enable automatic camera focus adjustment with per-region focus analysis (e.g., evaluate focus during auto-focus operations, detect optimal focus position for specific objects, trigger focus adjustments based on Tenengrad scores), enabling advanced auto-focus workflows
- **Image Quality Assessment**: Evaluate image sharpness and detect blurry images with visualization overlays for quality control (e.g., assess image quality in capture pipelines with HUD display, detect out-of-focus images with focus peaking, filter low-quality images using focus thresholds), enabling comprehensive quality assessment workflows
- **Professional Camera Control**: Provide real-time focus and exposure feedback for manual camera operation (e.g., display focus peaking for manual focus, show zebra warnings for exposure adjustment, use composition grid for framing), enabling professional camera control workflows
- **Object-Specific Focus Analysis**: Evaluate focus quality for specific detected objects within images (e.g., assess focus on detected objects, analyze focus per bounding box region, optimize focus for specific object classes), enabling object-focused analysis workflows
- **Camera Calibration**: Evaluate focus performance during camera setup and calibration with comprehensive visualization (e.g., assess focus during camera calibration with overlays, optimize focus settings using HUD feedback, evaluate camera performance with visualization aids), enabling enhanced camera calibration workflows
- **Video Focus Tracking**: Monitor focus quality across video frames with per-object focus measures (e.g., track focus for moving objects, monitor focus quality in video streams, analyze focus consistency across frames), enabling video focus tracking workflows

## Connecting to Other Blocks

This block receives an image (and optionally detections) and produces a visualization image, overall focus_measure float value, and bbox_focus_measures list:

- **After object detection or instance segmentation blocks** to compute focus measures for detected objects (e.g., assess focus on detected objects, analyze focus per detection region, evaluate object-specific focus quality), enabling detection-to-focus workflows
- **After image capture or preprocessing blocks** to assess focus quality of captured or processed images (e.g., evaluate focus after image capture, assess sharpness after preprocessing with visualization, measure focus in image pipelines with overlays), enabling enhanced focus assessment workflows
- **Before logic blocks** like Continue If to make decisions based on focus quality (e.g., continue if focus is good, filter images based on Tenengrad scores, make decisions using focus measures or per-object focus values), enabling focus-based decision workflows
- **Before analysis blocks** to assess image quality before analysis (e.g., evaluate focus before analysis with HUD display, assess sharpness for processing, measure quality before analysis), enabling quality-based analysis workflows
- **In auto-focus systems** where focus measurement is part of a feedback loop with per-object analysis (e.g., measure focus for auto-focus with object prioritization, assess focus in feedback systems, evaluate focus in control loops), enabling advanced auto-focus system workflows
- **Before visualization blocks** to display focus quality information (e.g., visualize focus scores with overlays, display focus measures, show focus quality with professional camera aids), enabling comprehensive focus visualization workflows

## Version Differences

**Enhanced from v1:**

- **Different Focus Algorithm**: Uses Tenengrad focus measure (Sobel gradient magnitudes) instead of Brenner measure, providing more robust edge detection and focus assessment
- **Visualization Overlays**: Includes comprehensive visualization features including zebra pattern exposure warnings, focus peaking (green highlight on sharp areas), heads-up display with focus values and brightness histogram, composition grid overlays (2x2, 3x3, 4x4, 5x5), and center crosshair marker for professional camera control
- **Per-Region Focus Analysis**: Supports optional detection bounding boxes to compute focus measures within specific object regions, enabling object-specific focus assessment
- **Enhanced Outputs**: Returns three outputs - visualization image, overall focus_measure float, and bbox_focus_measures list (per-detection focus values)
- **Configurable Visualization**: All visualization overlays are configurable (zebra warnings, HUD, focus peaking, grid, center marker can be enabled/disabled independently)
- **Exposure Analysis**: Includes exposure assessment with configurable thresholds for underexposed/overexposed regions with visual zebra pattern warnings
- **Professional Camera Aids**: Provides tools similar to professional camera displays including focus peaking, histogram display, and composition guides

## Requirements

This block works on color or grayscale input images. Color images are automatically converted to grayscale before processing (Tenengrad measure works on single-channel images). The block outputs a visualization image (with optional overlays), an overall focus_measure float value, and a bbox_focus_measures list (if detections are provided). Higher Tenengrad values indicate better focus and sharper images, while lower values indicate blur and poor focus. The focus measure is sensitive to image content, resolution, and edge density, so threshold values for "good" focus should be calibrated based on specific use cases and image characteristics. All visualization overlays are optional and can be enabled or disabled independently based on workflow needs.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/camera_focus@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `underexposed_threshold_percent` | `float` | Brightness percentage threshold below which pixels are marked as underexposed. Must be between 0.0 and 100.0. Default is 3.0%, meaning pixels with brightness below 3% (approximately value 8 in 0-255 range) are considered underexposed. Pixels below this threshold will show blue zebra pattern overlay when show_zebra_warnings is enabled. Lower values are stricter (fewer pixels marked as underexposed), higher values are more lenient (more pixels marked as underexposed). Adjust based on exposure tolerance and image requirements.. | ❌ |
| `overexposed_threshold_percent` | `float` | Brightness percentage threshold above which pixels are marked as overexposed. Must be between 0.0 and 100.0. Default is 97.0%, meaning pixels with brightness above 97% (approximately value 247 in 0-255 range) are considered overexposed. Pixels above this threshold will show red zebra pattern overlay when show_zebra_warnings is enabled. Higher values are stricter (fewer pixels marked as overexposed), lower values are more lenient (more pixels marked as overexposed). Adjust based on exposure tolerance and image requirements.. | ❌ |
| `show_zebra_warnings` | `bool` | Display diagonal zebra pattern overlay on under/overexposed regions. When enabled (default True), pixels below underexposed_threshold_percent show blue zebra stripes, and pixels above overexposed_threshold_percent show red zebra stripes. This provides visual feedback for exposure issues similar to professional camera zebra pattern displays. The zebra pattern helps identify regions with exposure problems (too dark or too bright) that may need adjustment. Disable if you don't want exposure warnings or want cleaner visualization.. | ❌ |
| `grid_overlay` | `str` | Composition grid overlay for framing assistance. Options: 'None' (no grid), '2x2' (four quadrants), '3x3' (default, rule of thirds with 9 sections), '4x4' (16 sections), or '5x5' (25 sections). The grid helps with composition and framing by dividing the image into sections. The 3x3 grid (rule of thirds) is commonly used for balanced composition. Grid lines are drawn in gray color. Choose based on composition needs: rule of thirds (3x3) for general use, 2x2 for simple quadrant composition, or higher divisions for more detailed composition guides.. | ❌ |
| `show_hud` | `bool` | Display heads-up display (HUD) overlay with focus scores and brightness histogram. When enabled (default True), shows a semi-transparent black overlay in the top-left corner displaying: focus value (labeled 'TFM Focus' with numerical score), brightness histogram showing distribution for each color channel (red, green, blue) and grayscale, and exposure label. The HUD provides comprehensive focus and exposure information for professional camera control. Disable if you don't need the HUD display or want cleaner visualization.. | ❌ |
| `show_focus_peaking` | `bool` | Display green overlay highlighting in-focus areas (focus peaking). When enabled (default True), regions with focus measures above a threshold (top 30% by default) are highlighted with a semi-transparent green overlay. This helps visualize which areas of the image are in sharp focus, similar to professional camera focus peaking displays. The green highlight makes it easy to see sharp regions at a glance. Disable if you don't want focus peaking overlay or want cleaner visualization.. | ❌ |
| `show_center_marker` | `bool` | Display crosshair marker at the center of the frame. When enabled (default True), shows a white crosshair at the image center for alignment and framing reference. The crosshair size scales with image dimensions for visibility. This helps with composition alignment and center framing, similar to professional camera center markers. Disable if you don't need the center marker or want cleaner visualization.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Camera Focus` in version `v2`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Preprocessing`](image_preprocessing.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Contours`](image_contours.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`SIFT`](sift.md), [`VLM As Detector`](vlm_as_detector.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Consensus`](detections_consensus.md), [`Barcode Detection`](barcode_detection.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Email Notification`](email_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Camera Focus` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image (color or grayscale) to calculate focus quality for. Color images are automatically converted to grayscale before processing (Tenengrad focus measure works on single-channel images). The block calculates the Tenengrad focus measure using Sobel gradient magnitudes to assess image sharpness. The output includes a visualization image (with optional overlays if enabled), an overall focus_measure float value, and a bbox_focus_measures list (per-detection focus values if detections are provided). Higher Tenengrad values indicate better focus and sharper images (stronger edges and gradients), while lower values indicate blur and poor focus (weaker gradients). The focus measure uses Sobel operators to compute gradient magnitudes efficiently. Original image metadata is preserved. Use this block to assess focus quality, detect blur, enable auto-focus systems, perform object-specific focus analysis, or perform image quality assessment with professional camera control visualization aids..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Optional detection predictions (object detection or instance segmentation) to compute focus measures within bounding box regions. When provided, the block calculates a separate focus measure for each detection's bounding box region and returns them in the bbox_focus_measures list output. This enables object-specific focus analysis, allowing you to assess focus quality for individual detected objects rather than just the overall image. Useful for evaluating focus on specific objects of interest, analyzing focus per object class, or optimizing focus for detected regions. Each bbox_focus_measure value corresponds to the mean Tenengrad focus measure within that object's bounding box. Leave as None if you only need overall image focus assessment..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.
        - `focus_measure` ([`float`](../kinds/float.md)): Float value.
        - `bbox_focus_measures` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Camera Focus` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/camera_focus@v2",
	    "image": "$inputs.image",
	    "underexposed_threshold_percent": "<block_does_not_provide_example>",
	    "overexposed_threshold_percent": "<block_does_not_provide_example>",
	    "show_zebra_warnings": "<block_does_not_provide_example>",
	    "grid_overlay": "<block_does_not_provide_example>",
	    "show_hud": "<block_does_not_provide_example>",
	    "show_focus_peaking": "<block_does_not_provide_example>",
	    "show_center_marker": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `CameraFocusBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/camera_focus/v1.py">inference.core.workflows.core_steps.classical_cv.camera_focus.v1.CameraFocusBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Calculate focus quality scores using the Brenner function measure to assess image sharpness, detect blur, evaluate camera focus performance, enable auto-focus systems, perform image quality assessment, and determine optimal focus settings for camera calibration and image capture workflows.

## How This Block Works

This block calculates the Brenner focus measure, which quantifies image sharpness by measuring texture detail at fine scales. The block:

1. Receives an input image (color or grayscale, automatically converts color to grayscale)
2. Converts the image to grayscale if it's in color format (Brenner measure works on single-channel images)
3. Converts the grayscale image to 16-bit integer format for precise calculations
4. Calculates horizontal and vertical intensity differences:
   - Computes horizontal differences by comparing pixels 2 positions apart horizontally: `pixel[x+2] - pixel[x]`
   - Computes vertical differences by comparing pixels 2 positions apart vertically: `pixel[y+2] - pixel[y]`
   - Uses clipping to keep only positive differences (highlights sharp edges and details)
   - Measures rapid intensity changes that indicate fine-scale texture and sharp edges
5. Calculates the focus measure matrix:
   - Takes the maximum of horizontal and vertical differences at each pixel location
   - Squares the differences to emphasize larger variations (stronger response to sharp edges)
   - Creates a matrix where higher values indicate sharper, more focused regions
6. Normalizes and converts to visualization format:
   - Normalizes the focus measure matrix to 0-255 range for display
   - Converts to 8-bit format for visualization
   - Creates a visual representation showing focus quality across the image
7. Overlays the Brenner value text on the image:
   - Displays the mean focus measure value on the top-left of the image
   - Shows focus quality as a numerical score for easy assessment
8. Preserves image structure and metadata
9. Returns the visualization image and the mean focus measure value as a float

The Brenner focus measure quantifies image sharpness by analyzing fine-scale texture and edge detail. In-focus images contain many sharp edges and fine texture details, resulting in large intensity differences between nearby pixels and high Brenner scores. Out-of-focus images have blurred edges and lack fine detail, resulting in small intensity differences and low Brenner scores. The measure uses a 2-pixel spacing to detect fine-scale texture while being robust to noise. Higher Brenner values indicate better focus, with typical ranges varying based on image content and resolution. The visualization shows focus quality distribution across the image, helping identify well-focused and blurred regions.

## Common Use Cases

- **Auto-Focus Systems**: Assess focus quality to enable automatic camera focus adjustment (e.g., evaluate focus during auto-focus operations, detect optimal focus position, trigger focus adjustments based on Brenner scores), enabling auto-focus workflows
- **Image Quality Assessment**: Evaluate image sharpness and detect blurry images for quality control (e.g., assess image quality in capture pipelines, detect out-of-focus images, filter low-quality images), enabling quality assessment workflows
- **Camera Calibration**: Evaluate focus performance during camera setup and calibration (e.g., assess focus during camera calibration, optimize focus settings, evaluate camera performance), enabling camera calibration workflows
- **Blur Detection**: Detect blurry images in image processing pipelines (e.g., identify blurry images for rejection, detect focus issues, assess image sharpness), enabling blur detection workflows
- **Focus Optimization**: Determine optimal focus settings for image capture systems (e.g., find best focus position, optimize focus parameters, evaluate focus across settings), enabling focus optimization workflows
- **Image Analysis**: Assess image sharpness as part of image analysis workflows (e.g., evaluate image quality before processing, assess focus for analysis tasks, measure image sharpness metrics), enabling focus analysis workflows

## Connecting to Other Blocks

This block receives an image and produces a focus measure visualization image and a focus_measure float value:

- **After image capture or preprocessing blocks** to assess focus quality of captured or processed images (e.g., evaluate focus after image capture, assess sharpness after preprocessing, measure focus in image pipelines), enabling focus assessment workflows
- **Before logic blocks** like Continue If to make decisions based on focus quality (e.g., continue if focus is good, filter images based on focus scores, make decisions using focus measures), enabling focus-based decision workflows
- **Before analysis blocks** to assess image quality before analysis (e.g., evaluate focus before analysis, assess sharpness for processing, measure quality before analysis), enabling quality-based analysis workflows
- **In auto-focus systems** where focus measurement is part of a feedback loop (e.g., measure focus for auto-focus, assess focus in feedback systems, evaluate focus in control loops), enabling auto-focus system workflows
- **Before visualization blocks** to display focus quality information (e.g., visualize focus scores, display focus measures, show focus quality), enabling focus visualization workflows
- **In image quality control pipelines** where focus assessment is part of quality checks (e.g., assess focus in quality pipelines, evaluate sharpness in QC workflows, measure focus for quality control), enabling quality control workflows

## Requirements

This block works on color or grayscale input images. Color images are automatically converted to grayscale before processing (Brenner measure works on single-channel images). The block outputs both a visualization image (with focus measure displayed) and a numerical focus_measure value. Higher Brenner values indicate better focus and sharper images, while lower values indicate blur and poor focus. The focus measure is sensitive to image content and resolution, so threshold values for "good" focus should be calibrated based on specific use cases and image characteristics.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/camera_focus@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Camera Focus` in version `v1`.

    - inputs: [`Icon Visualization`](icon_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Image Threshold`](image_threshold.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`Dot Visualization`](dot_visualization.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Barcode Detection`](barcode_detection.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Camera Focus` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image (color or grayscale) to calculate focus quality for. Color images are automatically converted to grayscale before processing (Brenner focus measure works on single-channel images). The block calculates the Brenner function score which measures fine-scale texture and edge detail to assess image sharpness. The output includes both a visualization image (with focus measure value displayed) and a numerical focus_measure float value. Higher Brenner values indicate better focus and sharper images (more fine-scale texture and sharp edges), while lower values indicate blur and poor focus (lacking fine detail). The focus measure uses intensity differences between pixels 2 positions apart to detect fine-scale texture. Original image metadata is preserved. Use this block to assess focus quality, detect blur, enable auto-focus systems, or perform image quality assessment..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.
        - `focus_measure` ([`float`](../kinds/float.md)): Float value.



??? tip "Example JSON definition of step `Camera Focus` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/camera_focus@v1",
	    "image": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

