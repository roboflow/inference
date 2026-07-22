
# Image Contours



??? "Class: `ImageContoursDetectionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/contours/v1.py">inference.core.workflows.core_steps.classical_cv.contours.v1.ImageContoursDetectionBlockV1</a>
    



Detect and extract contours (boundaries of shapes) from a thresholded binary or grayscale image using OpenCV's contour detection, drawing the detected contours on the image, and returning contour data including coordinates, hierarchy information, and count for shape analysis, object boundary detection, and contour-based image processing workflows.

## How This Block Works

This block detects contours (connected boundaries of shapes) in an image and draws them for visualization. The block:

1. Receives an input image that should be thresholded (binary or grayscale) for best results
2. Converts the image to grayscale if it's in color (handles BGR color images by converting to grayscale)
3. Detects contours using OpenCV's findContours function:
   - Uses RETR_EXTERNAL retrieval mode to find only external contours (outer boundaries of shapes)
   - Uses CHAIN_APPROX_SIMPLE approximation method to compress contour points (reduces redundant points)
   - Detects all connected boundary points that form closed or open contours
   - Returns contours as arrays of points and hierarchy information describing contour relationships
4. Draws detected contours on the image:
   - Converts the grayscale image back to BGR color format for visualization
   - Draws all contours on the image using a configurable line thickness
   - Uses purple color (255, 0, 255 in BGR) by default for contour lines
   - Draws contours directly on the image for visual inspection
5. Counts the total number of contours detected in the image
6. Returns the image with contours drawn, the contours data (point arrays), hierarchy information, and the contour count

The block expects a thresholded (binary) image where objects are white and background is black (or vice versa) for optimal contour detection. Contours are detected as the boundaries between different pixel intensity regions. The RETR_EXTERNAL mode focuses on outer boundaries, ignoring internal holes, which is useful for detecting separate objects. The CHAIN_APPROX_SIMPLE method simplifies contours by removing redundant points along straight lines, making the contour data more compact while preserving essential shape information.

## Common Use Cases

- **Shape Detection and Analysis**: Detect and analyze shapes in images by finding their boundaries (e.g., detect object boundaries for shape analysis, identify geometric shapes, extract shape outlines for measurement), enabling shape-based image analysis workflows
- **Object Boundary Extraction**: Extract object boundaries and outlines from thresholded images (e.g., extract object boundaries for further processing, identify object edges, detect object outlines in binary images), enabling boundary extraction workflows
- **Image Segmentation Analysis**: Analyze segmentation results by detecting contour boundaries (e.g., find contours from segmentation masks, analyze segmented regions, extract boundaries from segmented objects), enabling segmentation analysis workflows
- **Quality Control and Inspection**: Use contour detection for quality control and inspection tasks (e.g., detect defects by finding unexpected contours, verify object shapes, inspect object boundaries), enabling contour-based quality control workflows
- **Object Counting**: Count objects in images by detecting their contours (e.g., count objects by detecting contours, enumerate objects based on boundaries, quantify items using contour detection), enabling contour-based object counting workflows
- **Measurement and Analysis**: Use contours for measurements and geometric analysis (e.g., measure object perimeters using contours, analyze object shapes, calculate geometric properties from contours), enabling contour-based measurement workflows

## Connecting to Other Blocks

This block receives a thresholded image and produces contour data and visualizations:

- **After image thresholding blocks** to detect contours in thresholded binary images (e.g., find contours after thresholding, detect shapes in binary images, extract boundaries from thresholded images), enabling thresholding-to-contour workflows
- **After image preprocessing blocks** that prepare images for contour detection (e.g., detect contours after preprocessing, find shapes after filtering, extract boundaries after enhancement), enabling preprocessed contour detection workflows
- **After segmentation blocks** to extract contours from segmentation results (e.g., find contours from segmentation masks, detect boundaries of segmented regions, extract shape outlines from segments), enabling segmentation-to-contour workflows
- **Before visualization blocks** to display contour visualizations (e.g., visualize detected contours, display shape boundaries, show contour analysis results), enabling contour visualization workflows
- **Before analysis blocks** that process contour data (e.g., analyze contour shapes, process contour coordinates, measure contour properties), enabling contour analysis workflows
- **Before filtering or logic blocks** that use contour count or properties for decision-making (e.g., filter based on contour count, make decisions based on detected shapes, apply logic based on contour properties), enabling contour-based conditional workflows

## Requirements

The input image should be thresholded (converted to binary/grayscale) before using this block. Thresholded images have distinct foreground (white) and background (black) regions, which makes contour detection more reliable. Use thresholding blocks (e.g., Image Threshold) or segmentation blocks to prepare images before contour detection.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/contours_detection@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `line_thickness` | `int` | Thickness of the lines used to draw contours on the output image. Must be a positive integer. Thicker lines (e.g., 5-10) make contours more visible but may obscure fine details. Thinner lines (e.g., 1-2) show more detail but may be harder to see. Default is 3, which provides good visibility. Adjust based on image size and desired visibility. Use thicker lines for large images or when contours need to be highly visible, thinner lines for detailed analysis or small images.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Contours` in version `v1`.

    - inputs: [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Morphological Transformation`](morphological_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Grid Visualization`](grid_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Image Stack`](image_stack.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Stitch Images`](stitch_images.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`QR Code Generator`](qr_code_generator.md), [`Distance Measurement`](distance_measurement.md), [`Frame Delay`](frame_delay.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter`](line_counter.md), [`Morphological Transformation`](morphological_transformation.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Identify Outliers`](identify_outliers.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Contours` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to detect contours from. Should be thresholded (binary or grayscale) for best results - thresholded images have distinct foreground and background regions that make contour detection more reliable. The image will be converted to grayscale automatically if it's in color format. Contours are detected as boundaries between different pixel intensity regions. Use thresholding blocks (e.g., Image Threshold) or segmentation blocks to prepare images before contour detection. The block detects external contours (outer boundaries) and draws them on the image..
        - `line_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the lines used to draw contours on the output image. Must be a positive integer. Thicker lines (e.g., 5-10) make contours more visible but may obscure fine details. Thinner lines (e.g., 1-2) show more detail but may be harder to see. Default is 3, which provides good visibility. Adjust based on image size and desired visibility. Use thicker lines for large images or when contours need to be highly visible, thinner lines for detailed analysis or small images..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.
        - `contours` ([`contours`](../kinds/contours.md)): List of numpy arrays where each array represents contour points.
        - `hierarchy` ([`numpy_array`](../kinds/numpy_array.md)): Numpy array.
        - `number_contours` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Image Contours` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/contours_detection@v1",
	    "image": "$inputs.image",
	    "line_thickness": 3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

