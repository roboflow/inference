
# Image Slicer



## v2

??? "Class: `ImageSlicerBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/image_slicer/v2.py">inference.core.workflows.core_steps.transformations.image_slicer.v2.ImageSlicerBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Split input images into overlapping tiles or slices using the Slicing Adaptive Inference (SAHI) technique to enable small object detection by processing smaller image regions where objects appear larger relative to the image size, improving detection accuracy for small objects in large images through tiled inference workflows with equal-sized slices and deduplication.

## How This Block Works

This block implements the first step of the SAHI (Slicing Adaptive Inference) technique by dividing large images into smaller overlapping tiles. This approach helps detect small objects that might be missed when processing the entire image at once. The block:

1. Receives an input image and slicing configuration:
   - Takes an input image to be sliced
   - Receives slice dimensions (width and height in pixels)
   - Receives overlap ratios for width and height (controls overlap between adjacent slices)
2. Calculates slice positions:
   - Generates a grid of slice coordinates across the image
   - Positions slices with specified overlap between consecutive slices
   - Overlap helps ensure objects at slice boundaries are not missed
   - Adjusts border slice positions to ensure all slices are equal size (pushes border slices toward image center)
3. Creates image slices:
   - Extracts each slice from the original image using calculated coordinates
   - Creates WorkflowImageData objects for each slice with crop metadata
   - Stores offset information (x, y coordinates) for each slice relative to original image
   - Maintains parent image reference for coordinate mapping
4. Deduplicates slices:
   - Removes any duplicate slice coordinates that may occur from overlap calculations
   - Ensures each unique slice position appears only once in the output
   - Prevents redundant processing of identical image regions
5. Handles edge cases:
   - Filters out empty slices (if any occur)
   - Ensures all slices fit within image boundaries
   - Creates crop identifiers for tracking each slice
6. Returns list of slices:
   - Outputs all unique slices as a list of images
   - All slices have equal dimensions (border slices adjusted to match)
   - Increases dimensionality by 1 (one image becomes multiple slices)
   - Each slice can be processed independently by downstream blocks

The SAHI technique works by making small objects appear larger relative to the slice size. When an object is only a few pixels in a large image, scaling the image down to model input size makes the object too small to detect. By slicing the image and processing each slice separately, the same object occupies more pixels in each slice, making detection more reliable. Overlapping slices ensure objects near slice boundaries are detected in at least one slice.

## Common Use Cases

- **Small Object Detection**: Detect small objects in large images using SAHI technique (e.g., detect small vehicles in aerial images, find license plates in wide-angle camera views, detect insects in high-resolution photos), enabling small object detection workflows
- **High-Resolution Image Processing**: Process high-resolution images by slicing them into manageable pieces (e.g., process satellite imagery, analyze medical imaging scans, process large document images), enabling high-resolution processing workflows
- **Aerial and Drone Imagery**: Detect objects in aerial photography where objects are small relative to image size (e.g., detect vehicles in drone footage, find people in aerial surveillance, detect structures in satellite images), enabling aerial detection workflows
- **Wide-Angle Camera Monitoring**: Improve detection in wide-angle camera views where objects appear small (e.g., monitor large parking lots, detect objects in panoramic views, analyze traffic in wide camera coverage), enabling wide-angle monitoring workflows
- **Medical Imaging Analysis**: Analyze medical images by processing regions separately (e.g., detect lesions in large scans, find anomalies in medical images, analyze radiology images), enabling medical imaging workflows
- **Document and Text Processing**: Process large documents by slicing into regions (e.g., OCR large documents, detect text regions in scanned documents, analyze document layouts), enabling document processing workflows

## Connecting to Other Blocks

This block receives images and produces image slices:

- **After image input or preprocessing blocks** to slice images for SAHI processing (e.g., slice input images, process preprocessed images, slice transformed images), enabling image-to-slice workflows
- **Before detection model blocks** (Object Detection Model, Instance Segmentation Model) to process slices for small object detection (e.g., detect objects in slices, run detection on each slice, process slices with models), enabling slice-to-detection workflows
- **Before Detections Stitch block** (required after detection models) to merge detections from slices back to original image coordinates (e.g., merge slice detections, combine detection results, reconstruct full-image predictions), enabling slice-detection-stitch workflows
- **In SAHI workflows** following the pattern: Image Slicer → Detection Model → Detections Stitch to implement complete SAHI technique for small object detection
- **Before filtering or analytics blocks** to process slice-level results before stitching (e.g., filter detections per slice, analyze slice results, process slice outputs), enabling slice-to-analysis workflows
- **As part of multi-stage detection pipelines** where slices are processed independently and results are combined (e.g., multi-scale detection, hierarchical detection, parallel slice processing), enabling multi-stage detection workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Equal-Sized Slices**: All slices generated by the slicer have equal dimensions. Border slices that would normally be smaller in v1 are adjusted by pushing them toward the image center, ensuring consistent slice sizes. This provides more predictable processing behavior and ensures all slices are processed with the same dimensions, which can be important for model inference consistency.
- **Deduplication**: Duplicate slice coordinates are automatically removed, ensuring each unique slice position appears only once in the output. This prevents redundant processing of identical image regions that could occur due to overlap calculations, improving efficiency and preventing duplicate detections.

## Requirements

This block requires an input image. The slice dimensions (width and height) should ideally match the model's input size for optimal performance. If slice size differs from model input size, slices will be resized during inference which may affect accuracy. Default slice size is 640x640 pixels, but this should be adjusted based on your model's input size (e.g., use 320x320 for models with 320 input size, 1280x1280 for models with 1280 input size). Overlap ratios (default 0.2 or 20%) help ensure objects at slice boundaries are detected, but higher overlap increases processing time. The block should be used with object detection or instance segmentation models, followed by Detections Stitch block to merge results. For more information on SAHI technique, see: https://ieeexplore.ieee.org/document/9897990. For a practical guide, visit: https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/image_slicer@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `slice_width` | `int` | Width of each slice in pixels. Should ideally match your detection model's input width for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal width (border slices adjusted to match).. | ✅ |
| `slice_height` | `int` | Height of each slice in pixels. Should ideally match your detection model's input height for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal height (border slices adjusted to match).. | ✅ |
| `overlap_ratio_width` | `float` | Overlap ratio between consecutive slices in the width (horizontal) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice width overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed.. | ✅ |
| `overlap_ratio_height` | `float` | Overlap ratio between consecutive slices in the height (vertical) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice height overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Slicer` in version `v2`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`Stitch Images`](stitch_images.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Frame Delay`](frame_delay.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Image Slicer`](image_slicer.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Line Counter`](line_counter.md), [`Identify Outliers`](identify_outliers.md), [`Identify Changes`](identify_changes.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter`](line_counter.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Slicer` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to be sliced into smaller tiles. The image will be divided into overlapping slices based on the slice dimensions and overlap ratios. Each slice maintains metadata about its position in the original image for coordinate mapping. All slices will have equal dimensions (border slices are adjusted to match). Used in SAHI (Slicing Adaptive Inference) workflows to enable small object detection by processing image regions separately..
        - `slice_width` (*[`integer`](../kinds/integer.md)*): Width of each slice in pixels. Should ideally match your detection model's input width for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal width (border slices adjusted to match)..
        - `slice_height` (*[`integer`](../kinds/integer.md)*): Height of each slice in pixels. Should ideally match your detection model's input height for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time. All slices will have equal height (border slices adjusted to match)..
        - `overlap_ratio_width` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Overlap ratio between consecutive slices in the width (horizontal) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice width overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed..
        - `overlap_ratio_height` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Overlap ratio between consecutive slices in the height (vertical) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice height overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection. Duplicate slices created by overlap are automatically removed..

    - output
    
        - `slices` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Slicer` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/image_slicer@v2",
	    "image": "$inputs.image",
	    "slice_width": 320,
	    "slice_height": 320,
	    "overlap_ratio_width": 0.1,
	    "overlap_ratio_height": 0.1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `ImageSlicerBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/image_slicer/v1.py">inference.core.workflows.core_steps.transformations.image_slicer.v1.ImageSlicerBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Split input images into overlapping tiles or slices using the Slicing Adaptive Inference (SAHI) technique to enable small object detection by processing smaller image regions where objects appear larger relative to the image size, improving detection accuracy for small objects in large images through tiled inference workflows.

## How This Block Works

This block implements the first step of the SAHI (Slicing Adaptive Inference) technique by dividing large images into smaller overlapping tiles. This approach helps detect small objects that might be missed when processing the entire image at once. The block:

1. Receives an input image and slicing configuration:
   - Takes an input image to be sliced
   - Receives slice dimensions (width and height in pixels)
   - Receives overlap ratios for width and height (controls overlap between adjacent slices)
2. Calculates slice positions:
   - Generates a grid of slice coordinates across the image
   - Positions slices with specified overlap between consecutive slices
   - Overlap helps ensure objects at slice boundaries are not missed
   - Border slices may be smaller than specified size to fit within image bounds
3. Creates image slices:
   - Extracts each slice from the original image using calculated coordinates
   - Creates WorkflowImageData objects for each slice with crop metadata
   - Stores offset information (x, y coordinates) for each slice relative to original image
   - Maintains parent image reference for coordinate mapping
4. Handles edge cases:
   - Filters out empty slices (if any occur)
   - Ensures all slices fit within image boundaries
   - Creates crop identifiers for tracking each slice
5. Returns list of slices:
   - Outputs all slices as a list of images
   - Increases dimensionality by 1 (one image becomes multiple slices)
   - Each slice can be processed independently by downstream blocks

The SAHI technique works by making small objects appear larger relative to the slice size. When an object is only a few pixels in a large image, scaling the image down to model input size makes the object too small to detect. By slicing the image and processing each slice separately, the same object occupies more pixels in each slice, making detection more reliable. Overlapping slices ensure objects near slice boundaries are detected in at least one slice.

## Common Use Cases

- **Small Object Detection**: Detect small objects in large images using SAHI technique (e.g., detect small vehicles in aerial images, find license plates in wide-angle camera views, detect insects in high-resolution photos), enabling small object detection workflows
- **High-Resolution Image Processing**: Process high-resolution images by slicing them into manageable pieces (e.g., process satellite imagery, analyze medical imaging scans, process large document images), enabling high-resolution processing workflows
- **Aerial and Drone Imagery**: Detect objects in aerial photography where objects are small relative to image size (e.g., detect vehicles in drone footage, find people in aerial surveillance, detect structures in satellite images), enabling aerial detection workflows
- **Wide-Angle Camera Monitoring**: Improve detection in wide-angle camera views where objects appear small (e.g., monitor large parking lots, detect objects in panoramic views, analyze traffic in wide camera coverage), enabling wide-angle monitoring workflows
- **Medical Imaging Analysis**: Analyze medical images by processing regions separately (e.g., detect lesions in large scans, find anomalies in medical images, analyze radiology images), enabling medical imaging workflows
- **Document and Text Processing**: Process large documents by slicing into regions (e.g., OCR large documents, detect text regions in scanned documents, analyze document layouts), enabling document processing workflows

## Connecting to Other Blocks

This block receives images and produces image slices:

- **After image input or preprocessing blocks** to slice images for SAHI processing (e.g., slice input images, process preprocessed images, slice transformed images), enabling image-to-slice workflows
- **Before detection model blocks** (Object Detection Model, Instance Segmentation Model) to process slices for small object detection (e.g., detect objects in slices, run detection on each slice, process slices with models), enabling slice-to-detection workflows
- **Before Detections Stitch block** (required after detection models) to merge detections from slices back to original image coordinates (e.g., merge slice detections, combine detection results, reconstruct full-image predictions), enabling slice-detection-stitch workflows
- **In SAHI workflows** following the pattern: Image Slicer → Detection Model → Detections Stitch to implement complete SAHI technique for small object detection
- **Before filtering or analytics blocks** to process slice-level results before stitching (e.g., filter detections per slice, analyze slice results, process slice outputs), enabling slice-to-analysis workflows
- **As part of multi-stage detection pipelines** where slices are processed independently and results are combined (e.g., multi-scale detection, hierarchical detection, parallel slice processing), enabling multi-stage detection workflows

## Requirements

This block requires an input image. The slice dimensions (width and height) should ideally match the model's input size for optimal performance. If slice size differs from model input size, slices will be resized during inference which may affect accuracy. Default slice size is 640x640 pixels, but this should be adjusted based on your model's input size (e.g., use 320x320 for models with 320 input size, 1280x1280 for models with 1280 input size). Overlap ratios (default 0.2 or 20%) help ensure objects at slice boundaries are detected, but higher overlap increases processing time. The block should be used with object detection or instance segmentation models, followed by Detections Stitch block to merge results. For more information on SAHI technique, see: https://ieeexplore.ieee.org/document/9897990. For a practical guide, visit: https://blog.roboflow.com/how-to-use-sahi-to-detect-small-objects/.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/image_slicer@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `slice_width` | `int` | Width of each slice in pixels. Should ideally match your detection model's input width for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time.. | ✅ |
| `slice_height` | `int` | Height of each slice in pixels. Should ideally match your detection model's input height for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time.. | ✅ |
| `overlap_ratio_width` | `float` | Overlap ratio between consecutive slices in the width (horizontal) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice width overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection.. | ✅ |
| `overlap_ratio_height` | `float` | Overlap ratio between consecutive slices in the height (vertical) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice height overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Slicer` in version `v1`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`Stitch Images`](stitch_images.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Frame Delay`](frame_delay.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Image Slicer`](image_slicer.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Line Counter`](line_counter.md), [`Identify Outliers`](identify_outliers.md), [`Identify Changes`](identify_changes.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter`](line_counter.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Slicer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to be sliced into smaller tiles. The image will be divided into overlapping slices based on the slice dimensions and overlap ratios. Each slice maintains metadata about its position in the original image for coordinate mapping. Used in SAHI (Slicing Adaptive Inference) workflows to enable small object detection by processing image regions separately..
        - `slice_width` (*[`integer`](../kinds/integer.md)*): Width of each slice in pixels. Should ideally match your detection model's input width for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time..
        - `slice_height` (*[`integer`](../kinds/integer.md)*): Height of each slice in pixels. Should ideally match your detection model's input height for optimal performance. If different, slices will be resized during model inference which may affect accuracy. Common values: 320 (for models with 320px input), 640 (default, for most YOLO models), 1280 (for high-resolution models). Larger slices process fewer total slices but may miss very small objects. Smaller slices detect smaller objects but increase processing time..
        - `overlap_ratio_width` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Overlap ratio between consecutive slices in the width (horizontal) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice width overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection..
        - `overlap_ratio_height` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Overlap ratio between consecutive slices in the height (vertical) dimension. Range: 0.0 to <1.0. Specifies what fraction of the slice height overlaps with adjacent slices. Default 0.2 means 20% overlap. Higher overlap (e.g., 0.3-0.5) ensures objects at slice boundaries are more likely to be detected but increases processing time since more slices are created. Lower overlap (e.g., 0.1) is faster but may miss objects at boundaries. Typical values: 0.1-0.3 for most use cases, 0.3-0.5 for critical small object detection..

    - output
    
        - `slices` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Slicer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/image_slicer@v1",
	    "image": "$inputs.image",
	    "slice_width": 320,
	    "slice_height": 320,
	    "overlap_ratio_width": 0.1,
	    "overlap_ratio_height": 0.1
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

