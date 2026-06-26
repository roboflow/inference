
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

    - inputs: [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Camera Focus`](camera_focus.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Line Counter`](line_counter.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Distance Measurement`](distance_measurement.md), [`Pixel Color Count`](pixel_color_count.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`OCR Model`](ocr_model.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

    - inputs: [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Camera Focus`](camera_focus.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Line Counter`](line_counter.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Distance Measurement`](distance_measurement.md), [`Pixel Color Count`](pixel_color_count.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`OCR Model`](ocr_model.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

