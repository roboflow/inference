
# Morphological Transformation



## v2

??? "Class: `MorphologicalTransformationBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/morphological_transformation/v2.py">inference.core.workflows.core_steps.classical_cv.morphological_transformation.v2.MorphologicalTransformationBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Apply morphological transformations to color images using configurable operations to modify object shapes, remove noise, fill holes, detect edges, and extract features. Color images are processed with transformations applied to all channels, and the output is returned as color for compatibility with downstream processing blocks.

## How This Block Works

This block applies morphological transformations directly to color images (BGR or BGRA). The block:

1. Receives an input image to transform (works with color or grayscale images)
2. Ensures image is in color format (if grayscale, converts to BGR for consistent color output)
3. Creates a square structuring element (kernel) of specified kernel_size
4. Applies the selected morphological operation to all channels simultaneously
5. Preserves the color format throughout the transformation (including alpha channel for BGRA images)
6. Returns the morphologically transformed color image

Supported operations:

- **Erosion**: Shrinks objects by removing boundary pixels (separates objects, removes noise)
- **Dilation**: Expands objects by adding boundary pixels (connects objects, fills holes)
- **Opening**: Erosion followed by dilation (removes noise, separates objects)
- **Closing**: Dilation followed by erosion (fills holes, connects fragments)
- **Opening then Closing**: Opens then closes (specialized preprocessing for edge detection and refinement)
- **Gradient**: Dilation minus erosion (highlights object boundaries/edges)
- **Top Hat**: Input minus opening (detects small bright details)
- **Black Hat**: Closing minus input (detects small dark details)

All operations preserve the color format, making output compatible with downstream color-based blocks like Mask Edge Snap.

## Common Use Cases

- **Color Image Preprocessing**: Process color images while preserving color information for downstream blocks
- **Edge Detection and Refinement**: Use opening+closing for specialized preprocessing before edge snapping
- **Noise Removal**: Remove noise from color images while maintaining color channels
- **Feature Extraction**: Extract morphological features from color images
- **Image Conditioning**: Prepare color images for color-based downstream processing

## Connecting to Other Blocks

- **Before Mask Edge Snap**: Output color images for edge snapping and mask refinement
- **Before visualization blocks**: Display morphologically processed color images
- **In color image pipelines**: Process color images while maintaining color information


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/morphological_transformation@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `kernel_size` | `int` | Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. Typical values range from 3-15. Adjust based on object sizes and desired transformation scale.. | ✅ |
| `operation` | `str` | Type of morphological operation to apply. 'Opening then Closing' is specifically designed for preprocessing before edge detection and mask refinement (e.g., with Mask Edge Snap block). All other operations follow standard morphological definitions.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Morphological Transformation` in version `v2`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`OCR Model`](ocr_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Dominant Color`](dominant_color.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Buffer`](buffer.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Morphological Transformation` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input color image to apply morphological transformation to. Works with BGR, BGRA, or grayscale images. Grayscale images are automatically converted to BGR for consistent color output. Transformations are applied to all channels. Output is always in color format (BGR) for compatibility with downstream blocks like Mask Edge Snap..
        - `kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. Typical values range from 3-15. Adjust based on object sizes and desired transformation scale..
        - `operation` (*[`string`](../kinds/string.md)*): Type of morphological operation to apply. 'Opening then Closing' is specifically designed for preprocessing before edge detection and mask refinement (e.g., with Mask Edge Snap block). All other operations follow standard morphological definitions..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Morphological Transformation` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/morphological_transformation@v2",
	    "image": "$inputs.image",
	    "kernel_size": "5",
	    "operation": "Closing"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `MorphologicalTransformationBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/morphological_transformation/v1.py">inference.core.workflows.core_steps.classical_cv.morphological_transformation.v1.MorphologicalTransformationBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Apply morphological transformations to images using configurable operations (erosion, dilation, opening, closing, gradient, top hat, black hat) to modify object shapes, remove noise, fill holes, detect edges, and extract features from binary or grayscale images for image processing, noise removal, and feature extraction workflows.

## How This Block Works

This block applies morphological transformations to images using structuring elements (kernels) to modify object shapes and extract features. The block:

1. Receives an input image to transform (color images are automatically converted to grayscale)
2. Converts the image to grayscale if it's in color format (morphological operations work on single-channel images)
3. Creates a square structuring element (kernel) of specified kernel_size:
   - Creates a square matrix of ones with dimensions (kernel_size, kernel_size)
   - The kernel defines the neighborhood used for morphological operations
   - Larger kernels affect larger regions, smaller kernels affect smaller regions
4. Applies the selected morphological operation:

   **For Erosion:**
   - Shrinks objects by removing pixels from object boundaries
   - Replaces each pixel with the minimum value in its kernel neighborhood
   - Thins objects and removes small protrusions
   - Useful for separating touching objects and removing small noise

   **For Dilation:**
   - Expands objects by adding pixels to object boundaries
   - Replaces each pixel with the maximum value in its kernel neighborhood
   - Thickens objects and fills small holes
   - Useful for connecting nearby objects and filling gaps

   **For Opening (Erosion followed by Dilation):**
   - First erodes then dilates the image
   - Removes small noise and separates touching objects
   - Preserves larger objects while eliminating small details
   - Useful for noise removal and object separation

   **For Closing (Dilation followed by Erosion):**
   - First dilates then erodes the image
   - Fills small holes and connects nearby objects
   - Preserves object shape while closing gaps
   - Useful for filling holes and connecting fragmented objects

   **For Gradient (Dilation minus Erosion):**
   - Computes the difference between dilated and eroded images
   - Highlights object boundaries and edges
   - Creates an outline effect showing object perimeters
   - Useful for edge detection and boundary extraction

   **For Top Hat (Input minus Opening):**
   - Computes the difference between original image and its opening
   - Highlights small bright details that were removed by opening
   - Emphasizes bright features smaller than the kernel
   - Useful for detecting small bright objects or details

   **For Black Hat (Closing minus Input):**
   - Computes the difference between closing and original image
   - Highlights small dark details that were filled by closing
   - Emphasizes dark features smaller than the kernel
   - Useful for detecting small dark objects or holes

5. Preserves the image structure and metadata
6. Returns the morphologically transformed image

Morphological operations use structuring elements (kernels) to probe and modify image structures. The kernel_size controls the scale of the transformation - larger kernels affect larger regions and have more pronounced effects. Basic operations (erosion, dilation) modify object sizes, composite operations (opening, closing) combine effects for noise removal and gap filling, and derived operations (gradient, top hat, black hat) extract specific features. The operations work best on binary or high-contrast grayscale images where object boundaries are clearly defined.

## Common Use Cases

- **Noise Removal**: Remove small noise and artifacts from binary or grayscale images (e.g., remove salt-and-pepper noise, eliminate small artifacts, clean up thresholded images), enabling noise removal workflows
- **Object Separation**: Separate touching or overlapping objects in images (e.g., separate touching objects, split connected regions, isolate individual objects), enabling object separation workflows
- **Hole Filling**: Fill small holes and gaps in objects (e.g., fill holes in objects, close gaps in shapes, complete fragmented objects), enabling hole filling workflows
- **Edge Detection**: Detect object boundaries and edges using morphological gradient (e.g., find object edges, extract boundaries, detect object outlines), enabling morphological edge detection workflows
- **Feature Extraction**: Extract specific features like small bright or dark objects (e.g., detect small bright details with top hat, find small dark objects with black hat, extract specific morphological features), enabling feature extraction workflows
- **Image Preprocessing**: Prepare binary or grayscale images for further processing (e.g., clean up thresholded images, prepare images for contour detection, preprocess images for analysis), enabling morphological preprocessing workflows

## Connecting to Other Blocks

This block receives an image and produces a morphologically transformed image:

- **After image thresholding blocks** to clean up thresholded binary images (e.g., remove noise from thresholded images, fill holes in binary images, separate objects in thresholded images), enabling thresholding-to-morphology workflows
- **After preprocessing blocks** to apply morphological operations after other preprocessing (e.g., apply morphology after filtering, clean up after thresholding, process after enhancement), enabling multi-stage preprocessing workflows
- **Before contour detection blocks** to prepare images for contour detection (e.g., clean up images before contour detection, fill holes before finding contours, separate objects before contour analysis), enabling morphology-to-contour workflows
- **Before analysis blocks** that process binary or grayscale images (e.g., analyze cleaned images, process separated objects, work with morphologically processed images), enabling morphological analysis workflows
- **Before visualization blocks** to display morphologically transformed images (e.g., visualize cleaned images, display processed results, show transformation effects), enabling morphological visualization workflows
- **In image processing pipelines** where morphological operations are part of a larger processing chain (e.g., clean images in pipelines, apply morphology in multi-stage workflows, process images in transformation chains), enabling morphological processing pipeline workflows

## Requirements

This block works on single-channel (grayscale) images. Color images are automatically converted to grayscale before processing. Morphological operations work best on binary or high-contrast grayscale images where object boundaries are clearly defined. For optimal results, use thresholded binary images or high-contrast grayscale images as input.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/morphological_transformation@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `kernel_size` | `int` | Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. The kernel is a square matrix of ones with dimensions (kernel_size, kernel_size). Larger values create larger kernels that affect larger regions and produce more pronounced effects. Smaller values create smaller kernels that affect smaller regions with subtler effects. Typical values range from 3-15: smaller values (3-5) for fine operations and small objects, medium values (5-9) for general use, larger values (11-15) for coarse operations and large objects. Default is 5, which provides a good balance. Adjust based on object sizes and desired transformation scale.. | ✅ |
| `operation` | `str` | Type of morphological operation to apply: 'Erosion' shrinks objects by removing boundary pixels (separates objects, removes noise), 'Dilation' expands objects by adding boundary pixels (connects objects, fills holes), 'Opening' (erosion then dilation) removes noise and separates objects, 'Closing' (default, dilation then erosion) fills holes and connects objects, 'Gradient' (dilation minus erosion) finds object boundaries/edges, 'Top Hat' (input minus opening) detects small bright details, or 'Black Hat' (closing minus input) detects small dark details. Default is 'Closing' which is commonly used for hole filling. Choose based on goals: erosion/dilation for size modification, opening/closing for noise removal and gap filling, gradient for edge detection, top hat/black hat for feature extraction.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Morphological Transformation` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`OCR Model`](ocr_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Dominant Color`](dominant_color.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Buffer`](buffer.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Morphological Transformation` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to apply morphological transformation to. Color images are automatically converted to grayscale before processing (morphological operations work on single-channel images). Works best on binary or high-contrast grayscale images where object boundaries are clearly defined. Thresholded binary images produce optimal results. The morphologically transformed image will have modified object shapes, removed noise, filled holes, or extracted features depending on the selected operation. Original image metadata is preserved in the output..
        - `kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the square structuring element (kernel) used for morphological operations. Must be a positive integer. The kernel is a square matrix of ones with dimensions (kernel_size, kernel_size). Larger values create larger kernels that affect larger regions and produce more pronounced effects. Smaller values create smaller kernels that affect smaller regions with subtler effects. Typical values range from 3-15: smaller values (3-5) for fine operations and small objects, medium values (5-9) for general use, larger values (11-15) for coarse operations and large objects. Default is 5, which provides a good balance. Adjust based on object sizes and desired transformation scale..
        - `operation` (*[`string`](../kinds/string.md)*): Type of morphological operation to apply: 'Erosion' shrinks objects by removing boundary pixels (separates objects, removes noise), 'Dilation' expands objects by adding boundary pixels (connects objects, fills holes), 'Opening' (erosion then dilation) removes noise and separates objects, 'Closing' (default, dilation then erosion) fills holes and connects objects, 'Gradient' (dilation minus erosion) finds object boundaries/edges, 'Top Hat' (input minus opening) detects small bright details, or 'Black Hat' (closing minus input) detects small dark details. Default is 'Closing' which is commonly used for hole filling. Choose based on goals: erosion/dilation for size modification, opening/closing for noise removal and gap filling, gradient for edge detection, top hat/black hat for feature extraction..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Morphological Transformation` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/morphological_transformation@v1",
	    "image": "$inputs.image",
	    "kernel_size": "5",
	    "operation": "Closing"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

