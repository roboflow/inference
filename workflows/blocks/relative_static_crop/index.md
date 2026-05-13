
# Relative Static Crop



??? "Class: `RelativeStaticCropBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/relative_static_crop/v1.py">inference.core.workflows.core_steps.transformations.relative_static_crop.v1.RelativeStaticCropBlockV1</a>
    



Extract a fixed rectangular region from input images using relative coordinates (normalized 0.0-1.0 values proportional to image dimensions) specified by center point and dimensions, creating consistent proportional crops from the same relative location across images of different sizes for region-of-interest extraction and size-agnostic fixed-area analysis workflows.

## How This Block Works

This block crops a fixed rectangular region from input images using relative coordinates (normalized 0.0-1.0 values), unlike absolute static cropping which uses pixel coordinates. The relative coordinates adapt to different image sizes, making it ideal for extracting the same proportional region from images of varying dimensions. The block:

1. Receives input images and relative coordinate specifications (x_center, y_center, width, height) as values between 0.0 and 1.0
2. Converts relative coordinates to absolute pixel coordinates by multiplying with image dimensions:
   - Converts x_center from relative (0.0-1.0) to absolute pixels: `x_center_pixels = image_width * x_center`
   - Converts y_center from relative (0.0-1.0) to absolute pixels: `y_center_pixels = image_height * y_center`
   - Converts width from relative (0.0-1.0) to absolute pixels: `width_pixels = image_width * width`
   - Converts height from relative (0.0-1.0) to absolute pixels: `height_pixels = image_height * height`
3. Calculates the crop boundaries from the converted center point and dimensions:
   - Computes x_min and y_min by subtracting half the width/height from the center coordinates
   - Computes x_max and y_max by adding the width/height to the minimum coordinates
   - Rounds coordinate values to integer pixel positions
4. Extracts the rectangular region from the image using array slicing (from y_min to y_max, x_min to x_max)
5. Validates that the cropped region has content (returns None if the crop would be empty, such as when coordinates are outside image bounds)
6. Creates a cropped image object with metadata tracking the crop's origin (original image, offset coordinates, unique crop identifier)
7. Preserves video metadata if the input is from video (maintains frame information and temporal context)
8. Returns the cropped image for each input image

The block uses relative coordinates (0.0-1.0), so the same proportional region is extracted from all images regardless of their size. For example, x_center=0.5, y_center=0.5, width=0.4, height=0.4 extracts a 40% by 40% region centered in the image, whether the image is 100x100 pixels or 2000x2000 pixels. This makes the block particularly useful for extracting consistent regions from images of varying sizes (e.g., always cropping the top-right 20% corner, extracting a fixed percentage of the image center, or focusing on a specific proportional area). The center-based coordinate system allows specifying crops by their center point rather than corner coordinates, which can be more intuitive for defining proportional regions.

## Common Use Cases

- **Size-Agnostic Region Extraction**: Extract the same proportional region from images of different sizes for consistent analysis (e.g., crop the top-right 20% corner from images regardless of resolution, extract a fixed percentage of the image center, crop a consistent proportional area for pattern matching), enabling standardized region analysis across images with varying dimensions
- **Multi-Resolution Image Processing**: Extract consistent proportional regions from images with different resolutions (e.g., crop the same relative area from high-resolution and low-resolution images, extract proportional regions from resized images, maintain consistent cropping across different image sizes), enabling size-independent region extraction
- **Proportional Region-of-Interest Focus**: Isolate specific proportional areas of images for detailed processing (e.g., crop a specific relative quadrant of images, extract a fixed percentage region for text recognition, focus on a known proportional area of interest), enabling focused analysis of predetermined proportional regions
- **Multi-Stage Workflow Preparation**: Extract fixed proportional regions for secondary processing steps (e.g., crop a specific relative area from full images, then run OCR or classification on the cropped region), enabling hierarchical workflows with proportional region focus
- **Standardized Crop Generation**: Create consistent proportional crops from images for training or analysis (e.g., extract a fixed relative region from all images for dataset creation, crop a standard proportional area for comparison, generate uniform proportional crops for feature extraction), enabling standardized data preparation workflows across varying image sizes
- **Video Frame Proportional Cropping**: Extract the same proportional region from video frames of different resolutions (e.g., crop a fixed percentage area from each video frame for temporal analysis, extract a consistent proportional monitoring zone for tracking, focus on a specific relative region across frames), enabling temporal analysis of proportional regions across varying frame sizes

## Connecting to Other Blocks

This block receives images and produces cropped images from fixed proportional regions:

- **After image loading blocks** to extract a fixed proportional region of interest before processing, enabling focused analysis of predetermined image areas without processing entire images, particularly useful when working with images of varying sizes
- **Before classification or analysis blocks** that need region-focused inputs (e.g., OCR for text in a fixed proportional area, fine-grained classification for cropped regions, specialized models for specific proportional image areas), enabling optimized processing of consistent proportional regions
- **In video processing workflows** to extract the same proportional region from multiple frames regardless of resolution changes (e.g., crop a fixed percentage area from each video frame for temporal analysis, extract a consistent proportional monitoring zone for tracking, focus on a specific relative region across frames), enabling temporal analysis of proportional regions
- **After detection blocks** where you know the approximate relative location and want to extract a fixed-size proportional region around it (e.g., detect objects in a general relative area, then crop a fixed proportional region around that area for detailed analysis), enabling region-focused multi-stage workflows with size-agnostic cropping
- **Before visualization blocks** that display specific regions (e.g., display only the cropped proportional region, visualize a fixed relative area of interest, show isolated region annotations), enabling focused visualization of extracted proportional regions
- **In batch processing workflows** where the same proportional region needs to be extracted from all images for consistent analysis or comparison, regardless of individual image sizes, enabling standardized proportional region extraction across image sets with varying dimensions


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/relative_statoic_crop@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `x_center` | `float` | X coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the left edge of the image, 1.0 represents the right edge, and 0.5 represents the center horizontally. The crop region is centered at this X coordinate after converting to absolute pixels. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width, where all values are converted to pixels based on image width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size.. | ✅ |
| `y_center` | `float` | Y coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the top edge of the image, 1.0 represents the bottom edge, and 0.5 represents the center vertically. The crop region is centered at this Y coordinate after converting to absolute pixels. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height, where all values are converted to pixels based on image height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size.. | ✅ |
| `width` | `float` | Width of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image width, 0.5 represents 50% of the image width, etc. Defines the horizontal extent of the crop as a proportion of the image width. The crop extends width/2 pixels to the left and right of the x_center coordinate after converting to absolute pixels. Total crop width equals this relative value multiplied by the image width. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries. Relative width adapts to different image sizes - the same relative value extracts the same proportional width from images of any size.. | ✅ |
| `height` | `float` | Height of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image height, 0.5 represents 50% of the image height, etc. Defines the vertical extent of the crop as a proportion of the image height. The crop extends height/2 pixels above and below the y_center coordinate after converting to absolute pixels. Total crop height equals this relative value multiplied by the image height. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries. Relative height adapts to different image sizes - the same relative value extracts the same proportional height from images of any size.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Relative Static Crop` in version `v1`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Color Visualization`](color_visualization.md), [`Label Visualization`](label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT`](sift.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md)
    - outputs: [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Barcode Detection`](barcode_detection.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Preprocessing`](image_preprocessing.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dominant Color`](dominant_color.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Relative Static Crop` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `x_center` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): X coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the left edge of the image, 1.0 represents the right edge, and 0.5 represents the center horizontally. The crop region is centered at this X coordinate after converting to absolute pixels. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width, where all values are converted to pixels based on image width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size..
        - `y_center` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Y coordinate of the center point of the crop region as a relative value (0.0 to 1.0). 0.0 represents the top edge of the image, 1.0 represents the bottom edge, and 0.5 represents the center vertically. The crop region is centered at this Y coordinate after converting to absolute pixels. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height, where all values are converted to pixels based on image height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty. Relative coordinates adapt to different image sizes - the same relative value extracts the same proportional position from images of any size..
        - `width` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Width of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image width, 0.5 represents 50% of the image width, etc. Defines the horizontal extent of the crop as a proportion of the image width. The crop extends width/2 pixels to the left and right of the x_center coordinate after converting to absolute pixels. Total crop width equals this relative value multiplied by the image width. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries. Relative width adapts to different image sizes - the same relative value extracts the same proportional width from images of any size..
        - `height` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Height of the crop region as a relative value (0.0 to 1.0). 1.0 represents 100% of the image height, 0.5 represents 50% of the image height, etc. Defines the vertical extent of the crop as a proportion of the image height. The crop extends height/2 pixels above and below the y_center coordinate after converting to absolute pixels. Total crop height equals this relative value multiplied by the image height. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries. Relative height adapts to different image sizes - the same relative value extracts the same proportional height from images of any size..

    - output
    
        - `crops` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Relative Static Crop` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/relative_statoic_crop@v1",
	    "images": "$inputs.image",
	    "x_center": 0.3,
	    "y_center": 0.3,
	    "width": 0.3,
	    "height": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

