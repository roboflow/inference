
# Absolute Static Crop



??? "Class: `AbsoluteStaticCropBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/absolute_static_crop/v1.py">inference.core.workflows.core_steps.transformations.absolute_static_crop.v1.AbsoluteStaticCropBlockV1</a>
    



Extract a fixed rectangular region from input images using absolute pixel coordinates specified by center point and dimensions, creating consistent crops from the same image location across all inputs for region-of-interest extraction and fixed-area analysis workflows.

## How This Block Works

This block crops a fixed rectangular region from input images using absolute pixel coordinates, unlike dynamic cropping which uses detection bounding boxes. The block:

1. Receives input images and absolute coordinate specifications (x_center, y_center, width, height)
2. Calculates the crop boundaries from the center point and dimensions:
   - Computes x_min and y_min by subtracting half the width/height from the center coordinates
   - Computes x_max and y_max by adding the width/height to the minimum coordinates
   - Rounds coordinate values to integer pixel positions
3. Extracts the rectangular region from the image using array slicing (from y_min to y_max, x_min to x_max)
4. Validates that the cropped region has content (returns None if the crop would be empty, such as when coordinates are outside image bounds)
5. Creates a cropped image object with metadata tracking the crop's origin (original image, offset coordinates, unique crop identifier)
6. Preserves video metadata if the input is from video (maintains frame information and temporal context)
7. Returns the cropped image for each input image

The block uses fixed coordinates, so the same region is extracted from all images in a batch, making it suitable for extracting consistent regions across multiple images (e.g., always cropping the top-right corner, extracting a fixed area of interest, or focusing on a specific image section). The center-based coordinate system allows specifying crops by their center point rather than corner coordinates, which can be more intuitive for defining regions. The block handles edge cases gracefully by returning None for invalid crops (coordinates outside image bounds or resulting in empty regions).

## Common Use Cases

- **Fixed Region Extraction**: Extract the same image region from multiple images for consistent analysis (e.g., crop a specific area of interest like a logo zone, extract a fixed region for watermark detection, crop a consistent area for pattern matching), enabling standardized region analysis across image batches
- **Region-of-Interest Focus**: Isolate specific areas of images for detailed processing (e.g., crop a specific quadrant of surveillance frames, extract a fixed region for text recognition, focus on a known area of interest), enabling focused analysis of predetermined image regions
- **Multi-Stage Workflow Preparation**: Extract fixed regions for secondary processing steps (e.g., crop a specific area from full images, then run OCR or classification on the cropped region), enabling hierarchical workflows with fixed region focus
- **Standardized Crop Generation**: Create consistent crops from images for training or analysis (e.g., extract a fixed region from all images for dataset creation, crop a standard area for comparison, generate uniform crops for feature extraction), enabling standardized data preparation workflows
- **Fixed-Area Monitoring**: Monitor specific image regions across time or batches (e.g., crop the same area from video frames for change detection, extract a fixed region for tracking analysis, focus on a consistent monitoring zone), enabling temporal analysis of fixed regions
- **Pre-Processing for Specialized Blocks**: Extract fixed regions before processing with specialized models (e.g., crop a specific area before running OCR, extract a fixed region for fine-grained classification, isolate a region for specialized analysis), enabling optimized processing of known image regions

## Connecting to Other Blocks

This block receives images and produces cropped images from fixed regions:

- **After image loading blocks** to extract a fixed region of interest before processing, enabling focused analysis of predetermined image areas without processing entire images
- **Before classification or analysis blocks** that need region-focused inputs (e.g., OCR for text in a fixed area, fine-grained classification for cropped regions, specialized models for specific image areas), enabling optimized processing of consistent regions
- **In video processing workflows** to extract the same region from multiple frames (e.g., crop a fixed area from each video frame for temporal analysis, extract a consistent monitoring zone for tracking, focus on a specific region across frames), enabling temporal analysis of fixed regions
- **After detection blocks** where you know the approximate location and want to extract a fixed-size region around it (e.g., detect objects in a general area, then crop a fixed region around that area for detailed analysis), enabling region-focused multi-stage workflows
- **Before visualization blocks** that display specific regions (e.g., display only the cropped region, visualize a fixed area of interest, show isolated region annotations), enabling focused visualization of extracted regions
- **In batch processing workflows** where the same region needs to be extracted from all images for consistent analysis or comparison, enabling standardized region extraction across image sets


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/absolute_static_crop@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `x_center` | `int` | X coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this X coordinate. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty.. | ✅ |
| `y_center` | `int` | Y coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this Y coordinate. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty.. | ✅ |
| `width` | `int` | Width of the crop region in pixels. Must be a positive integer. Defines the horizontal extent of the crop. The crop extends width/2 pixels to the left and right of the x_center coordinate. Total crop width equals this value. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries.. | ✅ |
| `height` | `int` | Height of the crop region in pixels. Must be a positive integer. Defines the vertical extent of the crop. The crop extends height/2 pixels above and below the y_center coordinate. Total crop height equals this value. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Absolute Static Crop` in version `v1`.

    - inputs: [`Perspective Correction`](perspective_correction.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Line Counter`](line_counter.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Line Counter`](line_counter.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Mask Visualization`](mask_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Pixel Color Count`](pixel_color_count.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detection Event Log`](detection_event_log.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Grid Visualization`](grid_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT`](sift.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Absolute Static Crop` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `x_center` (*[`integer`](../kinds/integer.md)*): X coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this X coordinate. The actual crop boundaries are calculated as x_min = x_center - width/2 and x_max = x_min + width. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty..
        - `y_center` (*[`integer`](../kinds/integer.md)*): Y coordinate of the center point of the crop region in absolute pixel coordinates. Must be a positive integer. The crop region is centered at this Y coordinate. The actual crop boundaries are calculated as y_min = y_center - height/2 and y_max = y_min + height. If the calculated crop extends beyond image bounds, the crop will be clipped or may return None if the crop would be empty..
        - `width` (*[`integer`](../kinds/integer.md)*): Width of the crop region in pixels. Must be a positive integer. Defines the horizontal extent of the crop. The crop extends width/2 pixels to the left and right of the x_center coordinate. Total crop width equals this value. If the calculated crop extends beyond the image's width, it will be clipped to image boundaries..
        - `height` (*[`integer`](../kinds/integer.md)*): Height of the crop region in pixels. Must be a positive integer. Defines the vertical extent of the crop. The crop extends height/2 pixels above and below the y_center coordinate. Total crop height equals this value. If the calculated crop extends beyond the image's height, it will be clipped to image boundaries..

    - output
    
        - `crops` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Absolute Static Crop` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/absolute_static_crop@v1",
	    "images": "$inputs.image",
	    "x_center": 40,
	    "y_center": 40,
	    "width": 40,
	    "height": 40
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

