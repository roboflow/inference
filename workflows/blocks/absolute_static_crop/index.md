
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

    - inputs: [`Icon Visualization`](icon_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Image Preprocessing`](image_preprocessing.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Relative Static Crop`](relative_static_crop.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Image Threshold`](image_threshold.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Barcode Detection`](barcode_detection.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md)

    
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

