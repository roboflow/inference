
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

    - inputs: [`Heatmap Visualization`](heatmap_visualization.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter`](line_counter.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`Blur Visualization`](blur_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Depth Estimation`](depth_estimation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch Images`](stitch_images.md), [`Halo Visualization`](halo_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Contrast Equalization`](contrast_equalization.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Line Counter`](line_counter.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Image Slicer`](image_slicer.md), [`Text Display`](text_display.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Polygon Visualization`](polygon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Camera Focus`](camera_focus.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen3.5`](qwen3.5.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
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

