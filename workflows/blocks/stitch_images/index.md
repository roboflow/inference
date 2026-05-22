
# Stitch Images



??? "Class: `StitchImagesBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/stitch_images/v1.py">inference.core.workflows.core_steps.transformations.stitch_images.v1.StitchImagesBlockV1</a>
    



Stitch two overlapping images together into a single panoramic image using SIFT (Scale Invariant Feature Transform) feature matching and homography-based image alignment, automatically detecting common features, calculating geometric transformations, and blending images to create seamless panoramic compositions from overlapping scenes.

## How This Block Works

This block stitches two overlapping images together by detecting common features, calculating geometric transformations, and aligning the images into a single panoramic result. The block:

1. Receives two input images (image1 and image2) that contain overlapping regions with sufficient detail for feature matching
2. Detects keypoints and computes descriptors using SIFT (Scale Invariant Feature Transform) for both images:
   - Identifies distinctive feature points (keypoints) in each image that are invariant to scale and rotation
   - Computes feature descriptors (128-dimensional vectors) describing the visual characteristics around each keypoint
3. Matches keypoints between the two images using brute force matching:
   - Finds the best matching descriptors for each keypoint in image1 among all keypoints in image2
   - Uses k-nearest neighbor matching (configurable via count_of_best_matches_per_query_descriptor) to find multiple potential matches per query keypoint
4. Filters good matches using Lowe's ratio test:
   - Compares the distance to the best match with the distance to the second-best match
   - Keeps matches where the best match distance is less than 0.75 times the second-best match distance (reduces false matches)
5. Determines image ordering based on keypoint positions (identifies which image should be placed first based on spatial distribution of matched features)
6. Calculates homography transformation matrix using RANSAC (Random Sample Consensus):
   - Finds a perspective transformation matrix that maps points from one image to the other
   - Uses RANSAC to robustly estimate the transformation while filtering out outlier matches
   - Configurable maximum reprojection error (max_allowed_reprojection_error) controls which point pairs are considered inliers
7. Calculates canvas size and translation:
   - Determines the size needed to contain both images after transformation
   - Calculates translation needed to ensure both images fit within the canvas boundaries
8. Warps the second image using the homography transformation:
   - Applies perspective transformation to align the second image with the first
   - Combines homography matrix with translation matrix for correct positioning
9. Stitches images together:
   - Places the first image onto the warped second image canvas
   - Creates the final stitched panoramic image containing both input images aligned and blended
10. Returns the stitched image, or None if stitching fails (e.g., insufficient matches, transformation calculation failure)

The block uses SIFT for robust feature detection that works well with images containing sufficient detail and texture. The RANSAC-based homography calculation handles perspective distortions and ensures robust alignment even with some incorrect matches. The reprojection error threshold controls the sensitivity of the alignment - lower values require more precise matches, while higher values (useful for low-detail images) allow more tolerance for matching variations.

## Common Use Cases

- **Panoramic Image Creation**: Stitch overlapping images together to create wide panoramic views (e.g., create panoramic photos from overlapping camera shots, stitch together images from rotating cameras, combine multiple overlapping images into panoramas), enabling panoramic image generation workflows
- **Wide-Area Scene Reconstruction**: Combine multiple overlapping views of a scene into a single comprehensive image (e.g., reconstruct wide scenes from multiple camera angles, combine overlapping surveillance camera views, stitch together images from multiple viewpoints), enabling wide-area scene visualization
- **Multi-Image Mosaicking**: Create image mosaics from overlapping image tiles or sections (e.g., stitch together image tiles for large-scale mapping, combine overlapping satellite image sections, create mosaics from overlapping image captures), enabling image mosaic creation workflows
- **Scene Documentation**: Combine multiple overlapping images to document large scenes or areas (e.g., document large spaces with multiple overlapping photos, combine overlapping views for scene documentation, stitch together images for comprehensive scene capture), enabling comprehensive scene documentation
- **Video Frame Stitching**: Stitch together overlapping frames from video sequences (e.g., create panoramic views from video frames, combine overlapping frames from moving cameras, stitch together consecutive video frames), enabling video-based panoramic workflows
- **Multi-Camera View Combination**: Combine overlapping views from multiple cameras into a single unified view (e.g., stitch together overlapping camera feeds, combine multi-camera views for monitoring, merge overlapping camera perspectives), enabling multi-camera view integration workflows

## Connecting to Other Blocks

This block receives two images and produces a single stitched image:

- **After image input blocks** or **image preprocessing blocks** to stitch preprocessed images together (e.g., stitch images after preprocessing, combine images after enhancement, merge images after filtering), enabling image stitching workflows
- **After crop blocks** to stitch together cropped image regions from different sources (e.g., stitch cropped regions from different images, combine cropped sections from multiple sources, merge cropped regions into panoramas), enabling cropped region stitching workflows
- **After transformation blocks** to stitch images that have been transformed or adjusted (e.g., stitch images after perspective correction, combine images after geometric transformations, merge images after adjustments), enabling transformed image stitching workflows
- **Before detection or analysis blocks** that benefit from panoramic views (e.g., detect objects in stitched panoramic images, analyze wide-area stitched scenes, process comprehensive stitched views), enabling panoramic analysis workflows
- **Before visualization blocks** to display stitched panoramic images (e.g., visualize stitched panoramas, display wide-area stitched views, show comprehensive stitched scenes), enabling panoramic visualization outputs
- **In multi-stage image processing workflows** where images need to be stitched before further processing (e.g., stitch images before detection, combine images before analysis, merge images for comprehensive processing), enabling multi-stage panoramic processing workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stitch_images@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `max_allowed_reprojection_error` | `float` | Maximum allowed reprojection error (in pixels) to treat a point pair as an inlier during RANSAC homography calculation. This corresponds to cv.findHomography's ransacReprojThreshold parameter. Lower values require more precise matches (stricter alignment) but may fail with noisy matches. Higher values allow more tolerance for matching variations (more lenient alignment) and can improve results for low-detail images or images with imperfect feature matches. Default is 3 pixels. Increase this value (e.g., 5-10) for images with less detail or when stitching fails with default settings.. | ✅ |
| `count_of_best_matches_per_query_descriptor` | `int` | Number of best matches to find per query descriptor during keypoint matching. This corresponds to cv.BFMatcher.knnMatch's k parameter. Must be greater than 0. The block finds the k nearest neighbor matches for each keypoint descriptor in image1 among all descriptors in image2. Then uses Lowe's ratio test to filter good matches (comparing best match distance with second-best match distance). Higher values provide more candidate matches but increase computation. Default is 2 (finds 2 best matches per descriptor). Typical values range from 2-5. Use higher values if you need more match candidates for difficult images.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stitch Images` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Identify Outliers`](identify_outliers.md), [`Crop Visualization`](crop_visualization.md), [`Camera Focus`](camera_focus.md), [`Blur Visualization`](blur_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Slicer`](image_slicer.md), [`Trace Visualization`](trace_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Detection Event Log`](detection_event_log.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Stitch Images`](stitch_images.md), [`Distance Measurement`](distance_measurement.md), [`Image Blur`](image_blur.md), [`Clip Comparison`](clip_comparison.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Template Matching`](template_matching.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Line Counter`](line_counter.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Line Counter`](line_counter.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Halo Visualization`](halo_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Identify Changes`](identify_changes.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)
    - outputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Seg Preview`](seg_preview.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Blur`](image_blur.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Buffer`](buffer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`LMM`](lmm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Object Detection Model`](object_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen3.5`](qwen3.5.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`CogVLM`](cog_vlm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Moondream2`](moondream2.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Template Matching`](template_matching.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`QR Code Detection`](qr_code_detection.md), [`Label Visualization`](label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Circle Visualization`](circle_visualization.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stitch Images` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image1` (*[`image`](../kinds/image.md)*): First input image to stitch. Should contain overlapping regions with image2 and sufficient detail/texture for SIFT feature detection. The images must have overlapping content for successful stitching. The block will determine the optimal positioning and alignment of this image relative to image2 during stitching. Images with rich texture and detail work best for SIFT-based feature matching..
        - `image2` (*[`image`](../kinds/image.md)*): Second input image to stitch. Should contain overlapping regions with image1 and sufficient detail/texture for SIFT feature detection. The images must have overlapping content for successful stitching. The block will warp and align this image to match image1's perspective during stitching. Images with rich texture and detail work best for SIFT-based feature matching..
        - `max_allowed_reprojection_error` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Maximum allowed reprojection error (in pixels) to treat a point pair as an inlier during RANSAC homography calculation. This corresponds to cv.findHomography's ransacReprojThreshold parameter. Lower values require more precise matches (stricter alignment) but may fail with noisy matches. Higher values allow more tolerance for matching variations (more lenient alignment) and can improve results for low-detail images or images with imperfect feature matches. Default is 3 pixels. Increase this value (e.g., 5-10) for images with less detail or when stitching fails with default settings..
        - `count_of_best_matches_per_query_descriptor` (*[`integer`](../kinds/integer.md)*): Number of best matches to find per query descriptor during keypoint matching. This corresponds to cv.BFMatcher.knnMatch's k parameter. Must be greater than 0. The block finds the k nearest neighbor matches for each keypoint descriptor in image1 among all descriptors in image2. Then uses Lowe's ratio test to filter good matches (comparing best match distance with second-best match distance). Higher values provide more candidate matches but increase computation. Default is 2 (finds 2 best matches per descriptor). Typical values range from 2-5. Use higher values if you need more match candidates for difficult images..

    - output
    
        - `stitched_image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Stitch Images` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stitch_images@v1",
	    "image1": "$inputs.image1",
	    "image2": "$inputs.image2",
	    "max_allowed_reprojection_error": 3,
	    "count_of_best_matches_per_query_descriptor": 2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

