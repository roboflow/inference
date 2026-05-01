
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

    - inputs: [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Consensus`](detections_consensus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Mask Visualization`](mask_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Barcode Detection`](barcode_detection.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md)

    
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

