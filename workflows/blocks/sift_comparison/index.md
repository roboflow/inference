
# SIFT Comparison



## v2

??? "Class: `SIFTComparisonBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/sift_comparison/v2.py">inference.core.workflows.core_steps.classical_cv.sift_comparison.v2.SIFTComparisonBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Compare two images or their SIFT descriptors using configurable matcher algorithms (FLANN or brute force), automatically computing SIFT features when images are provided, applying Lowe's ratio test filtering, and optionally generating visualizations of keypoints and matches for image matching, similarity detection, duplicate detection, and feature-based image comparison workflows.

## How This Block Works

This block compares two images or their SIFT descriptors to determine if they match by finding corresponding features and counting good matches. The block:

1. Receives two inputs (input_1 and input_2) that can be either images or pre-computed SIFT descriptors
2. Processes each input based on its type:
   - **If input is an image**: Automatically computes SIFT keypoints and descriptors using OpenCV's SIFT detector
     - Converts image to grayscale
     - Detects keypoints and computes 128-dimensional SIFT descriptors
     - Optionally creates keypoint visualization if visualize=True
     - Converts keypoints to dictionary format for output
   - **If input is descriptors**: Uses the provided descriptors directly (skips SIFT computation)
3. Validates that both descriptor arrays have at least 2 descriptors (required for ratio test filtering)
4. Selects matcher algorithm based on matcher parameter:
   - **FlannBasedMatcher** (default): Uses FLANN for efficient approximate nearest neighbor search, faster for large descriptor sets
   - **BFMatcher**: Uses brute force matching with L2 norm, exact matching but slower for large descriptor sets
5. Performs k-nearest neighbor matching (k=2) to find the 2 closest descriptor matches for each descriptor in input_1:
   - For each descriptor in descriptors_1, finds the 2 most similar descriptors in descriptors_2
   - Uses Euclidean distance (L2 norm) in descriptor space to measure similarity
   - Returns matches with distance values indicating how similar the descriptors are
6. Filters good matches using Lowe's ratio test:
   - For each match, compares the distance to the best match (m.distance) with the distance to the second-best match (n.distance)
   - Keeps matches where m.distance < ratio_threshold * n.distance
   - This ratio test filters out ambiguous matches where multiple descriptors are similarly close
   - Lower ratio_threshold values (e.g., 0.6) require more distinct matches (stricter filtering)
   - Higher ratio_threshold values (e.g., 0.8) allow more matches (more lenient filtering)
7. Counts the number of good matches after ratio test filtering
8. Determines if images match by comparing good_matches_count to good_matches_threshold:
   - If good_matches_count >= good_matches_threshold, images_match = True
   - If good_matches_count < good_matches_threshold, images_match = False
9. Optionally generates visualizations if visualize=True and images were provided:
   - Creates keypoint visualizations for each image (images with keypoints drawn)
   - Creates a matches visualization showing corresponding keypoints between the two images connected by lines
10. Returns match results, keypoints, descriptors, and optional visualizations

The block provides flexibility by accepting either images (with automatic SIFT computation) or pre-computed descriptors. When images are provided, the block handles all SIFT processing internally, making it easier to use without requiring separate SIFT feature detection steps. The optional visualization feature helps debug and understand matching results by showing keypoints and matches visually. SIFT descriptors are scale and rotation invariant, making the block effective for matching images with different scales, rotations, or viewing angles.

## Common Use Cases

- **Image Similarity Detection**: Determine if two images are similar or match each other (e.g., detect similar images in collections, find matching images in databases, identify duplicate images), enabling image similarity workflows
- **Duplicate Image Detection**: Identify duplicate or near-duplicate images in image collections (e.g., find duplicate images in photo libraries, detect repeated images in datasets, identify identical images with different scales or orientations), enabling duplicate detection workflows
- **Feature-Based Image Matching**: Match images based on visual features and keypoints (e.g., match images with similar content, find corresponding images across different views, identify matching images in image sequences), enabling feature-based matching workflows
- **Image Verification**: Verify if images match expected patterns or references (e.g., verify image authenticity, check if images match reference images, validate image content against templates), enabling image verification workflows
- **Image Comparison and Analysis**: Compare images to analyze similarities and differences (e.g., compare images for quality control, analyze image variations, measure image similarity scores), enabling image comparison analysis workflows
- **Content-Based Image Retrieval**: Use feature matching for content-based image search and retrieval (e.g., find similar images in databases, retrieve images by visual similarity, search images by content matching), enabling content-based retrieval workflows

## Connecting to Other Blocks

This block receives images or SIFT descriptors and produces match results with optional visualizations:

- **After image input blocks** to compare images directly (e.g., compare input images, match images from camera feeds, analyze image similarities), enabling direct image comparison workflows
- **After SIFT feature detection blocks** to compare pre-computed SIFT descriptors (e.g., compare descriptors from different images, match images using existing SIFT features, analyze image similarity with pre-computed descriptors), enabling descriptor-based comparison workflows
- **Before filtering or logic blocks** that use match results for decision-making (e.g., filter based on image matches, make decisions based on similarity, apply logic based on match results), enabling match-based conditional workflows
- **Before data storage blocks** to store match results and visualizations (e.g., store image match results, save similarity scores, record comparison data with visualizations), enabling match result storage workflows
- **Before visualization blocks** to further process or display visualizations (e.g., display match visualizations, show keypoint images, render comparison results), enabling visualization workflow outputs
- **In image comparison pipelines** where multiple images need to be compared (e.g., compare images in sequences, analyze image similarities in workflows, process image comparisons in pipelines), enabling image comparison pipeline workflows

## Version Differences

This version (v2) includes several enhancements over v1:

- **Flexible Input Types**: Accepts both images and pre-computed SIFT descriptors as input (v1 only accepted descriptors), allowing direct image comparison without requiring separate SIFT feature detection steps
- **Automatic SIFT Computation**: Automatically computes SIFT keypoints and descriptors when images are provided, eliminating the need for separate SIFT feature detection blocks in simple workflows
- **Matcher Selection**: Added configurable matcher parameter to choose between FlannBasedMatcher (default, faster) and BFMatcher (exact, slower), providing flexibility for different performance requirements
- **Visualization Support**: Added optional visualization feature that generates keypoint visualizations and match visualizations when images are provided, helping debug and understand matching results
- **Enhanced Outputs**: Returns keypoints and descriptors for both images, plus optional visualizations (keypoint images and match visualization), providing more comprehensive output data for downstream processing


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sift_comparison@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `good_matches_threshold` | `int` | Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features.. | ✅ |
| `ratio_threshold` | `float` | Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features.. | ✅ |
| `matcher` | `str` | Matcher algorithm to use for comparing SIFT descriptors: 'FlannBasedMatcher' (default) uses FLANN for efficient approximate nearest neighbor search - faster for large descriptor sets, suitable for most use cases. 'BFMatcher' uses brute force matching with L2 norm - exact matching but slower for large descriptor sets, useful when you need exact results or have small descriptor sets. Default is 'FlannBasedMatcher' for optimal performance. Choose BFMatcher only if you need exact matching or have performance constraints that favor brute force.. | ✅ |
| `visualize` | `bool` | Whether to generate visualizations of keypoints and matches. When True and images are provided as input, the block generates: (1) visualization_1 and visualization_2 showing keypoints drawn on each image, (2) visualization_matches showing corresponding keypoints between the two images connected by lines. Visualizations are only generated when images (not descriptors) are provided. Default is False. Set to True when you need to debug matching results, understand why images match or don't match, or want visual output for display or analysis purposes.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SIFT Comparison` in version `v2`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`EasyOCR`](easy_ocr.md), [`OpenAI`](open_ai.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Consensus`](detections_consensus.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Icon Visualization`](icon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Contours`](image_contours.md), [`VLM As Classifier`](vlm_as_classifier.md), [`JSON Parser`](json_parser.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Mask Visualization`](mask_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI`](open_ai.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SIFT Comparison` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `input_1` (*Union[[`image`](../kinds/image.md), [`numpy_array`](../kinds/numpy_array.md)]*): First input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors..
        - `input_2` (*Union[[`image`](../kinds/image.md), [`numpy_array`](../kinds/numpy_array.md)]*): Second input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors. This input will be matched against input_1 to determine image similarity..
        - `good_matches_threshold` (*[`integer`](../kinds/integer.md)*): Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features..
        - `ratio_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features..
        - `matcher` (*[`string`](../kinds/string.md)*): Matcher algorithm to use for comparing SIFT descriptors: 'FlannBasedMatcher' (default) uses FLANN for efficient approximate nearest neighbor search - faster for large descriptor sets, suitable for most use cases. 'BFMatcher' uses brute force matching with L2 norm - exact matching but slower for large descriptor sets, useful when you need exact results or have small descriptor sets. Default is 'FlannBasedMatcher' for optimal performance. Choose BFMatcher only if you need exact matching or have performance constraints that favor brute force..
        - `visualize` (*[`boolean`](../kinds/boolean.md)*): Whether to generate visualizations of keypoints and matches. When True and images are provided as input, the block generates: (1) visualization_1 and visualization_2 showing keypoints drawn on each image, (2) visualization_matches showing corresponding keypoints between the two images connected by lines. Visualizations are only generated when images (not descriptors) are provided. Default is False. Set to True when you need to debug matching results, understand why images match or don't match, or want visual output for display or analysis purposes..

    - output
    
        - `images_match` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `good_matches_count` ([`integer`](../kinds/integer.md)): Integer value.
        - `keypoints_1` ([`image_keypoints`](../kinds/image_keypoints.md)): Image keypoints detected by classical Computer Vision method.
        - `descriptors_1` ([`numpy_array`](../kinds/numpy_array.md)): Numpy array.
        - `keypoints_2` ([`image_keypoints`](../kinds/image_keypoints.md)): Image keypoints detected by classical Computer Vision method.
        - `descriptors_2` ([`numpy_array`](../kinds/numpy_array.md)): Numpy array.
        - `visualization_1` ([`image`](../kinds/image.md)): Image in workflows.
        - `visualization_2` ([`image`](../kinds/image.md)): Image in workflows.
        - `visualization_matches` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `SIFT Comparison` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sift_comparison@v2",
	    "input_1": "$inputs.image1",
	    "input_2": "$inputs.image2",
	    "good_matches_threshold": 50,
	    "ratio_threshold": 0.7,
	    "matcher": "FlannBasedMatcher",
	    "visualize": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `SIFTComparisonBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/sift_comparison/v1.py">inference.core.workflows.core_steps.classical_cv.sift_comparison.v1.SIFTComparisonBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Compare SIFT (Scale Invariant Feature Transform) descriptors from two images using FLANN-based matching and Lowe's ratio test, determining image similarity by counting feature matches and returning a boolean match result based on a configurable threshold for image matching, similarity detection, duplicate detection, and feature-based image comparison workflows.

## How This Block Works

This block compares SIFT descriptors from two images to determine if they match by finding corresponding features and counting good matches. The block:

1. Receives SIFT descriptors from two images (descriptor_1 and descriptor_2) - these descriptors should come from a SIFT feature detection step that has already extracted keypoints and computed descriptors for both images
2. Validates that both descriptor arrays have at least 2 descriptors (required for ratio test filtering - needs at least 2 nearest neighbors)
3. Creates a FLANN (Fast Library for Approximate Nearest Neighbors) based matcher:
   - Uses FLANN algorithm for efficient approximate nearest neighbor search in high-dimensional descriptor space
   - Configures FLANN with algorithm parameters optimized for SIFT descriptors (algorithm=1, trees=5, checks=50)
   - FLANN is faster than brute force matching for large descriptor sets while maintaining good accuracy
4. Performs k-nearest neighbor matching (k=2) to find the 2 closest descriptor matches for each descriptor in image 1:
   - For each descriptor in descriptor_1, finds the 2 most similar descriptors in descriptor_2
   - Uses Euclidean distance in descriptor space to measure similarity
   - Returns matches with distance values indicating how similar the descriptors are
5. Filters good matches using Lowe's ratio test:
   - For each match, compares the distance to the best match (m.distance) with the distance to the second-best match (n.distance)
   - Keeps matches where m.distance < ratio_threshold * n.distance
   - This ratio test filters out ambiguous matches where multiple descriptors are similarly close
   - Lower ratio_threshold values (e.g., 0.6) require more distinct matches (stricter filtering)
   - Higher ratio_threshold values (e.g., 0.8) allow more matches (more lenient filtering)
6. Counts the number of good matches after ratio test filtering
7. Determines if images match by comparing good_matches_count to good_matches_threshold:
   - If good_matches_count >= good_matches_threshold, images_match = True
   - If good_matches_count < good_matches_threshold, images_match = False
8. Returns the count of good matches and the boolean match result

The block uses SIFT descriptors which are scale and rotation invariant, making it effective for matching images with different scales, rotations, or viewing angles. FLANN matching provides efficient approximate nearest neighbor search for fast comparison of large descriptor sets. Lowe's ratio test improves match quality by filtering ambiguous matches where the best match isn't significantly better than alternatives. The threshold-based matching allows configurable sensitivity - lower thresholds require fewer matches (more lenient), higher thresholds require more matches (stricter).

## Common Use Cases

- **Image Similarity Detection**: Determine if two images are similar or match each other (e.g., detect similar images in collections, find matching images in databases, identify duplicate images), enabling image similarity workflows
- **Duplicate Image Detection**: Identify duplicate or near-duplicate images in image collections (e.g., find duplicate images in photo libraries, detect repeated images in datasets, identify identical images with different scales or orientations), enabling duplicate detection workflows
- **Feature-Based Image Matching**: Match images based on visual features and keypoints (e.g., match images with similar content, find corresponding images across different views, identify matching images in image sequences), enabling feature-based matching workflows
- **Image Verification**: Verify if images match expected patterns or references (e.g., verify image authenticity, check if images match reference images, validate image content against templates), enabling image verification workflows
- **Image Comparison and Analysis**: Compare images to analyze similarities and differences (e.g., compare images for quality control, analyze image variations, measure image similarity scores), enabling image comparison analysis workflows
- **Content-Based Image Retrieval**: Use feature matching for content-based image search and retrieval (e.g., find similar images in databases, retrieve images by visual similarity, search images by content matching), enabling content-based retrieval workflows

## Connecting to Other Blocks

This block receives SIFT descriptors from two images and produces match results:

- **After SIFT feature detection blocks** to compare SIFT descriptors from different images (e.g., compare descriptors from multiple images, match images using SIFT features, analyze image similarity with SIFT), enabling SIFT-based image comparison workflows
- **Before filtering or logic blocks** that use match results for decision-making (e.g., filter based on image matches, make decisions based on similarity, apply logic based on match results), enabling match-based conditional workflows
- **Before data storage blocks** to store match results (e.g., store image match results, save similarity scores, record comparison data), enabling match result storage workflows
- **In image comparison pipelines** where multiple images need to be compared (e.g., compare images in sequences, analyze image similarities in workflows, process image comparisons in pipelines), enabling image comparison pipeline workflows
- **Before visualization blocks** to visualize match results (e.g., display match results, visualize similar images, show comparison outcomes), enabling match visualization workflows
- **In duplicate detection workflows** where images need to be checked for duplicates (e.g., detect duplicates in image collections, find repeated images, identify identical images), enabling duplicate detection workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sift_comparison@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `good_matches_threshold` | `int` | Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features.. | ✅ |
| `ratio_threshold` | `float` | Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SIFT Comparison` in version `v1`.

    - inputs: [`SIFT`](sift.md), [`Image Contours`](image_contours.md), [`Template Matching`](template_matching.md), [`Line Counter`](line_counter.md), [`SIFT Comparison`](sift_comparison.md), [`Distance Measurement`](distance_measurement.md), [`Detection Event Log`](detection_event_log.md), [`Perspective Correction`](perspective_correction.md), [`Line Counter`](line_counter.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Email Notification`](email_notification.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Icon Visualization`](icon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Contours`](image_contours.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SIFT Comparison` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `descriptor_1` (*[`numpy_array`](../kinds/numpy_array.md)*): SIFT descriptors from the first image to compare. Should be a numpy array of SIFT descriptors (typically from a SIFT feature detection block). Each descriptor is a 128-dimensional vector describing the visual characteristics around a keypoint. The descriptors should be computed using the same SIFT parameters for both images. At least 2 descriptors are required for the ratio test to work. Use descriptors from a SIFT feature detection step that has processed the first image..
        - `descriptor_2` (*[`numpy_array`](../kinds/numpy_array.md)*): SIFT descriptors from the second image to compare. Should be a numpy array of SIFT descriptors (typically from a SIFT feature detection block). Each descriptor is a 128-dimensional vector describing the visual characteristics around a keypoint. The descriptors should be computed using the same SIFT parameters for both images. At least 2 descriptors are required for the ratio test to work. Use descriptors from a SIFT feature detection step that has processed the second image. These descriptors will be matched against descriptor_1 to determine image similarity..
        - `good_matches_threshold` (*[`integer`](../kinds/integer.md)*): Minimum number of good matches required to consider the images as matching. Must be a positive integer. If the number of good matches (after ratio test filtering) is greater than or equal to this threshold, images_match will be True. Lower values (e.g., 20-30) are more lenient and will match images with fewer feature correspondences. Higher values (e.g., 80-100) are stricter and require more feature matches. Default is 50, which provides a good balance. Adjust based on image content, expected similarity level, and false positive/negative tolerance. Use lower thresholds for images with few features, higher thresholds for images with rich texture and many features..
        - `ratio_threshold` (*[`integer`](../kinds/integer.md)*): Ratio threshold for Lowe's ratio test used to filter ambiguous matches. The ratio test compares the distance to the best match with the distance to the second-best match. Matches are kept only if best_match_distance < ratio_threshold * second_best_match_distance. Lower values (e.g., 0.6) require more distinct matches and are stricter (filter out more matches, leaving only high-confidence matches). Higher values (e.g., 0.8) are more lenient (allow more matches, including some ambiguous ones). Default is 0.7, which provides good balance between match quality and quantity. Typical range is 0.6-0.8. Use lower values when you need high-confidence matches only, higher values when you want more matches or have images with sparse features..

    - output
    
        - `good_matches_count` ([`integer`](../kinds/integer.md)): Integer value.
        - `images_match` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `SIFT Comparison` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sift_comparison@v1",
	    "descriptor_1": "$steps.sift.descriptors",
	    "descriptor_2": "$steps.sift.descriptors",
	    "good_matches_threshold": 50,
	    "ratio_threshold": 0.7
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

