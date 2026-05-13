
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

    - inputs: [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Line Counter`](line_counter.md), [`Dynamic Crop`](dynamic_crop.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Template Matching`](template_matching.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Email Notification`](email_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detection Event Log`](detection_event_log.md), [`Distance Measurement`](distance_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`LMM`](lmm.md), [`Grid Visualization`](grid_visualization.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md)
    - outputs: [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Color Visualization`](color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Barcode Detection`](barcode_detection.md), [`Google Gemma API`](google_gemma_api.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`Identify Outliers`](identify_outliers.md), [`QR Code Detection`](qr_code_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dominant Color`](dominant_color.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Blur`](image_blur.md), [`Contrast Equalization`](contrast_equalization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SIFT Comparison` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `input_1` (*Union[[`numpy_array`](../kinds/numpy_array.md), [`image`](../kinds/image.md)]*): First input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors..
        - `input_2` (*Union[[`numpy_array`](../kinds/numpy_array.md), [`image`](../kinds/image.md)]*): Second input to compare - can be either an image or pre-computed SIFT descriptors (numpy array). If an image is provided, SIFT keypoints and descriptors will be automatically computed. If descriptors are provided, they will be used directly. Supports images from inputs or workflow steps, or descriptors from SIFT feature detection blocks. Images should be in standard image format, descriptors should be numpy arrays of 128-dimensional SIFT descriptors. This input will be matched against input_1 to determine image similarity..
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

    - inputs: [`Detection Event Log`](detection_event_log.md), [`Template Matching`](template_matching.md), [`Perspective Correction`](perspective_correction.md), [`Image Contours`](image_contours.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter`](line_counter.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Background Subtraction`](background_subtraction.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Identify Outliers`](identify_outliers.md), [`Template Matching`](template_matching.md), [`Triangle Visualization`](triangle_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`QR Code Generator`](qr_code_generator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Pixel Color Count`](pixel_color_count.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dominant Color`](dominant_color.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Gaze Detection`](gaze_detection.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`Detections Consensus`](detections_consensus.md)

    
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

