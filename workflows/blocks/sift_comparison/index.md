
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

    - inputs: [`Perspective Correction`](perspective_correction.md), [`S3 Sink`](s3_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Qwen-VL`](qwen_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Identify Outliers`](identify_outliers.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Local File Sink`](local_file_sink.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OCR Model`](ocr_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Blur Visualization`](blur_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Mask Visualization`](mask_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`JSON Parser`](json_parser.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Text Display`](text_display.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Label Visualization`](label_visualization.md), [`Camera Focus`](camera_focus.md), [`Qwen3.5`](qwen3.5.md), [`SmolVLM2`](smol_vlm2.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`SIFT`](sift.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Identify Changes`](identify_changes.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Webhook Sink`](webhook_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Seg Preview`](seg_preview.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Gemma`](google_gemma.md)

    
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

    - inputs: [`Distance Measurement`](distance_measurement.md), [`Perspective Correction`](perspective_correction.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter`](line_counter.md), [`Image Contours`](image_contours.md), [`SIFT`](sift.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter`](line_counter.md), [`Template Matching`](template_matching.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md)
    - outputs: [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Generator`](qr_code_generator.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Consensus`](detections_consensus.md), [`Dominant Color`](dominant_color.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md)

    
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

