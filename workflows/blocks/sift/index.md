
# SIFT



??? "Class: `SIFTBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/sift/v1.py">inference.core.workflows.core_steps.classical_cv.sift.v1.SIFTBlockV1</a>
    



Detect and describe distinctive visual features in images using SIFT (Scale-Invariant Feature Transform), extracting keypoints (interest points) and computing 128-dimensional feature descriptors that are invariant to scale, rotation, and lighting conditions, enabling feature-based image matching, object recognition, and image similarity detection workflows.

## How This Block Works

This block detects distinctive visual features in an image using SIFT and computes feature descriptors for each detected keypoint. The block:

1. Receives an input image to analyze for feature detection
2. Converts the image to grayscale (SIFT operates on grayscale images for efficiency and robustness)
3. Creates a SIFT detector using OpenCV's SIFT implementation
4. Detects keypoints and computes descriptors simultaneously using detectAndCompute:
   - **Keypoint Detection**: Identifies distinctive interest points (keypoints) in the image that are stable across different viewing conditions
   - Keypoints are detected at multiple scales (pyramid of scale-space images) to handle scale variations
   - Keypoints are detected with orientation assignment to handle rotation variations
   - Each keypoint has properties: position (x, y coordinates), size (scale at which it was detected), angle (orientation), response (strength), octave (scale level), and class_id
   - **Descriptor Computation**: Computes 128-dimensional feature descriptors for each keypoint that describe the local image region around the keypoint
   - Descriptors encode gradient information in the local region, making them distinctive and robust to lighting changes
   - Descriptors are normalized to be partially invariant to illumination changes
5. Draws keypoints on the original image for visualization:
   - Uses OpenCV's drawKeypoints to overlay keypoint markers on the image
   - Visualizes keypoint locations, orientations, and scales
   - Creates a visual representation showing where features were detected
6. Converts keypoints to dictionary format:
   - Extracts keypoint properties (position, size, angle, response, octave, class_id) into dictionaries
   - Makes keypoint data accessible for downstream processing and analysis
7. Returns the image with keypoints drawn, the keypoints data (as dictionaries), and the descriptors (as numpy array)

SIFT features are scale-invariant (work at different zoom levels), rotation-invariant (handle rotated images), and partially lighting-invariant (robust to illumination changes). This makes them highly effective for matching the same object or scene across different images taken from different viewpoints, distances, angles, or lighting conditions. The 128-dimensional descriptors provide rich information about local image regions, enabling robust feature matching and comparison.

## Common Use Cases

- **Feature-Based Image Matching**: Detect features for matching objects or scenes across different images (e.g., match objects in multiple images, find corresponding features across viewpoints, identify matching regions in image pairs), enabling feature-based matching workflows
- **Object Recognition**: Use SIFT features for object recognition and identification (e.g., recognize objects using feature matching, identify objects by their distinctive features, match object features for classification), enabling feature-based object recognition workflows
- **Image Similarity Detection**: Detect similar images by comparing SIFT features (e.g., find similar images in databases, detect duplicate images, identify matching scenes), enabling image similarity workflows
- **Feature Extraction for Analysis**: Extract distinctive features from images for further analysis (e.g., extract features for processing, analyze image characteristics, identify interesting regions), enabling feature extraction workflows
- **Visual Localization**: Use SIFT features for visual localization and mapping (e.g., localize objects in scenes, track features across frames, map feature correspondences), enabling visual localization workflows
- **Image Registration**: Align images using SIFT feature correspondences (e.g., register images for stitching, align images from different viewpoints, match images for alignment), enabling image registration workflows

## Connecting to Other Blocks

This block receives an image and produces SIFT keypoints and descriptors:

- **After image input blocks** to extract SIFT features from input images (e.g., detect features in camera feeds, extract features from image inputs, analyze features in images), enabling SIFT feature extraction workflows
- **After preprocessing blocks** to extract features from preprocessed images (e.g., detect features after filtering, extract features from enhanced images, analyze features after preprocessing), enabling preprocessed feature extraction workflows
- **Before SIFT Comparison blocks** to provide SIFT descriptors for image comparison (e.g., provide descriptors for matching, prepare features for comparison, supply descriptors for similarity detection), enabling SIFT-based image comparison workflows
- **Before filtering or logic blocks** that use feature counts or properties for decision-making (e.g., filter based on feature count, make decisions based on detected features, apply logic based on feature properties), enabling feature-based conditional workflows
- **Before data storage blocks** to store feature data (e.g., store keypoints and descriptors, save feature information, record feature data for analysis), enabling feature data storage workflows
- **Before visualization blocks** to display detected features (e.g., visualize keypoints, display feature locations, show feature analysis results), enabling feature visualization workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sift@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SIFT` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Camera Focus`](camera_focus.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`OCR Model`](ocr_model.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SIFT` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to analyze for SIFT feature detection. The image will be converted to grayscale internally for SIFT processing. SIFT works best on images with good texture and detail - images with rich visual content (edges, corners, patterns) produce more keypoints than uniform or smooth images. Each detected keypoint will have a 128-dimensional descriptor computed. The output includes an image with keypoints drawn for visualization, keypoint data (position, size, angle, response, octave), and descriptor arrays for matching and comparison. SIFT features are scale and rotation invariant, making them effective for matching across different viewpoints and conditions..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.
        - `keypoints` ([`image_keypoints`](../kinds/image_keypoints.md)): Image keypoints detected by classical Computer Vision method.
        - `descriptors` ([`numpy_array`](../kinds/numpy_array.md)): Numpy array.



??? tip "Example JSON definition of step `SIFT` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sift@v1",
	    "image": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

