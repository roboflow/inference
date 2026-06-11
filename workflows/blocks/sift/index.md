
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

    - inputs: [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Contrast Equalization`](contrast_equalization.md), [`Relative Static Crop`](relative_static_crop.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Threshold`](image_threshold.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Dynamic Crop`](dynamic_crop.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md)

    
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

