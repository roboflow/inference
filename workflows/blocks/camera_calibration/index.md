
# Camera Calibration



??? "Class: `CameraCalibrationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/camera_calibration/v1.py">inference.core.workflows.core_steps.transformations.camera_calibration.v1.CameraCalibrationBlockV1</a>
    



Remove lens distortions from images using camera calibration parameters (focal lengths, optical centers, and distortion coefficients) to correct radial and tangential distortions introduced by camera lenses, producing undistorted images suitable for accurate measurement, geometric analysis, and precision computer vision applications.

## How This Block Works

Camera lenses introduce distortions that cause straight lines to appear curved and objects near image edges to appear stretched or compressed. This block corrects these distortions using known camera calibration parameters. The block:

1. Receives input images and camera calibration parameters (focal lengths fx/fy, optical centers cx/cy, radial distortion coefficients k1/k2/k3, tangential distortion coefficients p1/p2)
2. Constructs a camera matrix from the intrinsic parameters (focal lengths and optical centers) in the standard OpenCV format: 3x3 matrix with fx, fy on the diagonal, cx, cy as the optical center, and 1 in the bottom-right corner
3. Assembles distortion coefficients into a 5-element array (k1, k2, p1, p2, k3) representing radial and tangential distortion parameters
4. Computes an optimal new camera matrix using OpenCV's `getOptimalNewCameraMatrix` to maximize the usable image area after correction (removes black borders that result from distortion correction)
5. Applies OpenCV's `undistort` function to correct both radial distortions (barrel and pincushion distortion causing curved lines) and tangential distortions (lens misalignment causing skewed images)
6. Returns the corrected, undistorted image with straight lines corrected, edge distortions removed, and geometric accuracy restored

The block uses OpenCV's camera calibration functions under the hood, following standard computer vision camera calibration methodology (see [OpenCV calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for details on obtaining calibration parameters). Radial distortion coefficients (k1, k2, k3) correct barrel/pincushion distortion where image points are displaced radially from the optical center. Tangential distortion coefficients (p1, p2) correct distortion caused by lens misalignment. The calibration parameters must be obtained beforehand through a camera calibration process (typically using checkerboard patterns) or provided by the camera manufacturer.

## Requirements

**Camera Calibration Parameters**: This block requires pre-computed camera calibration parameters obtained through camera calibration:
- **Focal lengths (fx, fy)**: Pixel focal lengths along x and y axes (may differ for non-square pixels)
- **Optical centers (cx, cy)**: Principal point coordinates (image center in ideal cameras)
- **Radial distortion coefficients (k1, k2, k3)**: Correct barrel and pincushion distortion
- **Tangential distortion coefficients (p1, p2)**: Correct lens misalignment distortion

These parameters are typically obtained using OpenCV's camera calibration process with a checkerboard pattern or similar calibration target. See [OpenCV camera calibration documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for calibration methodology.

## Common Use Cases

- **Measurement and Metrology Applications**: Correct lens distortions for accurate measurement workflows (e.g., remove distortions before measuring object sizes, correct geometric distortions for precision measurements, undistort images for dimensional analysis), enabling accurate measurements from camera images
- **Geometric Analysis Workflows**: Prepare images for geometric computer vision tasks (e.g., undistort images before line detection, correct distortions for geometric shape analysis, prepare images for accurate angle measurements), enabling precise geometric analysis with corrected images
- **Multi-Camera Systems**: Standardize images from multiple cameras with different lens characteristics (e.g., undistort images from different camera angles, correct wide-angle lens distortions, standardize images from multiple cameras for stereo vision), enabling consistent image geometry across camera setups
- **Pre-Processing for Precision Models**: Prepare images for models requiring high geometric accuracy (e.g., undistort images before running geometric models, correct distortions for accurate feature detection, prepare images for precise pose estimation), enabling better accuracy for geometric computer vision tasks
- **Wide-Angle and Fisheye Correction**: Correct severe distortions from wide-angle or fisheye lenses (e.g., correct barrel distortion from wide-angle lenses, remove fisheye distortion effects, straighten curved lines in wide-angle images), enabling use of wide-angle lenses with standard computer vision workflows
- **Video Stabilization Preparation**: Correct lens distortions as part of video stabilization pipelines (e.g., undistort video frames before stabilization, correct camera-specific distortions in video streams, prepare frames for motion analysis), enabling more accurate video processing

## Connecting to Other Blocks

This block receives images and produces undistorted images:

- **After image loading blocks** to correct lens distortions before processing, enabling accurate analysis with geometrically correct images
- **Before measurement and analysis blocks** that require geometric accuracy (e.g., size measurement, angle measurement, distance calculation, geometric shape analysis), enabling precise measurements from undistorted images
- **Before geometric computer vision blocks** that analyze lines, shapes, or spatial relationships (e.g., line detection, contour analysis, geometric pattern matching, pose estimation), enabling accurate geometric analysis with corrected images
- **In multi-camera workflows** to standardize images from different cameras before processing (e.g., undistort images from different camera angles, correct camera-specific distortions before comparison, standardize images for stereo vision), enabling consistent processing across camera setups
- **Before detection or classification blocks** in precision applications where geometric accuracy matters (e.g., detect objects in undistorted images for accurate localization, classify objects in geometrically correct images, run models requiring precise spatial relationships), enabling improved accuracy for detection and classification tasks
- **In video processing workflows** to correct distortions in video frames (e.g., undistort video frames for motion analysis, correct camera distortions in video streams, prepare frames for tracking algorithms), enabling accurate video analysis with corrected frames


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/camera-calibration@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `fx` | `float` | Focal length along the x-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's horizontal focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications.. | ✅ |
| `fy` | `float` | Focal length along the y-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's vertical focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications.. | ✅ |
| `cx` | `float` | Optical center (principal point) x-coordinate in pixels. Part of the camera's intrinsic parameters representing the x-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image width. Obtained through camera calibration. Used with cy to define the optical center of the camera.. | ✅ |
| `cy` | `float` | Optical center (principal point) y-coordinate in pixels. Part of the camera's intrinsic parameters representing the y-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image height. Obtained through camera calibration. Used with cx to define the optical center of the camera.. | ✅ |
| `k1` | `float` | First radial distortion coefficient. Part of the camera's distortion parameters used to correct barrel and pincushion distortion (where straight lines appear curved). k1 is typically the dominant radial distortion term. Positive values often indicate barrel distortion, negative values indicate pincushion distortion. Obtained through camera calibration.. | ✅ |
| `k2` | `float` | Second radial distortion coefficient. Part of the camera's distortion parameters used to correct higher-order radial distortion effects. k2 helps correct more complex radial distortion patterns beyond the first-order k1 term. Obtained through camera calibration. Often smaller in magnitude than k1.. | ✅ |
| `k3` | `float` | Third radial distortion coefficient. Part of the camera's distortion parameters used to correct additional higher-order radial distortion effects. k3 is typically the smallest radial distortion term and is used for very precise distortion correction, especially for wide-angle lenses. Obtained through camera calibration. Often set to 0 for standard lenses.. | ✅ |
| `p1` | `float` | First tangential distortion coefficient. Part of the camera's distortion parameters used to correct tangential distortion caused by lens misalignment. p1 corrects skew distortions where the lens is not perfectly aligned with the image sensor. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero.. | ✅ |
| `p2` | `float` | Second tangential distortion coefficient. Part of the camera's distortion parameters used to correct additional tangential distortion effects. p2 works together with p1 to correct lens misalignment distortions. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero.. | ✅ |
| `use_fisheye_model` | `bool` | Enable Fisheye distortion model (Rational/Divisional). If true, uses a different mathematical model better suited for fisheye lenses. When enabled, k1 is the primary parameter, and other coefficients are typically 0.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Camera Calibration` in version `v1`.

    - inputs: [`Slack Notification`](slack_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Circle Visualization`](circle_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Image Blur`](image_blur.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Blur Visualization`](blur_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Relative Static Crop`](relative_static_crop.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Camera Focus`](camera_focus.md), [`Grid Visualization`](grid_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Gaze Detection`](gaze_detection.md), [`Camera Calibration`](camera_calibration.md), [`SIFT`](sift.md), [`Email Notification`](email_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Preprocessing`](image_preprocessing.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`JSON Parser`](json_parser.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Motion Detection`](motion_detection.md), [`Crop Visualization`](crop_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Perspective Correction`](perspective_correction.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Image Slicer`](image_slicer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Camera Focus`](camera_focus.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Trace Visualization`](trace_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`QR Code Detection`](qr_code_detection.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Halo Visualization`](halo_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Camera Calibration` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to remove lens distortions from. The image will be corrected for radial and tangential distortions using the provided camera calibration parameters. Works with images from cameras with known calibration parameters. The undistorted output image will have corrected geometry with straight lines straightened and edge distortions removed..
        - `fx` (*[`float`](../kinds/float.md)*): Focal length along the x-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's horizontal focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications..
        - `fy` (*[`float`](../kinds/float.md)*): Focal length along the y-axis in pixels. Part of the camera's intrinsic parameters. Typically obtained through camera calibration (e.g., using OpenCV calibration with a checkerboard pattern). Represents the camera's vertical focal length. For square pixels, fx and fy are usually equal. Must be obtained from camera calibration or manufacturer specifications..
        - `cx` (*[`float`](../kinds/float.md)*): Optical center (principal point) x-coordinate in pixels. Part of the camera's intrinsic parameters representing the x-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image width. Obtained through camera calibration. Used with cy to define the optical center of the camera..
        - `cy` (*[`float`](../kinds/float.md)*): Optical center (principal point) y-coordinate in pixels. Part of the camera's intrinsic parameters representing the y-coordinate of the camera's principal point (image center in ideal cameras). Typically near half the image height. Obtained through camera calibration. Used with cx to define the optical center of the camera..
        - `k1` (*[`float`](../kinds/float.md)*): First radial distortion coefficient. Part of the camera's distortion parameters used to correct barrel and pincushion distortion (where straight lines appear curved). k1 is typically the dominant radial distortion term. Positive values often indicate barrel distortion, negative values indicate pincushion distortion. Obtained through camera calibration..
        - `k2` (*[`float`](../kinds/float.md)*): Second radial distortion coefficient. Part of the camera's distortion parameters used to correct higher-order radial distortion effects. k2 helps correct more complex radial distortion patterns beyond the first-order k1 term. Obtained through camera calibration. Often smaller in magnitude than k1..
        - `k3` (*[`float`](../kinds/float.md)*): Third radial distortion coefficient. Part of the camera's distortion parameters used to correct additional higher-order radial distortion effects. k3 is typically the smallest radial distortion term and is used for very precise distortion correction, especially for wide-angle lenses. Obtained through camera calibration. Often set to 0 for standard lenses..
        - `p1` (*[`float`](../kinds/float.md)*): First tangential distortion coefficient. Part of the camera's distortion parameters used to correct tangential distortion caused by lens misalignment. p1 corrects skew distortions where the lens is not perfectly aligned with the image sensor. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero..
        - `p2` (*[`float`](../kinds/float.md)*): Second tangential distortion coefficient. Part of the camera's distortion parameters used to correct additional tangential distortion effects. p2 works together with p1 to correct lens misalignment distortions. Obtained through camera calibration. For well-aligned lenses, p1 and p2 are often close to zero..
        - `use_fisheye_model` (*[`boolean`](../kinds/boolean.md)*): Enable Fisheye distortion model (Rational/Divisional). If true, uses a different mathematical model better suited for fisheye lenses. When enabled, k1 is the primary parameter, and other coefficients are typically 0..

    - output
    
        - `calibrated_image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Camera Calibration` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/camera-calibration@v1",
	    "image": "$inputs.image",
	    "fx": 0.123,
	    "fy": 0.123,
	    "cx": 0.123,
	    "cy": 0.123,
	    "k1": 0.123,
	    "k2": 0.123,
	    "k3": 0.123,
	    "p1": 0.123,
	    "p2": 0.123,
	    "use_fisheye_model": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

