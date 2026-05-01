
# Image Threshold



??? "Class: `ImageThresholdBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/threshold/v1.py">inference.core.workflows.core_steps.classical_cv.threshold.v1.ImageThresholdBlockV1</a>
    



Convert grayscale images to binary images using configurable thresholding methods (binary, binary_inv, trunc, tozero, tozero_inv, adaptive_mean, adaptive_gaussian, otsu) to separate foreground from background, isolate objects, prepare images for morphological operations, and create binary masks for segmentation, object detection, and analysis workflows.

## How This Block Works

This block applies thresholding operations to convert grayscale images into binary images where pixels are classified as either foreground (white) or background (black). The block:

1. Receives a grayscale input image (color images must be converted to grayscale first using an Image Convert Grayscale block)
2. Determines the thresholding method from the threshold_type parameter
3. Applies the selected thresholding operation:

   **For Binary Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold become max_value (white), pixels below become 0 (black)
   - Creates a binary image with clear separation between foreground and background
   - Best for images with uniform lighting and clear contrast

   **For Binary Inverse Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold become 0 (black), pixels below become max_value (white)
   - Inverts the binary result - dark objects become white, light backgrounds become black
   - Useful when dark objects need to be foreground (white) in the output

   **For Truncate Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold are set to the threshold value, pixels below remain unchanged
   - Clips bright pixels while preserving dark pixel values
   - Useful for reducing brightness in overexposed regions

   **For To Zero Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels below the threshold are set to 0 (black), pixels above remain unchanged
   - Removes dark pixels while preserving bright pixel values
   - Useful for removing noise in dark regions

   **For To Zero Inverse Threshold:**
   - Uses a fixed threshold value (thresh_value)
   - Pixels above the threshold are set to 0 (black), pixels below remain unchanged
   - Removes bright pixels while preserving dark pixel values
   - Useful for removing noise in bright regions

   **For Adaptive Mean Threshold:**
   - Calculates threshold values locally using mean of neighborhood pixels
   - Adapts to local image characteristics (block size 11x11, constant 2)
   - Handles varying lighting conditions and illumination gradients
   - Best for images with non-uniform lighting

   **For Adaptive Gaussian Threshold:**
   - Calculates threshold values locally using weighted Gaussian mean of neighborhood pixels
   - Adapts to local image characteristics (block size 11x11, constant 2)
   - Handles varying lighting conditions with smoother transitions than adaptive mean
   - Best for images with non-uniform lighting requiring smoother adaptation

   **For Otsu's Threshold:**
   - Automatically calculates optimal threshold value using Otsu's method
   - Analyzes histogram to find threshold that minimizes intra-class variance
   - No manual threshold value needed (thresh_value is ignored)
   - Best for bimodal histograms with clear foreground/background separation

4. For fixed threshold methods (binary, binary_inv, trunc, tozero, tozero_inv), uses thresh_value as the threshold and max_value (typically 255) as the maximum output value
5. For adaptive methods (adaptive_mean, adaptive_gaussian), uses max_value as the maximum output value and adapts locally
6. For Otsu's method, automatically determines the optimal threshold and uses max_value as the maximum output value
7. Preserves image structure and metadata
8. Returns the thresholded binary image

Thresholding converts grayscale images to binary images by classifying pixels based on intensity values. Fixed threshold methods use a single threshold value for the entire image - simple and fast but sensitive to lighting variations. Adaptive threshold methods calculate local thresholds for each pixel neighborhood - more robust to lighting variations but computationally more expensive. Otsu's method automatically selects an optimal global threshold by analyzing the image histogram - works well for images with bimodal intensity distributions. The threshold_type controls the method, thresh_value sets the threshold for fixed methods, and max_value (typically 255) determines the white pixel value in binary outputs.

## Common Use Cases

- **Binary Image Creation**: Convert grayscale images to binary images for object detection and analysis (e.g., create binary masks, isolate objects from background, separate foreground and background), enabling binary image creation workflows
- **Object Segmentation**: Isolate objects from backgrounds for segmentation tasks (e.g., segment objects from backgrounds, create object masks, isolate regions of interest), enabling object segmentation workflows
- **Document Processing**: Extract text and content from scanned documents (e.g., binarize document images, enhance text contrast, prepare documents for OCR), enabling document processing workflows
- **Image Preprocessing**: Prepare images for morphological operations and contour detection (e.g., create binary images for morphology, prepare images for contour analysis, binarize for shape analysis), enabling preprocessing workflows
- **Noise Removal Preparation**: Create binary images for noise removal and cleaning operations (e.g., prepare images for morphological cleaning, create masks for filtering, binarize for denoising), enabling noise removal workflows
- **Feature Detection**: Prepare images for feature detection and analysis (e.g., create binary images for edge detection, prepare for feature extraction, binarize for pattern recognition), enabling feature detection workflows

## Connecting to Other Blocks

This block receives a grayscale image and produces a thresholded binary image:

- **After Image Convert Grayscale blocks** to convert color images to grayscale before thresholding (e.g., convert color to grayscale then threshold, prepare color images for binarization, grayscale before binary conversion), enabling color-to-binary workflows
- **After preprocessing blocks** that output grayscale images (e.g., apply thresholding after filtering, binarize after enhancement, threshold preprocessed images), enabling preprocessing-to-threshold workflows
- **Before morphological transformation blocks** to prepare binary images for morphological operations (e.g., clean thresholded images with morphology, apply morphology to binary images, process binary masks), enabling threshold-to-morphology workflows
- **Before contour detection blocks** to prepare binary images for contour detection (e.g., find contours in thresholded images, detect shapes in binary images, analyze binary object boundaries), enabling threshold-to-contour workflows
- **Before analysis blocks** that process binary images (e.g., analyze binary masks, process thresholded regions, work with binary object data), enabling threshold analysis workflows
- **In image processing pipelines** where thresholding is part of a larger binary image processing chain (e.g., binarize images in pipelines, create masks in workflows, process binary images in chains), enabling threshold processing pipeline workflows

## Requirements

This block requires grayscale input images. Color images must be converted to grayscale first using an Image Convert Grayscale block. For optimal results, use images with good contrast between foreground and background. Fixed threshold methods work best with uniform lighting, while adaptive methods handle non-uniform lighting better. Otsu's method works best with bimodal histograms (clear foreground/background separation).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/threshold@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `threshold_type` | `str` | Type of thresholding operation to apply: 'binary' (default, pixels above threshold become white, below become black) for uniform lighting, 'binary_inv' (inverse binary, pixels above threshold become black, below become white) for dark object isolation, 'trunc' (truncate, pixels above threshold set to threshold value) for brightness clipping, 'tozero' (to zero, pixels below threshold set to zero) for dark region removal, 'tozero_inv' (to zero inverse, pixels above threshold set to zero) for bright region removal, 'adaptive_mean' (adaptive mean, local threshold using mean of neighborhood) for non-uniform lighting, 'adaptive_gaussian' (adaptive Gaussian, local threshold using weighted Gaussian mean) for non-uniform lighting with smoother transitions, or 'otsu' (Otsu's method, automatic optimal threshold calculation) for bimodal histograms. Default is 'binary'. Fixed methods (binary, binary_inv, trunc, tozero, tozero_inv) use thresh_value, adaptive methods (adaptive_mean, adaptive_gaussian) adapt locally, and Otsu's method automatically calculates the optimal threshold.. | ✅ |
| `thresh_value` | `int` | Threshold value used for fixed threshold methods (binary, binary_inv, trunc, tozero, tozero_inv). Must be an integer between 0 and 255. Pixels above this value are treated as foreground for binary/binary_inv, or clipped/preserved for trunc/tozero/tozero_inv depending on the operation. Typical values range from 100-200: lower values (100-127) for darker images or to preserve more dark regions, medium values (127-150) for balanced separation, higher values (150-200) for brighter images or to preserve more bright regions. Default is 127 (middle gray). This parameter is ignored for adaptive methods (adaptive_mean, adaptive_gaussian) and Otsu's method (otsu) which calculate thresholds automatically. Adjust based on image brightness and desired foreground/background separation.. | ✅ |
| `max_value` | `int` | Maximum value used in thresholded binary outputs. Must be an integer, typically 255 (white pixel value in 8-bit images). This value is assigned to pixels classified as foreground in binary operations (binary, binary_inv) or used as the maximum value in adaptive thresholding (adaptive_mean, adaptive_gaussian, otsu). For truncate operations (trunc), this parameter is used but the actual output values are clipped to thresh_value. Default is 255, which creates standard binary images with white (255) foreground and black (0) background. For 8-bit grayscale images, keep at 255. For 16-bit images, use 65535. Controls the brightness/intensity of foreground pixels in the thresholded output.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Threshold` in version `v1`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Preprocessing`](image_preprocessing.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Color Visualization`](color_visualization.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Relative Static Crop`](relative_static_crop.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Icon Visualization`](icon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Contours`](image_contours.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Mask Visualization`](mask_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI`](open_ai.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Barcode Detection`](barcode_detection.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Threshold` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input grayscale image to apply thresholding to. Must be a single-channel grayscale image (color images must be converted to grayscale first using an Image Convert Grayscale block). Thresholding converts grayscale images to binary images where pixels are classified as foreground (white) or background (black). The thresholded binary image will have pixels set to either 0 (black) or max_value (typically 255, white) based on the selected thresholding method. Original image metadata is preserved in the output. For optimal results, use images with good contrast between foreground and background..
        - `threshold_type` (*[`string`](../kinds/string.md)*): Type of thresholding operation to apply: 'binary' (default, pixels above threshold become white, below become black) for uniform lighting, 'binary_inv' (inverse binary, pixels above threshold become black, below become white) for dark object isolation, 'trunc' (truncate, pixels above threshold set to threshold value) for brightness clipping, 'tozero' (to zero, pixels below threshold set to zero) for dark region removal, 'tozero_inv' (to zero inverse, pixels above threshold set to zero) for bright region removal, 'adaptive_mean' (adaptive mean, local threshold using mean of neighborhood) for non-uniform lighting, 'adaptive_gaussian' (adaptive Gaussian, local threshold using weighted Gaussian mean) for non-uniform lighting with smoother transitions, or 'otsu' (Otsu's method, automatic optimal threshold calculation) for bimodal histograms. Default is 'binary'. Fixed methods (binary, binary_inv, trunc, tozero, tozero_inv) use thresh_value, adaptive methods (adaptive_mean, adaptive_gaussian) adapt locally, and Otsu's method automatically calculates the optimal threshold..
        - `thresh_value` (*[`integer`](../kinds/integer.md)*): Threshold value used for fixed threshold methods (binary, binary_inv, trunc, tozero, tozero_inv). Must be an integer between 0 and 255. Pixels above this value are treated as foreground for binary/binary_inv, or clipped/preserved for trunc/tozero/tozero_inv depending on the operation. Typical values range from 100-200: lower values (100-127) for darker images or to preserve more dark regions, medium values (127-150) for balanced separation, higher values (150-200) for brighter images or to preserve more bright regions. Default is 127 (middle gray). This parameter is ignored for adaptive methods (adaptive_mean, adaptive_gaussian) and Otsu's method (otsu) which calculate thresholds automatically. Adjust based on image brightness and desired foreground/background separation..
        - `max_value` (*[`integer`](../kinds/integer.md)*): Maximum value used in thresholded binary outputs. Must be an integer, typically 255 (white pixel value in 8-bit images). This value is assigned to pixels classified as foreground in binary operations (binary, binary_inv) or used as the maximum value in adaptive thresholding (adaptive_mean, adaptive_gaussian, otsu). For truncate operations (trunc), this parameter is used but the actual output values are clipped to thresh_value. Default is 255, which creates standard binary images with white (255) foreground and black (0) background. For 8-bit grayscale images, keep at 255. For 16-bit images, use 65535. Controls the brightness/intensity of foreground pixels in the thresholded output..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Threshold` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/threshold@v1",
	    "image": "$inputs.image",
	    "threshold_type": "binary",
	    "thresh_value": 127,
	    "max_value": 255
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

