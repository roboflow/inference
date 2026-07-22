
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

    - inputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Email Notification`](email_notification.md), [`Distance Measurement`](distance_measurement.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`LMM`](lmm.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Image Slicer`](image_slicer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Cosmos 3`](cosmos3.md), [`CogVLM`](cog_vlm.md), [`Polygon Visualization`](polygon_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Crop Visualization`](crop_visualization.md), [`PP-OCR`](ppocr.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`PLC Writer`](plc_writer.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`S3 Sink`](s3_sink.md), [`Event Writer`](event_writer.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

