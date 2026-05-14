
# Contrast Equalization



??? "Class: `ContrastEqualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/contrast_equalization/v1.py">inference.core.workflows.core_steps.classical_cv.contrast_equalization.v1.ContrastEqualizationBlockV1</a>
    



Enhance image contrast using configurable equalization methods (Contrast Stretching, Histogram Equalization, or Adaptive Equalization) to improve image visibility, distribute pixel intensities more evenly, and enhance details in low-contrast or poorly lit images for preprocessing, enhancement, and quality improvement workflows.

## How This Block Works

This block enhances image contrast by redistributing pixel intensities using one of three equalization methods. The block:

1. Receives an input image to enhance with contrast equalization
2. Selects the contrast equalization method based on equalization_type parameter
3. Applies the selected equalization method:

   **For Contrast Stretching:**
   - Calculates the 2nd and 98th percentiles of pixel intensities in the image (finds the darkest and brightest meaningful values, ignoring extreme outliers)
   - Stretches the intensity range between these percentiles to span the full 0-255 range
   - Enhances contrast by expanding the dynamic range while preserving relative intensity relationships
   - Useful for images with a narrow intensity range that need stretching to full range

   **For Histogram Equalization:**
   - Normalizes pixel intensities to 0-1 range for processing
   - Computes and equalizes the image histogram to create a uniform distribution of pixel intensities
   - Redistributes pixel values so that each intensity level has approximately equal frequency
   - Scales the equalized values back to 0-255 range
   - Enhances contrast globally across the entire image, improving visibility of features

   **For Adaptive Equalization:**
   - Normalizes pixel intensities to 0-1 range for processing
   - Applies adaptive histogram equalization (CLAHE - Contrast Limited Adaptive Histogram Equalization)
   - Divides the image into small regions and equalizes each region independently
   - Uses clip_limit=0.03 to limit contrast enhancement and prevent over-amplification of noise
   - Combines local equalized regions using bilinear interpolation for smooth transitions
   - Scales the result back to 0-255 range
   - Enhances contrast adaptively, preserving local details while improving overall visibility

4. Preserves image metadata from the original image
5. Returns the enhanced image with improved contrast

The block provides three methods with different characteristics: Contrast Stretching expands intensity ranges linearly, Histogram Equalization creates uniform intensity distribution globally, and Adaptive Equalization enhances contrast locally while preventing over-amplification. Each method works best for different scenarios - Contrast Stretching for images with narrow intensity ranges, Histogram Equalization for overall contrast improvement, and Adaptive Equalization for images with varying contrast across regions.

## Common Use Cases

- **Image Preprocessing for Models**: Enhance image contrast before feeding to detection or classification models (e.g., improve contrast before object detection, enhance visibility before classification, prepare images for model processing), enabling improved model performance workflows
- **Low-Contrast Image Enhancement**: Improve visibility and details in low-contrast or poorly lit images (e.g., enhance dark images, improve visibility in low-light conditions, reveal details in low-contrast scenes), enabling image enhancement workflows
- **Detail Enhancement**: Reveal hidden details in images with poor contrast (e.g., enhance details in shadow regions, reveal features in dark areas, improve visibility of subtle details), enabling detail enhancement workflows
- **Image Quality Improvement**: Improve overall image quality and visibility (e.g., enhance overall image quality, improve visibility for analysis, optimize images for display), enabling image quality workflows
- **Medical and Scientific Imaging**: Enhance contrast in medical or scientific images for better analysis (e.g., enhance medical imaging contrast, improve scientific image visibility, prepare images for analysis), enabling scientific imaging workflows
- **Document Image Enhancement**: Improve contrast in scanned documents or document images (e.g., enhance document contrast, improve text visibility, optimize scanned documents), enabling document enhancement workflows

## Connecting to Other Blocks

This block receives an image and produces an enhanced image with improved contrast:

- **After image input blocks** to enhance input images before further processing (e.g., enhance contrast in camera feeds, improve visibility in image inputs, optimize images for workflow processing), enabling image enhancement workflows
- **Before detection or classification models** to improve model performance with better contrast (e.g., enhance images before object detection, improve visibility for classification models, prepare images for model analysis), enabling enhanced model input workflows
- **After preprocessing blocks** to apply contrast enhancement after other preprocessing (e.g., enhance contrast after filtering, improve visibility after transformations, optimize images after preprocessing), enabling multi-stage enhancement workflows
- **Before visualization blocks** to display enhanced images with better visibility (e.g., visualize enhanced images, display improved contrast results, show enhancement effects), enabling enhanced visualization workflows
- **Before analysis blocks** that benefit from improved contrast (e.g., analyze enhanced images, process improved visibility images, work with optimized contrast), enabling enhanced analysis workflows
- **In image quality improvement pipelines** where contrast enhancement is part of a larger enhancement workflow (e.g., enhance images in multi-stage pipelines, improve quality through enhancement steps, optimize images in processing chains), enabling image quality pipeline workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/contrast_equalization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `equalization_type` | `str` | Type of contrast equalization method to apply: 'Contrast Stretching' stretches the intensity range between 2nd and 98th percentiles to full 0-255 range (linear expansion, good for narrow intensity ranges), 'Histogram Equalization' (default) creates uniform intensity distribution globally (equalizes histogram across entire image, good for overall contrast improvement), or 'Adaptive Equalization' enhances contrast locally in small regions while limiting over-amplification (CLAHE with clip_limit=0.03, good for images with varying contrast). Default is 'Histogram Equalization' which provides good general-purpose contrast enhancement. Choose based on image characteristics and enhancement needs.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Contrast Equalization` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`GLM-OCR`](glmocr.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Dot Visualization`](dot_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`OCR Model`](ocr_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`EasyOCR`](easy_ocr.md), [`SAM 3`](sam3.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`CogVLM`](cog_vlm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter Visualization`](line_counter_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Contrast Equalization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to enhance with contrast equalization. The block applies one of three contrast equalization methods based on the equalization_type parameter. Works on color or grayscale images. The enhanced image will have improved contrast, better visibility, and enhanced details. Original image metadata is preserved in the output..
        - `equalization_type` (*[`string`](../kinds/string.md)*): Type of contrast equalization method to apply: 'Contrast Stretching' stretches the intensity range between 2nd and 98th percentiles to full 0-255 range (linear expansion, good for narrow intensity ranges), 'Histogram Equalization' (default) creates uniform intensity distribution globally (equalizes histogram across entire image, good for overall contrast improvement), or 'Adaptive Equalization' enhances contrast locally in small regions while limiting over-amplification (CLAHE with clip_limit=0.03, good for images with varying contrast). Default is 'Histogram Equalization' which provides good general-purpose contrast enhancement. Choose based on image characteristics and enhancement needs..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Contrast Equalization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/contrast_equalization@v1",
	    "image": "$inputs.image",
	    "equalization_type": "Histogram Equalization"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

