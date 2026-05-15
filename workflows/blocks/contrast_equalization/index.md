
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

    - inputs: [`Perspective Correction`](perspective_correction.md), [`S3 Sink`](s3_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen-VL`](qwen_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Dynamic Crop`](dynamic_crop.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Local File Sink`](local_file_sink.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OCR Model`](ocr_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Blur Visualization`](blur_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Text Display`](text_display.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT`](sift.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
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

