
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

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CogVLM`](cog_vlm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Email Notification`](email_notification.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Current Time`](current_time.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`SIFT`](sift.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`OCR Model`](ocr_model.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Relative Static Crop`](relative_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Qwen-VL`](qwen_vl.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PP-OCR`](ppocr.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Local File Sink`](local_file_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemma`](google_gemma.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Polygon Visualization`](polygon_visualization.md), [`Cosmos 3`](cosmos3.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Icon Visualization`](icon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Camera Calibration`](camera_calibration.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Perspective Correction`](perspective_correction.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
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

