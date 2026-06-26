
# Image Blur



??? "Class: `ImageBlurBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/image_blur/v1.py">inference.core.workflows.core_steps.classical_cv.image_blur.v1.ImageBlurBlockV1</a>
    



Apply configurable blur filters to images using different blur algorithms (average, Gaussian, median, or bilateral), smoothing image details, reducing noise, and creating blur effects for noise reduction, privacy protection, preprocessing, and image enhancement workflows.

## How This Block Works

This block applies blur filtering to images using one of four blur algorithms, each with different characteristics and use cases. The block:

1. Receives an input image to apply blur filtering to
2. Selects the blur algorithm based on blur_type parameter
3. Applies the selected blur method using the specified kernel_size:

   **For Average Blur:**
   - Uses a simple box filter that replaces each pixel with the average of its neighbors
   - Creates uniform blur across all pixels within the kernel area
   - Fast and simple blurring suitable for general smoothing
   - Good for basic noise reduction and smoothing

   **For Gaussian Blur:**
   - Uses a Gaussian-weighted kernel that applies more weight to pixels closer to the center
   - Creates smooth, natural-looking blur with gradual falloff from center
   - Provides high-quality blurring that preserves image structure better than average blur
   - Good for general-purpose blurring, noise reduction, and preprocessing

   **For Median Blur:**
   - Uses a nonlinear filter that replaces each pixel with the median value of its neighbors
   - Particularly effective at removing salt-and-pepper noise while preserving edges
   - Better at preserving sharp edges than linear blur methods
   - Good for noise reduction in images with impulse noise, speckle noise, or artifacts

   **For Bilateral Blur:**
   - Uses a nonlinear filter that blurs while preserving edges
   - Combines spatial smoothing with intensity similarity (blurs similar colors, preserves edges between different colors)
   - Reduces noise and smooths textures while maintaining sharp edges and boundaries
   - Good for noise reduction when edge preservation is important, image denoising, and detail smoothing

4. Preserves image metadata from the original image
5. Returns the blurred image with applied blur filtering

The kernel_size parameter controls the blur intensity - larger values create more blur, smaller values create less blur. Different blur types have different characteristics: Average and Gaussian provide general smoothing, Median is excellent for noise removal, and Bilateral preserves edges while smoothing. The choice of blur type depends on the specific requirements - general smoothing, noise reduction, edge preservation, or artifact removal.

## Common Use Cases

- **Noise Reduction**: Reduce image noise and artifacts using blur filtering (e.g., remove noise from camera images, reduce compression artifacts, smooth out image imperfections), enabling noise reduction workflows
- **Privacy Protection**: Blur sensitive regions or faces in images (e.g., blur faces for privacy, obscure sensitive information, anonymize image content), enabling privacy protection workflows
- **Image Preprocessing**: Smooth images before further processing or analysis (e.g., preprocess images before detection, smooth images before analysis, reduce noise before processing), enabling preprocessing workflows
- **Detail Smoothing**: Smooth fine details and textures in images (e.g., smooth skin in portraits, reduce texture detail, create softer appearance), enabling detail smoothing workflows
- **Artifact Removal**: Remove artifacts and imperfections from images (e.g., remove compression artifacts, reduce JPEG artifacts, smooth out image defects), enabling artifact removal workflows
- **Background Blurring**: Create depth-of-field effects or blur backgrounds (e.g., blur backgrounds for focus effects, create bokeh effects, emphasize foreground subjects), enabling background blurring workflows

## Connecting to Other Blocks

This block receives an image and produces a blurred image:

- **After image input blocks** to blur input images before further processing (e.g., blur images from camera feeds, reduce noise in image inputs, preprocess images for workflows), enabling image blurring workflows
- **Before detection or classification models** to preprocess images with noise reduction (e.g., reduce noise before object detection, smooth images before classification, preprocess images for model input), enabling preprocessed model input workflows
- **After preprocessing blocks** to apply blur after other preprocessing steps (e.g., blur after filtering, smooth after enhancement, reduce artifacts after processing), enabling multi-stage preprocessing workflows
- **Before visualization blocks** to display blurred images (e.g., visualize privacy-protected images, display smoothed images, show blur effects), enabling blurred image visualization workflows
- **In privacy protection workflows** where sensitive regions need to be blurred (e.g., blur faces in privacy workflows, obscure sensitive content, anonymize image data), enabling privacy protection workflows
- **In noise reduction pipelines** where blur is part of a larger denoising workflow (e.g., reduce noise in multi-stage pipelines, apply blur for artifact removal, smooth images in processing chains), enabling noise reduction pipeline workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/image_blur@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `blur_type` | `str` | Type of blur algorithm to apply: 'average' uses simple box filter for uniform blur (fast, basic smoothing), 'gaussian' (default) uses Gaussian-weighted kernel for smooth natural blur (high-quality, preserves structure), 'median' uses nonlinear median filter for noise removal while preserving edges (excellent for impulse noise, salt-and-pepper noise), or 'bilateral' uses edge-preserving filter that blurs similar colors while maintaining sharp edges (good for denoising with edge preservation). Default is 'gaussian' which provides good general-purpose blurring. Choose based on requirements: average for speed, gaussian for quality, median for noise removal, bilateral for edge preservation.. | ✅ |
| `kernel_size` | `int` | Size of the blur kernel (must be positive and typically odd). Controls the blur intensity - larger values create more blur, smaller values create less blur. For average and gaussian blur, this is the width and height of the kernel (e.g., 5 means 5x5 kernel). For median blur, this must be an odd integer (automatically handled). For bilateral blur, this controls the diameter of the pixel neighborhood. Typical values range from 3-15: smaller values (3-5) provide subtle blur, medium values (5-9) provide moderate blur, larger values (11-15) provide strong blur. Default is 5, which provides moderate blur. Adjust based on image size and desired blur intensity.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Blur` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Template Matching`](template_matching.md), [`LMM For Classification`](lmm_for_classification.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Stack`](image_stack.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detection Event Log`](detection_event_log.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`OCR Model`](ocr_model.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Blur` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to apply blur filtering to. The block will apply the specified blur type with the configured kernel size. Works on color or grayscale images. The blurred image will have reduced detail, smoothed textures, and noise reduction depending on the blur type and kernel size selected. Original image metadata is preserved in the output..
        - `blur_type` (*[`string`](../kinds/string.md)*): Type of blur algorithm to apply: 'average' uses simple box filter for uniform blur (fast, basic smoothing), 'gaussian' (default) uses Gaussian-weighted kernel for smooth natural blur (high-quality, preserves structure), 'median' uses nonlinear median filter for noise removal while preserving edges (excellent for impulse noise, salt-and-pepper noise), or 'bilateral' uses edge-preserving filter that blurs similar colors while maintaining sharp edges (good for denoising with edge preservation). Default is 'gaussian' which provides good general-purpose blurring. Choose based on requirements: average for speed, gaussian for quality, median for noise removal, bilateral for edge preservation..
        - `kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the blur kernel (must be positive and typically odd). Controls the blur intensity - larger values create more blur, smaller values create less blur. For average and gaussian blur, this is the width and height of the kernel (e.g., 5 means 5x5 kernel). For median blur, this must be an odd integer (automatically handled). For bilateral blur, this controls the diameter of the pixel neighborhood. Typical values range from 3-15: smaller values (3-5) provide subtle blur, medium values (5-9) provide moderate blur, larger values (11-15) provide strong blur. Default is 5, which provides moderate blur. Adjust based on image size and desired blur intensity..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Blur` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/image_blur@v1",
	    "image": "$inputs.image",
	    "blur_type": "gaussian",
	    "kernel_size": 5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

