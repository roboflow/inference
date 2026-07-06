
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

    - inputs: [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen-VL`](qwen_vl.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Google Gemini`](google_gemini.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`GLM-OCR`](glmocr.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Slack Notification`](slack_notification.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Dot Visualization`](dot_visualization.md), [`Current Time`](current_time.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`Dominant Color`](dominant_color.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
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

