
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
| `kernel_size` | `int` | Size of the blur kernel (must be positive and typically odd). Controls the blur intensity - larger values create more blur, smaller values create less blur. For average and gaussian blur, this is the width and height of the kernel (e.g., 5 means 5x5 kernel). For gaussian and median blur, the kernel size must be a positive odd integer; even or non-positive values are automatically coerced to the next positive odd value. For bilateral blur, this controls the diameter of the pixel neighborhood. Typical values range from 3-15: smaller values (3-5) provide subtle blur, medium values (5-9) provide moderate blur, larger values (11-15) provide strong blur. Default is 5, which provides moderate blur. Adjust based on image size and desired blur intensity.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Blur` in version `v1`.

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CogVLM`](cog_vlm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Email Notification`](email_notification.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`SIFT`](sift.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`OCR Model`](ocr_model.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Relative Static Crop`](relative_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Qwen-VL`](qwen_vl.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Distance Measurement`](distance_measurement.md), [`GLM-OCR`](glmocr.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PP-OCR`](ppocr.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Local File Sink`](local_file_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemma`](google_gemma.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Polygon Visualization`](polygon_visualization.md), [`Cosmos 3`](cosmos3.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Icon Visualization`](icon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Camera Calibration`](camera_calibration.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Perspective Correction`](perspective_correction.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Blur` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to apply blur filtering to. The block will apply the specified blur type with the configured kernel size. Works on color or grayscale images. The blurred image will have reduced detail, smoothed textures, and noise reduction depending on the blur type and kernel size selected. Original image metadata is preserved in the output..
        - `blur_type` (*[`string`](../kinds/string.md)*): Type of blur algorithm to apply: 'average' uses simple box filter for uniform blur (fast, basic smoothing), 'gaussian' (default) uses Gaussian-weighted kernel for smooth natural blur (high-quality, preserves structure), 'median' uses nonlinear median filter for noise removal while preserving edges (excellent for impulse noise, salt-and-pepper noise), or 'bilateral' uses edge-preserving filter that blurs similar colors while maintaining sharp edges (good for denoising with edge preservation). Default is 'gaussian' which provides good general-purpose blurring. Choose based on requirements: average for speed, gaussian for quality, median for noise removal, bilateral for edge preservation..
        - `kernel_size` (*[`integer`](../kinds/integer.md)*): Size of the blur kernel (must be positive and typically odd). Controls the blur intensity - larger values create more blur, smaller values create less blur. For average and gaussian blur, this is the width and height of the kernel (e.g., 5 means 5x5 kernel). For gaussian and median blur, the kernel size must be a positive odd integer; even or non-positive values are automatically coerced to the next positive odd value. For bilateral blur, this controls the diameter of the pixel neighborhood. Typical values range from 3-15: smaller values (3-5) provide subtle blur, medium values (5-9) provide moderate blur, larger values (11-15) provide strong blur. Default is 5, which provides moderate blur. Adjust based on image size and desired blur intensity..

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

