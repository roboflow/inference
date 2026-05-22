
# Image Preprocessing



??? "Class: `ImagePreprocessingBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/image_preprocessing/v1.py">inference.core.workflows.core_steps.classical_cv.image_preprocessing.v1.ImagePreprocessingBlockV1</a>
    



Apply geometric transformations to images including resizing to specified dimensions (with aspect ratio preservation), rotating by specified degrees (clockwise or counterclockwise), or flipping vertically, horizontally, or both, providing flexible image preprocessing for model input preparation, image orientation correction, and geometric image manipulation workflows.

## How This Block Works

This block applies one geometric transformation operation (resize, rotate, or flip) to an input image based on the selected task_type. The block:

1. Receives an input image and selects one transformation task (resize, rotate, or flip)
2. Validates task-specific parameters (width/height for resize, rotation_degrees for rotate, flip_type for flip)
3. Applies the selected transformation:

   **For resize task:**
   - Validates width and height are positive integers (greater than 0)
   - Supports aspect ratio preservation: if only width or only height is provided, calculates the missing dimension to maintain the original aspect ratio
   - If both width and height are provided, resizes to exact dimensions (may distort aspect ratio)
   - Uses OpenCV's INTER_AREA interpolation for high-quality downsampling
   - Returns resized image with specified dimensions

   **For rotate task:**
   - Validates rotation_degrees is between -360 and 360 degrees
   - Positive values rotate clockwise, negative values rotate counterclockwise
   - Calculates rotation matrix around image center
   - Automatically adjusts canvas size to contain the rotated image (no cropping)
   - Uses OpenCV's warpAffine for smooth rotation with bilinear interpolation
   - Returns rotated image with canvas sized to fit the full rotated image

   **For flip task:**
   - Validates flip_type is "vertical", "horizontal", or "both"
   - Vertical flip: flips image upside down (mirrors along horizontal axis)
   - Horizontal flip: flips image left-right (mirrors along vertical axis)
   - Both: applies both vertical and horizontal flips simultaneously (180-degree rotation equivalent)
   - Uses OpenCV's flip function for efficient mirroring
   - Returns flipped image with same dimensions as input

4. Preserves image metadata from the original image (parent metadata, image properties)
5. Returns the transformed image maintaining original image metadata structure

The block performs one transformation at a time - select resize, rotate, or flip via task_type. Each transformation is applied independently and produces a clean output. Resize supports flexible aspect ratio handling, rotation automatically adjusts canvas size to prevent cropping, and flip operations provide efficient mirroring along different axes. The transformations use OpenCV for efficient, high-quality geometric image manipulation.

## Common Use Cases

- **Model Input Preparation**: Resize images to match model input requirements (e.g., resize images to specific dimensions for object detection models, adjust image sizes for classification model inputs, normalize image dimensions for consistent model processing), enabling proper model input formatting
- **Image Orientation Correction**: Rotate images to correct orientation issues (e.g., rotate images captured in wrong orientation, correct camera rotation, adjust image orientation for proper display), enabling image orientation workflows
- **Data Augmentation**: Apply geometric transformations for data augmentation (e.g., flip images horizontally for augmentation, rotate images for training data variety, apply transformations to increase dataset diversity), enabling data augmentation workflows
- **Image Display Preparation**: Transform images for display or presentation purposes (e.g., flip images for mirror effects, resize images for display dimensions, rotate images for correct viewing orientation), enabling image presentation workflows
- **Workflow Image Standardization**: Standardize image dimensions or orientation across workflow inputs (e.g., resize all images to consistent dimensions, normalize image orientations, prepare images for uniform processing), enabling image standardization workflows
- **Image Formatting for Downstream Blocks**: Prepare images for blocks that require specific dimensions or orientations (e.g., resize before detection models, rotate for proper processing, flip for compatibility with other blocks), enabling image preparation workflows

## Connecting to Other Blocks

This block receives an image and produces a transformed image:

- **After image input blocks** to preprocess images before further processing (e.g., resize input images, correct image orientation, prepare images for workflow processing), enabling image preprocessing workflows
- **Before detection or classification models** to format images for model requirements (e.g., resize to model input dimensions, adjust orientation for proper detection, prepare images for model processing), enabling model-compatible image preparation
- **Before crop blocks** to prepare images before cropping (e.g., resize before cropping, rotate before region extraction, adjust orientation before cropping), enabling pre-crop image preparation
- **Before visualization blocks** to prepare images for display (e.g., resize for display, rotate for proper viewing, flip for presentation), enabling image display preparation workflows
- **In image processing pipelines** where geometric transformations are needed (e.g., resize in multi-stage pipelines, rotate in processing workflows, flip in transformation chains), enabling geometric transformation pipelines
- **After other transformation blocks** to apply additional geometric operations (e.g., resize after cropping, rotate after other transformations, flip after processing), enabling multi-stage geometric transformation workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/image_preprocessing@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Type of geometric transformation to apply to the image: 'resize' to change image dimensions (requires width/height), 'rotate' to rotate the image by specified degrees (requires rotation_degrees), or 'flip' to mirror the image along axes (requires flip_type). Only one transformation is applied per block execution. Select the appropriate task type based on your preprocessing needs.. | ❌ |
| `width` | `int` | Target width in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only width is provided (height is None), the height is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements.. | ✅ |
| `height` | `int` | Target height in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only height is provided (width is None), the width is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements.. | ✅ |
| `rotation_degrees` | `int` | Rotation angle in degrees. Required when task_type is 'rotate'. Must be between -360 and 360 degrees. Positive values rotate the image clockwise, negative values rotate counterclockwise. The rotation is performed around the image center, and the canvas size is automatically adjusted to contain the full rotated image (no cropping occurs). For example, 90 rotates 90 degrees clockwise, -90 rotates 90 degrees counterclockwise, 180 rotates 180 degrees. Default is 90 degrees.. | ✅ |
| `flip_type` | `str` | Type of flip operation to apply. Required when task_type is 'flip'. Options: 'vertical' flips the image upside down (mirrors along horizontal axis, top becomes bottom), 'horizontal' flips left-right (mirrors along vertical axis, left becomes right), 'both' applies both vertical and horizontal flips simultaneously (equivalent to 180-degree rotation). The image dimensions remain unchanged after flipping. Default is 'vertical'. Use this for mirroring images or data augmentation.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Preprocessing` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Crop Visualization`](crop_visualization.md), [`Camera Focus`](camera_focus.md), [`Blur Visualization`](blur_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen-VL`](qwen_vl.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`Email Notification`](email_notification.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Detection Event Log`](detection_event_log.md), [`CogVLM`](cog_vlm.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Vision OCR`](google_vision_ocr.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Distance Measurement`](distance_measurement.md), [`OpenRouter`](open_router.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`Florence-2 Model`](florence2_model.md), [`Label Visualization`](label_visualization.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Calibration`](camera_calibration.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM`](lmm.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OCR Model`](ocr_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CSV Formatter`](csv_formatter.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)
    - outputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Seg Preview`](seg_preview.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Blur`](image_blur.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Buffer`](buffer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`LMM`](lmm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Object Detection Model`](object_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen3.5`](qwen3.5.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`CogVLM`](cog_vlm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Moondream2`](moondream2.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Template Matching`](template_matching.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`QR Code Detection`](qr_code_detection.md), [`Label Visualization`](label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Circle Visualization`](circle_visualization.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Preprocessing` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to transform. The image will have one geometric transformation applied (resize, rotate, or flip) based on the selected task_type. Supports images from inputs, previous workflow steps, or crop outputs. The output image maintains the original image's metadata structure..
        - `width` (*[`integer`](../kinds/integer.md)*): Target width in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only width is provided (height is None), the height is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements..
        - `height` (*[`integer`](../kinds/integer.md)*): Target height in pixels for resizing. Required when task_type is 'resize'. Must be a positive integer (greater than 0). If only height is provided (width is None), the width is automatically calculated to preserve aspect ratio. If both width and height are provided, the image is resized to exact dimensions (may distort aspect ratio). Default is 640 pixels. Use this to resize images to specific dimensions for model inputs or display requirements..
        - `rotation_degrees` (*[`integer`](../kinds/integer.md)*): Rotation angle in degrees. Required when task_type is 'rotate'. Must be between -360 and 360 degrees. Positive values rotate the image clockwise, negative values rotate counterclockwise. The rotation is performed around the image center, and the canvas size is automatically adjusted to contain the full rotated image (no cropping occurs). For example, 90 rotates 90 degrees clockwise, -90 rotates 90 degrees counterclockwise, 180 rotates 180 degrees. Default is 90 degrees..
        - `flip_type` (*[`string`](../kinds/string.md)*): Type of flip operation to apply. Required when task_type is 'flip'. Options: 'vertical' flips the image upside down (mirrors along horizontal axis, top becomes bottom), 'horizontal' flips left-right (mirrors along vertical axis, left becomes right), 'both' applies both vertical and horizontal flips simultaneously (equivalent to 180-degree rotation). The image dimensions remain unchanged after flipping. Default is 'vertical'. Use this for mirroring images or data augmentation..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Preprocessing` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/image_preprocessing@v1",
	    "image": "$inputs.image",
	    "task_type": "<block_does_not_provide_example>",
	    "width": 640,
	    "height": 640,
	    "rotation_degrees": 90,
	    "flip_type": "vertical"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

