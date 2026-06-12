
# Image Convert Grayscale



??? "Class: `ConvertGrayscaleBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/convert_grayscale/v1.py">inference.core.workflows.core_steps.classical_cv.convert_grayscale.v1.ConvertGrayscaleBlockV1</a>
    



Convert color (RGB/BGR) images to single-channel grayscale images using weighted luminance conversion to reduce dimensionality, prepare images for operations that require grayscale input (thresholding, morphological operations, contour detection), reduce computational complexity, and enable intensity-based image analysis and processing workflows.

## How This Block Works

This block converts color images to grayscale images by combining the color channels into a single intensity channel. The block:

1. Receives a color input image (RGB or BGR format with three color channels)
2. Applies weighted luminance conversion using OpenCV's BGR to grayscale algorithm:
   - Uses the standard formula: Grayscale = 0.299*R + 0.587*G + 0.114*B (for RGB) or weighted BGR combination
   - Weights green channel most heavily (58.7%) as human eye is most sensitive to green
   - Weights red channel moderately (29.9%) and blue channel least (11.4%)
   - Creates a perceptually balanced grayscale representation that preserves visual information
3. Converts the three-channel color image to a single-channel grayscale image:
   - Reduces image from 3 channels (RGB/BGR) to 1 channel (grayscale)
   - Each pixel becomes a single intensity value between 0 (black) and 255 (white)
   - Preserves spatial information while removing color information
   - Reduces memory usage and computational complexity
4. Preserves image dimensions (width and height remain the same)
5. Maintains image metadata and structure
6. Returns the single-channel grayscale image

Grayscale conversion transforms color images into intensity-only images where each pixel represents brightness rather than color. The weighted luminance formula ensures the grayscale image perceptually matches the brightness distribution of the original color image. This conversion is essential for many computer vision operations that require single-channel input, such as thresholding, morphological transformations, edge detection, and contour analysis. The output retains spatial information and intensity relationships while removing color information, enabling intensity-based processing and analysis.

## Common Use Cases

- **Preprocessing for Thresholding**: Convert color images to grayscale before applying thresholding operations (e.g., prepare images for binary thresholding, convert before adaptive thresholding, grayscale before Otsu's method), enabling color-to-threshold workflows
- **Morphological Operations**: Prepare color images for morphological transformations that require grayscale input (e.g., convert before erosion/dilation, grayscale for opening/closing, prepare for morphological operations), enabling color-to-morphology workflows
- **Contour Detection**: Convert color images to grayscale before contour detection and shape analysis (e.g., prepare for contour detection, convert before shape analysis, grayscale for boundary extraction), enabling color-to-contour workflows
- **Edge Detection**: Prepare color images for edge detection algorithms that work on grayscale images (e.g., convert before Canny edge detection, grayscale for Sobel operators, prepare for edge detection), enabling color-to-edge workflows
- **Noise Reduction**: Reduce dimensionality for noise reduction operations that work on single-channel images (e.g., convert before filtering, grayscale for denoising, prepare for noise reduction), enabling color-to-filtering workflows
- **Feature Extraction**: Convert color images to grayscale for intensity-based feature extraction (e.g., prepare for SIFT/keypoint detection, convert for texture analysis, grayscale for pattern recognition), enabling color-to-feature workflows

## Connecting to Other Blocks

This block receives a color image and produces a grayscale image:

- **Before threshold blocks** to convert color images to grayscale before thresholding (e.g., convert color to grayscale then threshold, prepare color images for binarization, grayscale before binary conversion), enabling color-to-threshold workflows
- **Before morphological transformation blocks** to prepare color images for morphological operations (e.g., convert color to grayscale for morphology, prepare for erosion/dilation, grayscale before morphological operations), enabling color-to-morphology workflows
- **Before contour detection blocks** to convert color images to grayscale before contour detection (e.g., convert color to grayscale for contours, prepare color images for shape analysis, grayscale before contour detection), enabling color-to-contour workflows
- **Before classical CV blocks** that require grayscale input (e.g., prepare for edge detection, convert for feature extraction, grayscale for classical computer vision operations), enabling color-to-classical-CV workflows
- **After image preprocessing blocks** that output color images (e.g., convert preprocessed color images to grayscale, grayscale after color enhancements, convert after color transformations), enabling preprocessing-to-grayscale workflows
- **In image processing pipelines** where grayscale conversion is required for downstream processing (e.g., convert color to grayscale in pipelines, prepare images for single-channel operations, reduce dimensionality for processing), enabling grayscale conversion pipeline workflows

## Requirements

This block works on color images (RGB or BGR format with three color channels). The input image must have multiple color channels. If the input is already grayscale, the conversion will still be applied but will result in the same grayscale output. The conversion uses standard luminance weighting to create perceptually balanced grayscale images that preserve brightness information while removing color information.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/convert_grayscale@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Convert Grayscale` in version `v1`.

    - inputs: [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Threshold`](image_threshold.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch Images`](stitch_images.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Blur`](image_blur.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT`](sift.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Grid Visualization`](grid_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Camera Focus`](camera_focus.md), [`Contrast Equalization`](contrast_equalization.md), [`Circle Visualization`](circle_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Relative Static Crop`](relative_static_crop.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Background Color Visualization`](background_color_visualization.md)
    - outputs: [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen-VL`](qwen_vl.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Camera Focus`](camera_focus.md), [`SORT Tracker`](sort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Event Writer`](event_writer.md), [`Buffer`](buffer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Dominant Color`](dominant_color.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`OpenRouter`](open_router.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Halo Visualization`](halo_visualization.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SIFT`](sift.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Google Gemini`](google_gemini.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Barcode Detection`](barcode_detection.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Image Contours`](image_contours.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Qwen3.5`](qwen3.5.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Moondream2`](moondream2.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Convert Grayscale` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input color image (RGB or BGR format with three color channels) to convert to grayscale. The image will be converted from three-channel color (RGB/BGR) to single-channel grayscale using weighted luminance conversion (weights: green 58.7%, red 29.9%, blue 11.4%) to create a perceptually balanced grayscale representation. The output will have the same width and height but only one channel (grayscale intensity values 0-255). Original image metadata and spatial dimensions are preserved. If the input is already grayscale, the conversion will still be applied but will result in the same grayscale output. Use this block before operations that require grayscale input such as thresholding, morphological operations, contour detection, or edge detection..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Image Convert Grayscale` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/convert_grayscale@v1",
	    "image": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

