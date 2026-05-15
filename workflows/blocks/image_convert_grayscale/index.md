
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

    - inputs: [`Perspective Correction`](perspective_correction.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Morphological Transformation`](morphological_transformation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Mask Visualization`](mask_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Grid Visualization`](grid_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT`](sift.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
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

