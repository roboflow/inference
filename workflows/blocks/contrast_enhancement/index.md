
# Contrast Enhancement



??? "Class: `ContrastEnhancementBlock`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/contrast_enhancement/v1.py">inference.core.workflows.core_steps.classical_cv.contrast_enhancement.v1.ContrastEnhancementBlock</a>
    



Enhance image contrast using histogram normalization (the algorithm from GIMP's Auto Levels). This block stretches the image histogram to use the full available range [0-255], improving visibility of features with low contrast.

## How This Block Works

1. **Channel Analysis**: For grayscale images, find min/max directly. For color images, analyze each channel independently.
2. **Histogram Normalization**: For each channel, stretch values from [min, max] to [0, 255] using linear scaling: `output = (input - min) / (max - min) * 255`
3. **Clipping**: Values outside [0, 255] are clipped to the valid range

## Common Use Cases

- **Low-contrast medical imaging**: Normalize tissue visibility across varying acquisition parameters
- **Industrial inspection**: Enhance subtle surface defects on dull materials
- **Surveillance footage**: Improve nighttime or backlit scene visibility
- **Document scanning**: Brighten poorly lit document photos
- **Microscopy**: Boost signal from weak fluorescence or phase-contrast images

## Input Parameters

**image** : Input image to enhance (color or grayscale)
- Can be single-channel, 3-channel (BGR), or 4-channel (BGRA)
- Each channel is normalized independently for color images

**clip_limit** : Percentage of histogram range to skip at extremes (default: 0)
- Range: 0-50
- 0: No clipping, entire histogram from min to max is used
- 1-3: Skip dark and bright outliers (robust to noise)
- 5-10: Very aggressive outlier removal
- 20-50: Extreme outlier removal, may lose subtle details

**contrast_multiplier** : Multiplier for contrast scaling after normalization (default: 1.0)
- Range: 0.1-5.0
- 1.0: No additional scaling, just normalization
- 0.5-0.9: Reduce contrast for smoother images
- 1.1-2.0: Increase contrast for more dramatic enhancement

**normalize_brightness** : Apply brightness normalization using midtone equalization (default: False)
- False: Only histogram normalization and contrast scaling
- True: After histogram normalization, apply midtone adjustment for balanced brightness

## Outputs

**image** : Enhanced image with normalized contrast, same shape and type as input

## Notes

- **Sensitive to outliers**: Extreme min/max values (single dark/bright pixels) stretch most of the histogram into a narrow range. Use morphological opening (Morphological Transformation v2) as preprocessing to remove spurious dark/bright specks.
- **Color shift**: For color images, each channel stretches independently, which can shift hue if channels have very different dynamic ranges
- **Efficiency**: Very fast — linear scan for min/max, then linear transformation per pixel
- **Brightness normalization**: When enabled, applies midtone stretch (gamma ≈ 1.3) for more balanced perceived brightness


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/contrast_enhancement@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `clip_limit` | `int` | Percentage of histogram range to skip at dark and bright extremes. 0: use full range from min to max. 1-3: skip outliers (robust). 5-10: very aggressive outlier removal.. | ❌ |
| `contrast_multiplier` | `float` | Multiplier for contrast scaling after normalization. 1.0: no additional scaling (just histogram normalization). <1.0: reduce contrast. >1.0: increase contrast for more dramatic enhancement.. | ❌ |
| `normalize_brightness` | `bool` | Apply brightness normalization using midtone equalization. When False, only histogram normalization and contrast scaling are applied. When True, applies midtone adjustment for more balanced perceived brightness.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Contrast Enhancement` in version `v1`.

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Calibration`](camera_calibration.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Threshold`](image_threshold.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Color Visualization`](color_visualization.md), [`Text Display`](text_display.md), [`Image Preprocessing`](image_preprocessing.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Visualization`](mask_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Heatmap Visualization`](heatmap_visualization.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Trace Visualization`](trace_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Detection`](qr_code_detection.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SmolVLM2`](smol_vlm2.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Gaze Detection`](gaze_detection.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Camera Focus`](camera_focus.md), [`Dominant Color`](dominant_color.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Relative Static Crop`](relative_static_crop.md), [`OCR Model`](ocr_model.md), [`Seg Preview`](seg_preview.md), [`Moondream2`](moondream2.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Google Gemini`](google_gemini.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`Image Threshold`](image_threshold.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Contrast Enhancement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image to enhance (color or grayscale). Each channel is normalized independently..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Contrast Enhancement` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/contrast_enhancement@v1",
	    "image": "$inputs.image",
	    "clip_limit": "<block_does_not_provide_example>",
	    "contrast_multiplier": "<block_does_not_provide_example>",
	    "normalize_brightness": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

