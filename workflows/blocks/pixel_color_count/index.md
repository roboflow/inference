
# Pixel Color Count



??? "Class: `PixelationCountBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/pixel_color_count/v1.py">inference.core.workflows.core_steps.classical_cv.pixel_color_count.v1.PixelationCountBlockV1</a>
    



Count pixels in an image that match a target color within a specified tolerance range, using color range masking to identify matching pixels and returning the total count of pixels within the color tolerance range for color analysis, quality control, color-based measurements, and pixel-level color quantification workflows.

## How This Block Works

This block counts how many pixels in an image match a specific target color within a tolerance range, providing pixel-level color quantification. The block:

1. Receives an input image and a target color specification (hex string, RGB tuple string, or RGB tuple)
2. Converts the target color to BGR (Blue-Green-Red) format for OpenCV processing:
   - Parses hex color strings (e.g., "#431112" or "#412" shorthand)
   - Parses RGB tuple strings (e.g., "(128, 32, 64)")
   - Handles RGB tuples directly (e.g., (18, 17, 67))
   - Converts RGB to BGR format (reverses color channel order) since OpenCV uses BGR
3. Calculates color tolerance bounds:
   - Creates a lower bound by subtracting tolerance from each BGR channel of the target color
   - Creates an upper bound by adding tolerance to each BGR channel of the target color
   - Clips bounds to valid 0-255 range for each channel
   - Defines a 3D color cube in BGR space where matching pixels must fall
4. Creates a binary mask using OpenCV's inRange function:
   - Compares each pixel's BGR values against the lower and upper bounds
   - Sets mask pixel to 255 (white) if pixel color falls within the tolerance range
   - Sets mask pixel to 0 (black) if pixel color falls outside the tolerance range
   - Uses vectorized operations for efficient pixel-level comparison across the entire image
5. Counts matching pixels:
   - Counts non-zero pixels in the mask (pixels with value 255, representing matches)
   - Returns the total count of pixels that match the target color within tolerance

The block performs pixel-level color matching using a tolerance-based approach, allowing for slight color variations due to compression, lighting, or image processing. The tolerance creates a range around the target color - a tolerance of 10 means pixels can differ by up to ±10 in each BGR channel (for a total range of 21 values per channel). Lower tolerance values (e.g., 5-10) require very close color matches, while higher tolerance values (e.g., 20-30) allow more color variation. This is useful for counting pixels of a specific color when exact matches may not exist due to image artifacts or processing.

## Common Use Cases

- **Color Area Measurement**: Measure the area or coverage of specific colors in images (e.g., measure coverage of specific colors in images, quantify color distribution, assess color proportions), enabling color area quantification workflows
- **Quality Control and Inspection**: Count pixels of expected colors for quality control (e.g., verify color consistency in products, detect color defects, validate expected colors in images), enabling color-based quality control workflows
- **Color-Based Analysis**: Analyze images based on specific color presence or quantity (e.g., analyze color distribution in images, quantify color usage, measure color characteristics), enabling color quantification analysis workflows
- **Image Processing Validation**: Validate image processing results by counting expected colors (e.g., verify color transformations, validate color corrections, check color filtering results), enabling color validation workflows
- **Feature Detection and Measurement**: Detect and measure features based on color characteristics (e.g., count pixels in colored regions, measure color-based features, quantify color-defined areas), enabling color-based feature measurement workflows
- **Threshold-Based Color Detection**: Use pixel counting for threshold-based color detection (e.g., detect if enough pixels match a color, determine color presence thresholds, implement color-based triggers), enabling threshold-based color detection workflows

## Connecting to Other Blocks

This block receives an image and target color, and produces a pixel count:

- **After image input blocks** to count pixels of specific colors in input images (e.g., count color pixels in camera feeds, analyze colors in image inputs, quantify colors in images), enabling color pixel counting workflows
- **After crop blocks** to count pixels in specific image regions (e.g., count color pixels in cropped regions, analyze colors in specific areas, quantify colors in selected regions), enabling region-based color pixel counting
- **After preprocessing blocks** to count pixels after image processing (e.g., count colors after filtering, analyze colors after enhancement, quantify colors after transformations), enabling processed image color counting workflows
- **Before filtering or logic blocks** that use pixel counts for decision-making (e.g., filter based on pixel counts, make decisions based on color quantities, apply logic based on pixel counts), enabling count-based conditional workflows
- **Before data storage blocks** to store pixel count information (e.g., store color pixel counts with images, save color analysis results, record color quantification data), enabling color count metadata storage workflows
- **In quality control workflows** where pixel counting validates color characteristics (e.g., verify color quantities in quality control, validate color coverage, check color consistency), enabling color-based quality control workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/pixel_color_count@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `target_color` | `Union[Tuple[int, int, int], str]` | Target color to count in the image. Can be specified in multiple formats: (1) Hex string format: '#RRGGBB' (6-digit, e.g., '#431112') or '#RGB' (3-digit shorthand, e.g., '#412'), (2) RGB tuple string format: '(R, G, B)' (e.g., '(128, 32, 64)'), or (3) RGB tuple: (R, G, B) tuple of integers (e.g., (18, 17, 67)). Values should be in RGB color space (0-255 per channel). The color is automatically converted to BGR format for OpenCV processing. Use this to specify the exact color you want to count pixels for.. | ✅ |
| `tolerance` | `int` | Color matching tolerance value (0-255). Determines how much each BGR channel can vary from the target color and still be considered a match. The tolerance is applied to each color channel independently - a tolerance of 10 creates a range of ±10 for each BGR channel (total range of 21 values per channel). Lower values (e.g., 5-10) require very close color matches and are more precise but may miss slightly different shades. Higher values (e.g., 20-30) allow more color variation and match a wider range of similar colors but may include unintended colors. Default is 10, which provides a good balance. Adjust based on image quality, compression artifacts, and how strict you need the color matching to be.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Pixel Color Count` in version `v1`.

    - inputs: [`Circle Visualization`](circle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Background Subtraction`](background_subtraction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`OpenAI`](open_ai.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Image Slicer`](image_slicer.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Distance Measurement`](distance_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Calibration`](camera_calibration.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md)
    - outputs: [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Image Stack`](image_stack.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Email Notification`](email_notification.md), [`Dominant Color`](dominant_color.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Grid Visualization`](grid_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Background Subtraction`](background_subtraction.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Image Slicer`](image_slicer.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Pixel Color Count` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to analyze for pixel color counting. The block counts pixels in this image that match the target_color within the specified tolerance. All pixels in the image are analyzed. The image is processed in BGR format (OpenCV standard), and color matching is performed on each pixel's BGR values. Processing time depends on image size..
        - `target_color` (*Union[[`rgb_color`](../kinds/rgb_color.md), [`string`](../kinds/string.md)]*): Target color to count in the image. Can be specified in multiple formats: (1) Hex string format: '#RRGGBB' (6-digit, e.g., '#431112') or '#RGB' (3-digit shorthand, e.g., '#412'), (2) RGB tuple string format: '(R, G, B)' (e.g., '(128, 32, 64)'), or (3) RGB tuple: (R, G, B) tuple of integers (e.g., (18, 17, 67)). Values should be in RGB color space (0-255 per channel). The color is automatically converted to BGR format for OpenCV processing. Use this to specify the exact color you want to count pixels for..
        - `tolerance` (*[`integer`](../kinds/integer.md)*): Color matching tolerance value (0-255). Determines how much each BGR channel can vary from the target color and still be considered a match. The tolerance is applied to each color channel independently - a tolerance of 10 creates a range of ±10 for each BGR channel (total range of 21 values per channel). Lower values (e.g., 5-10) require very close color matches and are more precise but may miss slightly different shades. Higher values (e.g., 20-30) allow more color variation and match a wider range of similar colors but may include unintended colors. Default is 10, which provides a good balance. Adjust based on image quality, compression artifacts, and how strict you need the color matching to be..

    - output
    
        - `matching_pixels_count` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Pixel Color Count` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/pixel_color_count@v1",
	    "image": "$inputs.image",
	    "target_color": "#431112",
	    "tolerance": 10
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

