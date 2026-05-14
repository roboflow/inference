
# Text Display



??? "Class: `TextDisplayVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/text_display/v1.py">inference.core.workflows.core_steps.visualizations.text_display.v1.TextDisplayVisualizationBlockV1</a>
    



The **Text Display** block renders text on an image with full control over styling and positioning.

### Dynamic Text Content

Text content can be parameterized with workflow execution outcomes using the same templating syntax 
as Email and SMS notification blocks:

```
text = "Detected {{ '{{' }} $parameters.count {{ '}}' }} objects of class {{ '{{' }} $parameters.class_name {{ '}}' }}"
```

Parameters are provided via the `text_parameters` field:

```
text_parameters = {
    "count": "$steps.model.predictions",
    "class_name": "$inputs.target_class"
}
```

You can apply transformations to parameters using `text_parameters_operations`:

```
text_parameters_operations = {
    "count": [{"type": "SequenceLength"}]
}
```

### Styling Options

- **text_color**: Color of the text. Supports:
  - Supervision color names (uppercase): "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "ROBOFLOW", etc.
  - Hex format: "#RRGGBB" (e.g., "#FF0000" for red)
  - RGB format: "rgb(R, G, B)" (e.g., "rgb(255, 0, 0)" for red)
  - BGR format: "bgr(B, G, R)" (e.g., "bgr(0, 0, 255)" for red)
- **background_color**: Background color behind the text. Supports the same color formats as `text_color`. Use "transparent" for no background.
- **background_opacity**: Transparency of the background (0.0 = fully transparent, 1.0 = fully opaque)
- **font_scale**: Scale factor for the font size
- **font_thickness**: Thickness of the text strokes
- **padding**: Padding around the text in pixels
- **text_align**: Horizontal text alignment ("left", "center", "right")
- **border_radius**: Radius for rounded corners on the background

### Positioning Options

The block supports both absolute and relative positioning:

**Absolute Positioning** (`position_mode = "absolute"`):
- `position_x`: X coordinate in pixels from the left edge
- `position_y`: Y coordinate in pixels from the top edge

**Relative Positioning** (`position_mode = "relative"`):
- `anchor`: Where to anchor the text ("center", "top_left", "top_center", "top_right", 
  "bottom_left", "bottom_center", "bottom_right", "center_left", "center_right")
- `offset_x`: Horizontal offset from the anchor point (positive = right)
- `offset_y`: Vertical offset from the anchor point (positive = down)


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/text_display@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `text` | `str` | The text content to display. Supports parameter interpolation using {{ '{{' }} $parameters.name {{ '}}' }} syntax.. | ❌ |
| `text_parameters` | `Dict[str, Union[bool, float, int, str]]` | Parameters to interpolate into the text.. | ✅ |
| `text_parameters_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Operations to apply to text parameters before interpolation.. | ❌ |
| `text_color` | `str` | Color of the text. Supports supervision color names (WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ROBOFLOW, etc.), hex format (#RRGGBB), rgb(R,G,B) format, or bgr(B,G,R) format.. | ✅ |
| `background_color` | `str` | Background color behind the text. Supports the same color formats as text_color. Use 'transparent' for no background.. | ✅ |
| `background_opacity` | `float` | Opacity of the background (0.0 = fully transparent, 1.0 = fully opaque).. | ✅ |
| `font_scale` | `float` | Scale factor for the font size.. | ✅ |
| `font_thickness` | `int` | Thickness of the text strokes.. | ✅ |
| `padding` | `int` | Padding around the text in pixels.. | ✅ |
| `text_align` | `str` | Horizontal alignment of the text within its bounding box.. | ✅ |
| `border_radius` | `int` | Radius for rounded corners on the background rectangle.. | ✅ |
| `position_mode` | `str` | Positioning mode: 'absolute' uses exact pixel coordinates, 'relative' uses anchor points with offsets.. | ✅ |
| `position_x` | `int` | X coordinate (pixels from left edge) when using absolute positioning.. | ✅ |
| `position_y` | `int` | Y coordinate (pixels from top edge) when using absolute positioning.. | ✅ |
| `anchor` | `str` | Anchor point for relative positioning.. | ✅ |
| `offset_x` | `int` | Horizontal offset from anchor point (positive = right).. | ✅ |
| `offset_y` | `int` | Vertical offset from anchor point (positive = down).. | ✅ |
| `copy_image` | `bool` | Whether to copy the input image before drawing (preserves original).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Text Display` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`Polygon Visualization`](polygon_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Rate Limiter`](rate_limiter.md), [`QR Code Generator`](qr_code_generator.md), [`Cosine Similarity`](cosine_similarity.md), [`Identify Changes`](identify_changes.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Delta Filter`](delta_filter.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Environment Secrets Store`](environment_secrets_store.md), [`EasyOCR`](easy_ocr.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Expression`](expression.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`EasyOCR`](easy_ocr.md), [`SAM 3`](sam3.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`CogVLM`](cog_vlm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter Visualization`](line_counter_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Text Display` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to display text on..
        - `text_parameters` (*[`*`](../kinds/wildcard.md)*): Parameters to interpolate into the text..
        - `text_color` (*[`string`](../kinds/string.md)*): Color of the text. Supports supervision color names (WHITE, BLACK, RED, GREEN, BLUE, YELLOW, ROBOFLOW, etc.), hex format (#RRGGBB), rgb(R,G,B) format, or bgr(B,G,R) format..
        - `background_color` (*[`string`](../kinds/string.md)*): Background color behind the text. Supports the same color formats as text_color. Use 'transparent' for no background..
        - `background_opacity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Opacity of the background (0.0 = fully transparent, 1.0 = fully opaque)..
        - `font_scale` (*[`float`](../kinds/float.md)*): Scale factor for the font size..
        - `font_thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the text strokes..
        - `padding` (*[`integer`](../kinds/integer.md)*): Padding around the text in pixels..
        - `text_align` (*[`string`](../kinds/string.md)*): Horizontal alignment of the text within its bounding box..
        - `border_radius` (*[`integer`](../kinds/integer.md)*): Radius for rounded corners on the background rectangle..
        - `position_mode` (*[`string`](../kinds/string.md)*): Positioning mode: 'absolute' uses exact pixel coordinates, 'relative' uses anchor points with offsets..
        - `position_x` (*[`integer`](../kinds/integer.md)*): X coordinate (pixels from left edge) when using absolute positioning..
        - `position_y` (*[`integer`](../kinds/integer.md)*): Y coordinate (pixels from top edge) when using absolute positioning..
        - `anchor` (*[`string`](../kinds/string.md)*): Anchor point for relative positioning..
        - `offset_x` (*[`integer`](../kinds/integer.md)*): Horizontal offset from anchor point (positive = right)..
        - `offset_y` (*[`integer`](../kinds/integer.md)*): Vertical offset from anchor point (positive = down)..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Whether to copy the input image before drawing (preserves original)..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Text Display` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/text_display@v1",
	    "image": "$inputs.image",
	    "text": "Detection count: {{ '{{' }} $parameters.count {{ '}}' }}",
	    "text_parameters": {
	        "class_name": "$inputs.target_class",
	        "count": "$steps.model.predictions"
	    },
	    "text_parameters_operations": {
	        "count": [
	            {
	                "type": "SequenceLength"
	            }
	        ]
	    },
	    "text_color": "WHITE",
	    "background_color": "BLACK",
	    "background_opacity": 1.0,
	    "font_scale": 1.0,
	    "font_thickness": 1,
	    "padding": 5,
	    "text_align": "left",
	    "border_radius": 0,
	    "position_mode": "absolute",
	    "position_x": 10,
	    "position_y": 10,
	    "anchor": "center",
	    "offset_x": 0,
	    "offset_y": 0,
	    "copy_image": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

