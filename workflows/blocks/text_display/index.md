
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

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Detection Offset`](detection_offset.md), [`Cache Set`](cache_set.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`SIFT`](sift.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Property Definition`](property_definition.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Rate Limiter`](rate_limiter.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`GLM-OCR`](glmocr.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Merge`](detections_merge.md), [`Line Counter`](line_counter.md), [`Detections Filter`](detections_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Overlap Analysis`](overlap_analysis.md), [`Current Time`](current_time.md), [`Expression`](expression.md), [`Cache Get`](cache_get.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`Cosmos 3`](cosmos3.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Corner Visualization`](corner_visualization.md), [`Continue If`](continue_if.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dominant Color`](dominant_color.md), [`Delta Filter`](delta_filter.md), [`Mask Visualization`](mask_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Switch Case`](switch_case.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Combine`](detections_combine.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Relative Static Crop`](relative_static_crop.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`LMM`](lmm.md), [`Corner Visualization`](corner_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md)

    
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

