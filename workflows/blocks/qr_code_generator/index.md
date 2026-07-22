
# QR Code Generator



??? "Class: `QRCodeGeneratorBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/qr_code_generator/v1.py">inference.core.workflows.core_steps.transformations.qr_code_generator.v1.QRCodeGeneratorBlockV1</a>
    



Generate QR code images from text input (URLs, text content, or other data) with customizable error correction levels, visual styling (colors, borders), and automatic size optimization, producing PNG images suitable for embedding, printing, or further image processing in workflows.

## How This Block Works

This block creates QR code images from text input using the qrcode library, encoding the provided text into a scannable QR code pattern. The block:

1. Receives text input (URLs, strings, or other data to encode) and QR code configuration parameters
2. Checks an internal LRU cache for previously generated QR codes with the same parameters (caching improves performance for repeated QR code generation)
3. Parses color specifications for fill and background colors (supports hex codes, RGB strings, standard color names, and CSS3 color names) using a common color utility
4. Maps the error correction level string to the corresponding qrcode library constant (Low, Medium, Quartile, or High error correction)
5. Creates a QRCode object with the specified parameters:
   - Auto-determines version (size) based on data length when version is None
   - Sets error correction level based on the selected option
   - Uses fixed box_size of 10 pixels per module
   - Applies the specified border width
6. Encodes the text data into the QR code pattern
7. Generates a PIL Image from the QR code with the specified fill and background colors
8. Converts the image to RGB format, then to a NumPy array
9. Converts from RGB (PIL format) to BGR (OpenCV/WorkflowImageData format) for compatibility with workflow image processing
10. Creates a WorkflowImageData object with the QR code image and metadata
11. Stores the result in the cache for future reuse (cache has 100 entry capacity and 1-hour TTL)
12. Returns the QR code image

The block automatically optimizes QR code size based on the data length (auto-sizing), ensuring the QR code is large enough to encode the data but not unnecessarily large. Error correction levels trade off between data capacity and error recovery: higher error correction allows the QR code to be scanned even if partially damaged or obscured, but reduces the maximum data capacity. The block uses caching to improve performance when generating the same QR codes multiple times, which is common in workflows that generate QR codes for the same URLs or data repeatedly.

## Common Use Cases

- **URL and Link Encoding**: Generate QR codes for URLs and web links (e.g., create QR codes for product pages, generate QR codes for documentation links, encode URLs for easy mobile access), enabling quick access to web resources via QR code scanning
- **Data Encoding and Sharing**: Encode text data, identifiers, or information into QR codes (e.g., generate QR codes for product IDs, encode serial numbers, create QR codes for inventory tracking), enabling machine-readable data encoding and sharing
- **Document and Report Generation**: Include QR codes in generated documents or reports (e.g., add QR codes to PDF reports linking to detailed data, embed QR codes in generated images, include QR codes in formatted outputs), enabling interactive document features with scannable links
- **Workflow Result Sharing**: Generate QR codes linking to workflow results or outputs (e.g., create QR codes pointing to detection results, encode links to analysis reports, generate QR codes for sharing workflow outputs), enabling easy sharing and access to workflow-generated content
- **Label and Tag Generation**: Create QR codes for labeling and identification purposes (e.g., generate QR codes for asset tags, create QR codes for product labels, encode identification information), enabling automated label and tag creation workflows
- **Integration and Automation**: Generate QR codes as part of automated workflows (e.g., create QR codes for automated document processing, generate QR codes for workflow automation triggers, encode data for system integration), enabling QR code generation as part of automated processes

## Connecting to Other Blocks

This block receives text input and produces QR code images:

- **After data processing blocks** (e.g., Expression, Property Definition) that produce text output to encode computed values, URLs, or processed data into QR codes, enabling machine-readable encoding of workflow-generated data
- **Before visualization blocks** that can overlay or combine QR codes with other images (e.g., overlay QR codes on images, combine QR codes with detection visualizations, embed QR codes in composite images), enabling QR code integration into visual outputs
- **Before formatter blocks** (e.g., CSV Formatter) to include QR code references or links in formatted outputs, enabling QR code integration into structured data exports
- **Before sink blocks** (e.g., Local File Sink, Webhook Sink) to save or send generated QR code images, enabling QR code distribution and storage
- **In document generation workflows** where QR codes need to be embedded in documents or reports, enabling interactive document features with scannable codes
- **After detection or analysis blocks** to generate QR codes linking to detection results or analysis outputs, enabling easy sharing and access to workflow results via QR code scanning


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/qr_code_generator@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `text` | `str` | Text, URL, or data to encode into the QR code. Can be any string content including URLs (e.g., 'https://roboflow.com'), text messages, identifiers, serial numbers, or other data. The QR code will automatically size itself based on the data length. Longer text requires larger QR codes or lower error correction levels to fit. Maximum data capacity depends on the error correction level selected.. | ✅ |
| `error_correct` | `str` | Error correction level that determines how much damage or obscuration the QR code can tolerate while still being scannable. Higher error correction allows scanning even if the QR code is partially damaged, obscured, or transformed, but reduces the maximum data capacity (text length). Choose 'Low' for maximum data capacity when QR codes will be clearly visible and undamaged. Choose 'Medium' (default) for balanced capacity and error recovery. Choose 'Quartile' or 'High' when QR codes may be partially obscured, damaged, or need to be scanned from difficult angles. Trade-off: higher error correction = better error recovery but less data capacity.. | ❌ |
| `border` | `int` | Border thickness in modules (QR code units). Defaults to 4 modules. The border is a quiet zone around the QR code pattern that helps QR code scanners identify and decode the code. Larger borders improve scanning reliability but increase image size. Minimum recommended border is 4 modules. Border is measured in QR code modules (not pixels - actual pixel border size depends on box_size which is fixed at 10 pixels per module).. | ✅ |
| `fill_color` | `str` | Color of the QR code pattern blocks (the dark squares in the QR code). Defaults to 'BLACK'. Supports multiple color formats: hex codes (e.g., '#FF0000' for red, '#000000' for black), RGB strings (e.g., 'rgb(255, 0, 0)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The fill color should contrast well with the background color for reliable scanning. Traditional black-on-white provides the best scanning reliability.. | ✅ |
| `back_color` | `str` | Background color of the QR code (the light areas between pattern blocks). Defaults to 'WHITE'. Supports multiple color formats: hex codes (e.g., '#FFFFFF' for white, '#000000' for black), RGB strings (e.g., 'rgb(255, 255, 255)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The background color should contrast well with the fill color for reliable scanning. Traditional white background with black fill provides the best scanning reliability.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `QR Code Generator` in version `v1`.

    - inputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Distance Measurement`](distance_measurement.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`LMM`](lmm.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OCR Model`](ocr_model.md), [`Slack Notification`](slack_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`VLM As Detector`](vlm_as_detector.md), [`Cosmos 3`](cosmos3.md), [`CogVLM`](cog_vlm.md), [`Qwen-VL`](qwen_vl.md), [`PP-OCR`](ppocr.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`PLC Writer`](plc_writer.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`GLM-OCR`](glmocr.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`S3 Sink`](s3_sink.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`QR Code Generator` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `text` (*[`string`](../kinds/string.md)*): Text, URL, or data to encode into the QR code. Can be any string content including URLs (e.g., 'https://roboflow.com'), text messages, identifiers, serial numbers, or other data. The QR code will automatically size itself based on the data length. Longer text requires larger QR codes or lower error correction levels to fit. Maximum data capacity depends on the error correction level selected..
        - `border` (*[`integer`](../kinds/integer.md)*): Border thickness in modules (QR code units). Defaults to 4 modules. The border is a quiet zone around the QR code pattern that helps QR code scanners identify and decode the code. Larger borders improve scanning reliability but increase image size. Minimum recommended border is 4 modules. Border is measured in QR code modules (not pixels - actual pixel border size depends on box_size which is fixed at 10 pixels per module)..
        - `fill_color` (*[`string`](../kinds/string.md)*): Color of the QR code pattern blocks (the dark squares in the QR code). Defaults to 'BLACK'. Supports multiple color formats: hex codes (e.g., '#FF0000' for red, '#000000' for black), RGB strings (e.g., 'rgb(255, 0, 0)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The fill color should contrast well with the background color for reliable scanning. Traditional black-on-white provides the best scanning reliability..
        - `back_color` (*[`string`](../kinds/string.md)*): Background color of the QR code (the light areas between pattern blocks). Defaults to 'WHITE'. Supports multiple color formats: hex codes (e.g., '#FFFFFF' for white, '#000000' for black), RGB strings (e.g., 'rgb(255, 255, 255)'), standard color names (e.g., 'BLACK', 'WHITE', 'RED', 'BLUE'), or CSS3 color names. The background color should contrast well with the fill color for reliable scanning. Traditional white background with black fill provides the best scanning reliability..

    - output
    
        - `qr_code` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `QR Code Generator` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/qr_code_generator@v1",
	    "text": "https://roboflow.com",
	    "error_correct": "Low (~7% word recovery / highest data capacity)",
	    "border": 2,
	    "fill_color": "BLACK",
	    "back_color": "WHITE"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

