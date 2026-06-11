
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

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Image Contours`](image_contours.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`Email Notification`](email_notification.md), [`Local File Sink`](local_file_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Template Matching`](template_matching.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md)

    
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

