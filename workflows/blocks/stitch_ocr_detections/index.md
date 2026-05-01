
# Stitch OCR Detections



## v2

??? "Class: `StitchOCRDetectionsBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/stitch_ocr_detections/v2.py">inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v2.StitchOCRDetectionsBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Combine individual OCR detection results (words, characters, or text regions) into coherent text strings by organizing detections spatially, grouping them into lines, and concatenating text in proper reading order.

## Stitching Algorithms

This block supports three algorithms for reconstructing text from OCR detections:

### Tolerance-based (default)
Groups detections into lines using a fixed pixel tolerance. Detections within the tolerance distance vertically (or horizontally for vertical text) are grouped into the same line, then sorted by position within each line.

- **Best for**: Consistent font sizes and well-aligned horizontal/vertical text
- **Parameters**: `tolerance` (pixel threshold for line grouping)

### Otsu Thresholding
Uses Otsu's method on normalized gap distances to automatically find the optimal threshold separating character gaps from word gaps. Gaps are normalized by local character width, making it resolution-invariant.

- **Best for**: Variable font sizes, automatic word boundary detection
- **Parameters**: `otsu_threshold_multiplier` (adjust threshold sensitivity)
- **Key feature**: Detects bimodal distributions to distinguish single words from multi-word text

### Collimate (Skewed Text)
Uses greedy parent-child traversal to follow text flow. Starting from the first detection, it finds subsequent detections that "follow" in reading order (similar alignment + correct direction), building lines through traversal rather than bucketing.

- **Best for**: Skewed, curved, or non-axis-aligned text
- **Parameters**: `collimate_tolerance` (alignment tolerance in pixels)
- **Note**: Does not detect word boundaries - use `delimiter` parameter if spacing is needed

## Reading Directions

All algorithms support multiple reading directions:
- `left_to_right`: Standard horizontal (English, most languages)
- `right_to_left`: Right-to-left (Arabic, Hebrew)
- `vertical_top_to_bottom`: Vertical top-to-bottom (Traditional Chinese, Japanese)
- `vertical_bottom_to_top`: Vertical bottom-to-top
- `auto`: Automatically detect based on bounding box dimensions

## Common Use Cases

- **Document OCR**: Reconstruct paragraphs and lines from character/word detections
- **Multi-language support**: Handle different reading directions and writing systems
- **Skewed text processing**: Use collimate algorithm for tilted or curved text
- **Word detection**: Use Otsu algorithm to automatically insert spaces between words


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stitch_ocr_detections@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `stitching_algorithm` | `str` | Algorithm for grouping detections into words/lines. 'tolerance': Uses fixed pixel tolerance for line grouping (original algorithm). Good for consistent font sizes and line spacing. 'otsu': Uses Otsu's method on normalized gaps to find natural breaks between words. Resolution-invariant and works well with bimodal gap distributions. 'collimate': Uses greedy parent-child traversal to group detections. Good for skewed or curved text where bucket-based approaches fail.. | ❌ |
| `reading_direction` | `str` | Direction to read and organize text detections. 'left_to_right': Standard horizontal reading (English, most languages). 'right_to_left': Right-to-left reading (Arabic, Hebrew). 'vertical_top_to_bottom': Vertical reading from top to bottom (Traditional Chinese, Japanese). 'vertical_bottom_to_top': Vertical reading from bottom to top (rare vertical formats). 'auto': Automatically detects reading direction based on average bounding box dimensions (width > height = horizontal, height >= width = vertical). Determines how detections are grouped into lines and sorted within lines.. | ❌ |
| `tolerance` | `int` | Vertical (or horizontal for vertical text) distance threshold in pixels for grouping detections into the same line. Detections within this tolerance distance are grouped into the same line. Higher values group detections that are further apart (useful for text with variable line spacing or slanted text). Lower values create more lines (useful for tightly spaced text). Must be greater than zero.. | ✅ |
| `delimiter` | `str` | Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas.. | ✅ |
| `otsu_threshold_multiplier` | `float` | Multiplier applied to the Otsu-computed threshold when using the 'otsu' stitching algorithm. Values > 1.0 make word breaks less frequent (more conservative, fewer splits), values < 1.0 make word breaks more frequent (more aggressive, more splits). Default is 1.0 (use Otsu threshold as-is). Try 1.3-1.5 if words are being incorrectly split, or 0.7-0.9 if words are being incorrectly merged.. | ✅ |
| `collimate_tolerance` | `int` | Pixel tolerance for the 'collimate' stitching algorithm. Controls how much vertical (for horizontal text) or horizontal (for vertical text) deviation is allowed when determining if a detection follows another in reading order. Higher values handle more skewed text but may incorrectly merge separate lines. Default is 10 pixels.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stitch OCR Detections` in version `v2`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Pixel Color Count`](pixel_color_count.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Icon Visualization`](icon_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`LMM`](lmm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Dynamic Crop`](dynamic_crop.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stitch OCR Detections` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*[`object_detection_prediction`](../kinds/object_detection_prediction.md)*): OCR detection predictions from an OCR model. Should contain bounding boxes and class names with text content. Each detection represents a word, character, or text region that will be stitched together into coherent text. Supports object detection format with bounding boxes (xyxy) and class names in the data dictionary..
        - `tolerance` (*[`integer`](../kinds/integer.md)*): Vertical (or horizontal for vertical text) distance threshold in pixels for grouping detections into the same line. Detections within this tolerance distance are grouped into the same line. Higher values group detections that are further apart (useful for text with variable line spacing or slanted text). Lower values create more lines (useful for tightly spaced text). Must be greater than zero..
        - `delimiter` (*[`string`](../kinds/string.md)*): Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas..
        - `otsu_threshold_multiplier` (*[`float`](../kinds/float.md)*): Multiplier applied to the Otsu-computed threshold when using the 'otsu' stitching algorithm. Values > 1.0 make word breaks less frequent (more conservative, fewer splits), values < 1.0 make word breaks more frequent (more aggressive, more splits). Default is 1.0 (use Otsu threshold as-is). Try 1.3-1.5 if words are being incorrectly split, or 0.7-0.9 if words are being incorrectly merged..
        - `collimate_tolerance` (*[`integer`](../kinds/integer.md)*): Pixel tolerance for the 'collimate' stitching algorithm. Controls how much vertical (for horizontal text) or horizontal (for vertical text) deviation is allowed when determining if a detection follows another in reading order. Higher values handle more skewed text but may incorrectly merge separate lines. Default is 10 pixels..

    - output
    
        - `ocr_text` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Stitch OCR Detections` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stitch_ocr_detections@v2",
	    "stitching_algorithm": "tolerance",
	    "predictions": "$steps.ocr_model.predictions",
	    "reading_direction": "left_to_right",
	    "tolerance": 10,
	    "delimiter": "",
	    "otsu_threshold_multiplier": 1.0,
	    "collimate_tolerance": 5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `StitchOCRDetectionsBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/stitch_ocr_detections/v1.py">inference.core.workflows.core_steps.transformations.stitch_ocr_detections.v1.StitchOCRDetectionsBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Combine individual OCR detection results (words, characters, or text regions) into coherent text strings by organizing detections spatially according to reading direction, grouping detections into lines, sorting them within lines, and concatenating text in proper reading order to reconstruct readable text from OCR model outputs.

## How This Block Works

This block reconstructs readable text from individual OCR detections by organizing them spatially and concatenating text in proper reading order. The block:

1. Receives OCR detection predictions containing individual text detections with bounding boxes and class names (text content)
2. Prepares coordinates based on reading direction:
   - For vertical reading directions, swaps x and y coordinates to enable vertical line processing
   - For horizontal reading directions, uses coordinates as-is
3. Groups detections into lines:
   - Groups detections based on vertical position (or horizontal position for vertical text) using the tolerance parameter
   - Detections within the tolerance distance are considered part of the same line
   - Higher tolerance values group detections that are further apart, useful for text with variable line spacing
4. Sorts lines based on reading direction:
   - For left-to-right and vertical top-to-bottom: sorts lines from top to bottom
   - For right-to-left and vertical bottom-to-top: sorts lines in reverse order (bottom to top)
5. Sorts detections within each line:
   - For left-to-right and vertical top-to-bottom: sorts detections by horizontal position (left to right, or top to bottom for vertical)
   - For right-to-left and vertical bottom-to-top: sorts detections in reverse order (right to left, or bottom to top for vertical)
6. Concatenates text in reading order:
   - Extracts class names (text content) from detections in sorted order
   - Adds line separators (newline for horizontal text, space for vertical text) between lines
   - Optionally inserts a delimiter between each text element if specified
   - Produces a single coherent text string with proper reading order
7. Handles automatic reading direction detection (if "auto" is selected):
   - Analyzes average width and height of detection bounding boxes
   - If average width > average height: detects horizontal text (left-to-right)
   - If average height >= average width: detects vertical text (top-to-bottom)
8. Returns the stitched text string:
   - Outputs a single text string under the `ocr_text` key
   - Text is formatted with proper line breaks and spacing according to reading direction

The block enables reconstruction of multi-line text from individual OCR detections, maintaining proper reading order for different languages and writing systems. It handles both horizontal (left-to-right, right-to-left) and vertical (top-to-bottom, bottom-to-top) text orientations, making it useful for processing text in various languages and formats.

## Common Use Cases

- **Text Reconstruction**: Convert individual word or character detections from OCR models into readable text blocks (e.g., reconstruct documents from word detections, combine character detections into words, stitch OCR results into paragraphs), enabling text reconstruction workflows
- **Multi-Line Text Processing**: Reconstruct multi-line text from OCR results with proper line breaks and formatting (e.g., extract paragraphs from OCR results, reconstruct formatted text, process multi-line documents), enabling multi-line text workflows
- **Multi-Language OCR**: Process OCR results from different languages and writing systems (e.g., process Arabic right-to-left text, handle vertical Chinese/Japanese text, support multiple reading directions), enabling multi-language OCR workflows
- **Document Processing**: Extract and reconstruct text from documents and images (e.g., extract text from scanned documents, process invoice text, extract text from forms), enabling document processing workflows
- **Text Extraction and Formatting**: Extract text from images and format it for downstream use (e.g., extract text for database storage, format text for API responses, prepare text for analysis), enabling text extraction workflows
- **OCR Result Post-Processing**: Post-process OCR model outputs to produce usable text strings (e.g., format OCR outputs, organize OCR results, prepare text for downstream blocks), enabling OCR post-processing workflows

## Connecting to Other Blocks

This block receives OCR detection predictions and produces stitched text strings:

- **After OCR model blocks** to convert detection results into readable text (e.g., OCR model to text string, OCR detections to formatted text, OCR results to text output), enabling OCR-to-text workflows
- **Before data storage blocks** to store extracted text (e.g., store OCR text in databases, save extracted text, log OCR results), enabling text storage workflows
- **Before notification blocks** to send extracted text in notifications (e.g., send OCR text in alerts, include extracted text in messages, notify with OCR results), enabling text notification workflows
- **Before text processing blocks** to process stitched text (e.g., process text with NLP models, analyze extracted text, apply text transformations), enabling text processing workflows
- **Before API output blocks** to provide text in API responses (e.g., return OCR text in API, format text for responses, provide extracted text output), enabling text output workflows
- **In workflow outputs** to provide stitched text as final output (e.g., text extraction workflows, OCR output workflows, document processing workflows), enabling text output workflows

## Requirements

This block requires OCR detection predictions (object detection format) with bounding boxes and class names containing text content. The `tolerance` parameter must be greater than zero and controls the vertical (or horizontal for vertical text) distance threshold for grouping detections into lines. The `reading_direction` parameter supports five modes: "left_to_right" (standard horizontal), "right_to_left" (Arabic-style), "vertical_top_to_bottom" (vertical), "vertical_bottom_to_top" (vertical reversed), and "auto" (automatic detection based on bounding box dimensions). The `delimiter` parameter is optional and inserts a delimiter between each text element (empty string by default, meaning no delimiter). The block outputs a single text string under the `ocr_text` key.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stitch_ocr_detections@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `reading_direction` | `str` | Direction to read and organize text detections. 'left_to_right': Standard horizontal reading (English, most languages). 'right_to_left': Right-to-left reading (Arabic, Hebrew). 'vertical_top_to_bottom': Vertical reading from top to bottom (Traditional Chinese, Japanese). 'vertical_bottom_to_top': Vertical reading from bottom to top (rare vertical formats). 'auto': Automatically detects reading direction based on average bounding box dimensions (width > height = horizontal, height >= width = vertical). Determines how detections are grouped into lines and sorted within lines.. | ❌ |
| `tolerance` | `int` | Vertical (or horizontal for vertical text) distance threshold in pixels for grouping detections into the same line. Detections within this tolerance distance are grouped into the same line. Higher values group detections that are further apart (useful for text with variable line spacing or slanted text). Lower values create more lines (useful for tightly spaced text). Must be greater than zero.. | ✅ |
| `delimiter` | `str` | Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stitch OCR Detections` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Pixel Color Count`](pixel_color_count.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Icon Visualization`](icon_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`LMM`](lmm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Dynamic Crop`](dynamic_crop.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stitch OCR Detections` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*[`object_detection_prediction`](../kinds/object_detection_prediction.md)*): OCR detection predictions from an OCR model. Should contain bounding boxes and class names with text content. Each detection represents a word, character, or text region that will be stitched together into coherent text. Supports object detection format with bounding boxes (xyxy) and class names in the data dictionary..
        - `tolerance` (*[`integer`](../kinds/integer.md)*): Vertical (or horizontal for vertical text) distance threshold in pixels for grouping detections into the same line. Detections within this tolerance distance are grouped into the same line. Higher values group detections that are further apart (useful for text with variable line spacing or slanted text). Lower values create more lines (useful for tightly spaced text). Must be greater than zero..
        - `delimiter` (*[`string`](../kinds/string.md)*): Optional delimiter string to insert between each text element (word/character) when stitching. Empty string (default) means no delimiter - text elements are concatenated directly. Useful for adding spaces between words, commas between elements, or custom separators. Example: use ' ' (space) to add spaces between words, or ',' to add commas..

    - output
    
        - `ocr_text` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Stitch OCR Detections` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stitch_ocr_detections@v1",
	    "predictions": "$steps.ocr_model.predictions",
	    "reading_direction": "left_to_right",
	    "tolerance": 10,
	    "delimiter": ""
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

