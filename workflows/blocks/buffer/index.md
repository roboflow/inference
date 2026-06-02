
# Buffer



??? "Class: `BufferBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/buffer/v1.py">inference.core.workflows.core_steps.fusion.buffer.v1.BufferBlockV1</a>
    



Maintain a sliding window buffer of the last N values by storing recent inputs in a FIFO (First-In-First-Out) queue, with newest elements added to the beginning and oldest elements automatically removed when the buffer exceeds the specified length, enabling temporal data collection, frame history tracking, batch processing preparation, and sliding window analysis workflows.

## How This Block Works

This block maintains a rolling buffer that stores the most recent values passed to it, creating a sliding window of data over time. The block:

1. Receives input data of any type (images, detections, values, etc.) and configuration parameters (buffer length and padding option)
2. Maintains an internal buffer that persists across workflow executions:
   - Buffer is initialized as an empty list when the block is first created
   - Buffer state persists for the lifetime of the workflow execution
   - Each buffer block instance maintains its own separate buffer
3. Adds new data to the buffer:
   - Inserts the newest value at the beginning (index 0) of the buffer array
   - Most recent values appear first in the buffer
   - Older values are shifted to later positions in the array
4. Manages buffer size:
   - When buffer length exceeds the specified `length` parameter, removes the oldest elements
   - Keeps only the most recent `length` values
   - Automatically maintains the sliding window size
5. Applies optional padding:
   - If `pad` is True: Fills the buffer with `None` values until it reaches exactly `length` elements
   - Ensures consistent buffer size even when fewer than `length` values have been received
   - If `pad` is False: Buffer size grows from 0 to `length` as values are added, then stays at `length`
6. Returns the buffered array:
   - Outputs a list containing the buffered values in order (newest first)
   - List length equals `length` (if padding enabled) or current buffer size (if padding disabled)
   - Values are ordered from most recent (index 0) to oldest (last index)

The buffer implements a sliding window pattern where new data enters at the front and old data exits at the back when capacity is reached. This creates a temporal history of recent values, useful for operations that need to look back at previous frames, detections, or measurements. The buffer works with any data type, making it flexible for images, detections, numeric values, or other workflow outputs.

## Common Use Cases

- **Frame History Tracking**: Maintain a history of recent video frames for temporal analysis (e.g., track frame sequences, maintain recent image history, collect frames for comparison), enabling temporal frame analysis workflows
- **Detection History**: Buffer recent detections for trend analysis or comparison (e.g., track detection changes over time, compare current vs previous detections, analyze detection patterns), enabling detection history workflows
- **Batch Processing Preparation**: Collect multiple values before processing them together (e.g., batch process recent images, aggregate multiple detections, prepare data for batch operations), enabling batch processing workflows
- **Sliding Window Analysis**: Perform analysis on a rolling window of data (e.g., analyze trends over recent frames, calculate moving averages, detect changes in sequences), enabling sliding window analysis workflows
- **Visualization Sequences**: Maintain recent data for animation or sequence visualization (e.g., create frame sequences, visualize temporal changes, display recent history), enabling temporal visualization workflows
- **Temporal Comparison**: Compare current values with recent historical values (e.g., compare current frame with previous frames, detect changes over time, analyze temporal patterns), enabling temporal comparison workflows

## Connecting to Other Blocks

This block receives data of any type and produces a buffered output array:

- **After any block** that produces values to buffer (e.g., buffer images from image sources, buffer detections from detection models, buffer values from analytics blocks), enabling data buffering workflows
- **Before blocks that process arrays** to provide batched or historical data (e.g., process buffered images, analyze detection arrays, work with value sequences), enabling array processing workflows
- **Before visualization blocks** to display sequences or temporal data (e.g., visualize frame sequences, display detection history, show temporal patterns), enabling temporal visualization workflows
- **Before analysis blocks** that require historical data (e.g., analyze trends over time, compare current vs historical, process temporal sequences), enabling temporal analysis workflows
- **Before aggregation blocks** to provide multiple values for aggregation (e.g., aggregate buffered values, process multiple detections, combine recent data), enabling aggregation workflows
- **In temporal processing pipelines** where maintaining recent history is required (e.g., track changes over time, maintain frame sequences, collect data for temporal analysis), enabling temporal processing workflows

## Requirements

This block works with any data type (images, detections, values, etc.). The buffer maintains state across workflow executions within the same workflow instance. The `length` parameter determines the maximum number of values to keep in the buffer. When `pad` is enabled, the buffer will always return exactly `length` elements (padded with `None` if needed). When `pad` is disabled, the buffer grows from 0 to `length` elements as values are added, then maintains `length` elements by removing oldest values. The buffer persists for the lifetime of the workflow execution and resets when the workflow is restarted.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/buffer@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `length` | `int` | Maximum number of elements to keep in the buffer. When the buffer exceeds this length, the oldest elements are automatically removed. Determines the size of the sliding window. Must be greater than 0. Typical values range from 2-10 for frame sequences, or higher for longer histories.. | ❌ |
| `pad` | `bool` | Enable padding to maintain consistent buffer size. If True, the buffer is padded with `None` values until it reaches exactly `length` elements, ensuring the output always has `length` items even when fewer values have been received. If False, the buffer grows from 0 to `length` as values are added, then maintains `length` by removing oldest values. Use padding when downstream blocks require a fixed-size array.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Buffer` in version `v1`.

    - inputs: [`Distance Measurement`](distance_measurement.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Delta Filter`](delta_filter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Cache Get`](cache_get.md), [`Heatmap Visualization`](heatmap_visualization.md), [`CogVLM`](cog_vlm.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Cache Set`](cache_set.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Rate Limiter`](rate_limiter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Continue If`](continue_if.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Inner Workflow`](inner_workflow.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Cosine Similarity`](cosine_similarity.md), [`Google Gemini`](google_gemini.md), [`QR Code Detection`](qr_code_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Property Definition`](property_definition.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Slack Notification`](slack_notification.md), [`GLM-OCR`](glmocr.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SmolVLM2`](smol_vlm2.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Corner Visualization`](corner_visualization.md), [`Expression`](expression.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Data Aggregator`](data_aggregator.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dominant Color`](dominant_color.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Identify Changes`](identify_changes.md), [`OCR Model`](ocr_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Detections Merge`](detections_merge.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Image Threshold`](image_threshold.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen3-VL`](qwen3_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Path Deviation`](path_deviation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemma`](google_gemma.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Florence-2 Model`](florence2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Seg Preview`](seg_preview.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Path Deviation`](path_deviation.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Time in Zone`](timein_zone.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Buffer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`*`](../kinds/wildcard.md), [`image`](../kinds/image.md), [`list_of_values`](../kinds/list_of_values.md)]*): Input data of any type to add to the buffer. Can be images, detections, values, or any other workflow output. Newest values are added to the beginning of the buffer array. The buffer maintains a sliding window of the most recent values..

    - output
    
        - `output` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Buffer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/buffer@v1",
	    "data": "$steps.visualization",
	    "length": 5,
	    "pad": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

