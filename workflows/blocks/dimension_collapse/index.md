
# Dimension Collapse



??? "Class: `DimensionCollapseBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/dimension_collapse/v1.py">inference.core.workflows.core_steps.fusion.dimension_collapse.v1.DimensionCollapseBlockV1</a>
    



Flatten nested batch data by reducing dimensionality from level n to level n-1, aggregating nested lists into a single flat list to enable data aggregation, batch flattening, and dimensionality reduction workflows where nested batch outputs (such as classification or OCR results from dynamically cropped images) need to be collapsed into a single-level batch for downstream processing.

## How This Block Works

This block collapses the dimensionality of batch data by flattening nested lists one level. The block:

1. Receives batch data at dimensionality level n (nested batch structure)
2. Flattens the nested structure:
   - Takes all elements from the nested batch structure
   - Concatenates them into a single flat list
   - Removes one level of nesting from the data structure
3. Reduces dimensionality:
   - Input data at level n (e.g., list of lists)
   - Output data at level n-1 (e.g., single list)
   - Maintains all data elements, just removes the nested structure
4. Returns flattened output:
   - Outputs a single list containing all elements from the nested input
   - Elements are preserved in order (flattened sequentially)
   - Output dimensionality is one level lower than input

This block is particularly useful when working with dynamically cropped images or other operations that create nested batch structures. For example, when you crop multiple objects from each image, you get a nested batch (level 2): a list where each element is itself a list of crops. Classification results for those crops also form a nested batch. The Dimension Collapse block flattens this nested structure into a single-level batch (level 1), allowing you to work with all results together.

## Common Use Cases

- **Aggregating Classification Results**: Aggregate classification results from dynamically cropped images into a single list (e.g., classify crops from images then aggregate all results, collect classification results from multiple crops, flatten nested classification outputs), enabling classification aggregation workflows
- **Aggregating OCR Results**: Aggregate OCR results from dynamically cropped text regions into a single list (e.g., OCR crops from images then aggregate all text results, collect OCR results from multiple crops, flatten nested OCR outputs), enabling OCR aggregation workflows
- **Batch Flattening**: Flatten nested batch structures for downstream processing (e.g., flatten nested batches for analysis, reduce batch dimensionality for storage, collapse nested structures for filtering), enabling batch flattening workflows
- **Data Aggregation**: Aggregate results from nested batch operations into flat lists (e.g., aggregate results from nested operations, collect outputs from nested batches, flatten nested operation results), enabling data aggregation workflows
- **Dimensionality Reduction**: Reduce batch dimensionality to match requirements of downstream blocks (e.g., reduce dimensionality for blocks requiring level 1 inputs, flatten nested batches for compatibility, adjust dimensionality for workflow connections), enabling dimensionality adjustment workflows
- **Result Collection**: Collect and flatten results from nested processing operations (e.g., collect nested processing results, flatten operation outputs, aggregate nested operation data), enabling result collection workflows

## Connecting to Other Blocks

This block receives nested batch data and produces flattened batch data:

- **After blocks that create nested batches** (crop blocks, classification on crops, OCR on crops) to flatten nested results (e.g., crop then classify then flatten, OCR crops then flatten, process nested batches then collapse), enabling nested-to-flat workflows
- **Before blocks requiring single-level batches** to provide flattened data (e.g., flatten before filtering, collapse before storage, aggregate before analysis), enabling flat-to-processing workflows
- **Before data storage blocks** to store aggregated flattened results (e.g., store aggregated classifications, save flattened OCR results, log collapsed batch data), enabling aggregation-to-storage workflows
- **Before analytics blocks** to analyze aggregated results (e.g., analyze aggregated classifications, perform analytics on flattened data, process collapsed batches), enabling aggregation-to-analytics workflows
- **Before filtering blocks** to filter flattened aggregated data (e.g., filter aggregated results, apply filters to collapsed batches, process flattened data), enabling aggregation-to-filter workflows
- **In workflow outputs** to provide aggregated flattened results as final output (e.g., aggregated classification outputs, flattened OCR outputs, collapsed batch outputs), enabling aggregation output workflows

## Requirements

This block requires batch data at dimensionality level n (nested batch structure). The block automatically handles batch casting for the input parameter. The block reduces output dimensionality by 1 level (from level n to level n-1). All elements from the nested structure are preserved and flattened into a single list. The block works with any data type - it simply flattens the nested list structure without modifying individual elements. The output is a single-level batch containing all elements from the nested input, ordered sequentially as they appear in the nested structure.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/dimension_collapse@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Dimension Collapse` in version `v1`.

    - inputs: [`Cache Set`](cache_set.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Delta Filter`](delta_filter.md), [`Expression`](expression.md), [`Detection Offset`](detection_offset.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Switch Case`](switch_case.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Cache Set`](cache_set.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Circle Visualization`](circle_visualization.md), [`Path Deviation`](path_deviation.md), [`Path Deviation`](path_deviation.md), [`Email Notification`](email_notification.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Google Gemma`](google_gemma.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter`](line_counter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Grid Visualization`](grid_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Crop Visualization`](crop_visualization.md), [`OpenAI`](open_ai.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Dimension Collapse` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*[`*`](../kinds/wildcard.md)*): Reference to step outputs at dimensionality level n (nested batch structure) to be flattened and collapsed to level n-1. The input should be a nested batch (e.g., list of lists) where each nested level represents a batch dimension. The block flattens this structure by concatenating all nested elements into a single flat list. Common use cases: classification results from cropped images (level 2 → level 1), OCR results from cropped regions (level 2 → level 1), or any nested batch structure that needs to be flattened..

    - output
    
        - `output` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Dimension Collapse` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/dimension_collapse@v1",
	    "data": "$steps.classification_step.predictions"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

