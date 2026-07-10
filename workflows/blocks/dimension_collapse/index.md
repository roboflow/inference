
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

    - inputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PLC Writer`](plc_writer.md), [`JSON Parser`](json_parser.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Velocity`](velocity.md), [`Image Threshold`](image_threshold.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`CogVLM`](cog_vlm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`MQTT Writer`](mqtt_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Merge`](detections_merge.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Seg Preview`](seg_preview.md), [`SAM 3`](sam3.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Cache Set`](cache_set.md), [`Path Deviation`](path_deviation.md), [`OpenAI`](open_ai.md), [`Data Aggregator`](data_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Event Writer`](event_writer.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Buffer`](buffer.md), [`Slack Notification`](slack_notification.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`Crop Visualization`](crop_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections Transformation`](detections_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5`](qwen3.5.md), [`Local File Sink`](local_file_sink.md), [`Delta Filter`](delta_filter.md), [`Object Detection Model`](object_detection_model.md), [`Inner Workflow`](inner_workflow.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Expression`](expression.md), [`Depth Estimation`](depth_estimation.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PP-OCR`](ppocr.md), [`Blur Visualization`](blur_visualization.md), [`Clip Comparison`](clip_comparison.md), [`EasyOCR`](easy_ocr.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CSV Formatter`](csv_formatter.md), [`Color Visualization`](color_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Motion Detection`](motion_detection.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Rate Limiter`](rate_limiter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Current Time`](current_time.md), [`Icon Visualization`](icon_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Cache Get`](cache_get.md), [`PLC Reader`](plc_reader.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Image Stack`](image_stack.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Continue If`](continue_if.md), [`Google Vision OCR`](google_vision_ocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Outliers`](identify_outliers.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Detection Offset`](detection_offset.md), [`Switch Case`](switch_case.md), [`Classification Label Visualization`](classification_label_visualization.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Detection Event Log`](detection_event_log.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Path Deviation`](path_deviation.md), [`Relative Static Crop`](relative_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Track Class Lock`](track_class_lock.md), [`Dominant Color`](dominant_color.md), [`Moondream2`](moondream2.md), [`Property Definition`](property_definition.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenRouter`](open_router.md), [`Identify Changes`](identify_changes.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Analysis`](overlap_analysis.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Email Notification`](email_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`SIFT Comparison`](sift_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)
    - outputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`PLC Reader`](plc_reader.md), [`Google Gemini`](google_gemini.md), [`Google Gemini`](google_gemini.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Seg Preview`](seg_preview.md), [`SAM 3`](sam3.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Cache Set`](cache_set.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Path Deviation`](path_deviation.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`OpenRouter`](open_router.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Size Measurement`](size_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md)

    
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

