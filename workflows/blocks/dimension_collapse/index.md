
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

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Rate Limiter`](rate_limiter.md), [`Cosine Similarity`](cosine_similarity.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Delta Filter`](delta_filter.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`Environment Secrets Store`](environment_secrets_store.md), [`EasyOCR`](easy_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CogVLM`](cog_vlm.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Expression`](expression.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`Bounding Rectangle`](bounding_rectangle.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Halo Visualization`](halo_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Cache Set`](cache_set.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`Google Gemma API`](google_gemma_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md)

    
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

