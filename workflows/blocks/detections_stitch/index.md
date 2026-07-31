
# Detections Stitch



??? "Class: `DetectionsStitchBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/detections_stitch/v1.py">inference.core.workflows.core_steps.fusion.detections_stitch.v1.DetectionsStitchBlockV1</a>
    



Merge detections from multiple image slices or crops back into a single unified detection result by converting coordinates from slice/crop space to original image coordinates, combining all detections, and optionally filtering overlapping detections to enable SAHI workflows, multi-stage detection pipelines, and coordinate-space merging workflows where detections from sub-images need to be reconstructed as if they were detected on the original image.

## How This Block Works

This block merges detections that were made on multiple sub-parts (slices or crops) of the same input image, reconstructing them as a single detection result in the original image coordinate space. The block:

1. Receives reference image and slice/crop predictions:
   - Takes the original reference image that was sliced or cropped
   - Receives predictions from detection models that processed each slice/crop
   - Predictions must contain parent coordinate metadata indicating slice/crop position
2. Retrieves crop offsets for each detection:
   - Extracts parent coordinates from each detection's metadata
   - Gets the offset (x, y position) indicating where each slice/crop was located in the original image
   - Uses this offset to transform coordinates from slice space to original image space
3. Manages crop metadata:
   - Updates image dimensions in detection metadata to match reference image dimensions
   - Validates that detections were not scaled (scaled detections are not supported)
   - Attaches parent coordinate information to detections for proper coordinate transformation
4. Transforms coordinates to original image space:
   - Moves bounding box coordinates (xyxy) from slice/crop coordinates to original image coordinates
   - Transforms segmentation masks from slice/crop space to original image space (if present)
   - Applies offset to align detections with their position in the original image
5. Merges all transformed detections:
   - Combines all re-aligned detections from all slices/crops into a single detection result
   - Creates unified detection output containing all detections from all sub-images
6. Applies overlap filtering (optional):
   - **None strategy**: Returns all merged detections without filtering (may contain duplicates from overlapping slices)
   - **NMS (Non-Maximum Suppression)**: Removes lower-confidence detections when IoU exceeds threshold, keeping only the highest confidence detection for each overlapping region
   - **NMM (Non-Maximum Merge)**: Combines overlapping detections instead of discarding them, merging detections that exceed IoU threshold
7. Returns merged detections:
   - Outputs unified detection result in original image coordinate space
   - Reduces dimensionality by 1 (multiple slice detections → single image detections)
   - All detections are now referenced to the original image dimensions and coordinates

This block is essential for SAHI (Slicing Adaptive Inference) workflows where an image is sliced, each slice is processed separately, and results need to be merged back. Overlapping slices can produce duplicate detections for the same object, so overlap filtering (NMS/NMM) helps clean up these duplicates. The coordinate transformation ensures that detection coordinates are correctly positioned relative to the original image, not the slices.

## Common Use Cases

- **SAHI Workflows**: Complete SAHI technique by merging detections from image slices back to original image coordinates (e.g., merge slice detections from SAHI processing, reconstruct full-image detections from slices, combine small object detection results), enabling SAHI detection workflows
- **Multi-Stage Detection**: Merge detections from secondary high-resolution models applied to dynamically cropped regions (e.g., coarse detection → crop → precise detection → merge, two-stage detection pipelines, hierarchical detection workflows), enabling multi-stage detection workflows
- **Small Object Detection**: Combine detection results from sliced images processed separately for small object detection (e.g., merge detections from aerial image slices, combine slice detection results, reconstruct detections from tiled images), enabling small object detection workflows
- **High-Resolution Processing**: Merge detections from high-resolution images processed in smaller chunks (e.g., merge detections from satellite image tiles, combine results from medical image regions, reconstruct detections from large image segments), enabling high-resolution detection workflows
- **Coordinate Space Unification**: Convert detections from multiple coordinate spaces (slice/crop space) to a single unified coordinate space (original image space) for consistent processing (e.g., unify detection coordinates, merge coordinate spaces, standardize detection positions), enabling coordinate unification workflows
- **Overlapping Region Handling**: Handle duplicate detections from overlapping slices or crops by applying overlap filtering (e.g., remove duplicate detections from overlapping slices, merge overlapping detections, clean up overlapping results), enabling overlap resolution workflows

## Connecting to Other Blocks

This block receives slice/crop predictions and reference images, and produces merged detections:

- **After detection models in SAHI workflows** following Image Slicer → Detection Model → Detections Stitch pattern to merge slice detections (e.g., merge SAHI slice detections, reconstruct full-image detections, combine slice results), enabling SAHI completion workflows
- **After secondary detection models** in multi-stage pipelines following Dynamic Crop → Detection Model → Detections Stitch pattern to merge cropped detections (e.g., merge cropped region detections, combine two-stage detection results, unify multi-stage outputs), enabling multi-stage detection workflows
- **Before visualization blocks** to visualize merged detection results on the original image (e.g., visualize merged detections, display stitched results, show unified detection output), enabling visualization workflows
- **Before filtering or analytics blocks** to process merged detection results (e.g., filter merged detections, analyze stitched results, process unified outputs), enabling analysis workflows
- **Before sink or storage blocks** to store or export merged detection results (e.g., save merged detections, export stitched results, store unified outputs), enabling storage workflows
- **In workflow outputs** to provide merged detections as final workflow output (e.g., return merged detections, output stitched results, provide unified detection output), enabling output workflows

## Requirements

This block requires a reference image (the original image that was sliced/cropped) and predictions from detection models that processed slices/crops. The predictions must contain parent coordinate metadata (PARENT_COORDINATES_KEY) indicating the position of each slice/crop in the original image. The block does not support scaled detections (detections that were resized relative to the parent image). Predictions should be from object detection or instance segmentation models. The block supports three overlap filtering strategies: "none" (no filtering, may include duplicates), "nms" (Non-Maximum Suppression, removes lower-confidence overlapping detections, default), and "nmm" (Non-Maximum Merge, combines overlapping detections). The IoU threshold (default 0.3) determines when detections are considered overlapping for filtering purposes. For more information on SAHI technique, see: https://ieeexplore.ieee.org/document/9897990.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_stitch@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `overlap_filtering_strategy` | `str` | Strategy for handling overlapping detections when merging results from overlapping slices/crops. 'none': No filtering applied, all detections are kept (may include duplicates from overlapping regions). 'nms' (Non-Maximum Suppression, default): Removes lower-confidence detections when IoU exceeds threshold, keeping only the highest confidence detection for each overlapping region. 'nmm' (Non-Maximum Merge): Combines overlapping detections instead of discarding them, merging detections that exceed IoU threshold. Use 'none' when you want to preserve all detections, 'nms' to remove duplicates (recommended for most cases), or 'nmm' to combine overlapping detections.. | ✅ |
| `iou_threshold` | `float` | Intersection over Union (IoU) threshold for overlap filtering. Range: 0.0 to 1.0. When overlap filtering strategy is 'nms' or 'nmm', detections with IoU above this threshold are considered overlapping. For NMS: overlapping detections with IoU above threshold result in lower-confidence detection being removed. For NMM: overlapping detections with IoU above threshold are merged. Lower values (e.g., 0.2-0.3) are more aggressive, removing/merging more detections. Higher values (e.g., 0.5-0.7) are more permissive, only handling highly overlapping detections. Default 0.3 works well for most use cases with overlapping slices.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Stitch` in version `v1`.

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Path Deviation`](path_deviation.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Corner Visualization`](corner_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Offset`](detection_offset.md), [`Qwen-VL`](qwen_vl.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Camera Calibration`](camera_calibration.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`Camera Focus`](camera_focus.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Relative Static Crop`](relative_static_crop.md), [`Depth Estimation`](depth_estimation.md), [`Detections Combine`](detections_combine.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`EasyOCR`](easy_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Cosmos 3`](cosmos3.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)
    - outputs: [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`Label Visualization`](label_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Overlap Analysis`](overlap_analysis.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Size Measurement`](size_measurement.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Combine`](detections_combine.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Dot Visualization`](dot_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Filter`](detections_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Line Counter`](line_counter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Mask Visualization`](mask_visualization.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Stitch` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `reference_image` (*[`image`](../kinds/image.md)*): Original reference image that was sliced or cropped to produce the input predictions. This image is used to determine the target coordinate space and image dimensions for the merged detections. All detection coordinates will be transformed to match this reference image's coordinate system. The same image that was provided to Image Slicer or Dynamic Crop blocks should be used here to ensure proper coordinate alignment..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions (object detection or instance segmentation) from detection models that processed image slices or crops. These predictions must contain parent coordinate metadata indicating the position of each slice/crop in the original image. Predictions are collected from multiple slices/crops and merged into a single unified detection result. The block converts coordinates from slice/crop space to original image space and combines all detections..
        - `overlap_filtering_strategy` (*[`string`](../kinds/string.md)*): Strategy for handling overlapping detections when merging results from overlapping slices/crops. 'none': No filtering applied, all detections are kept (may include duplicates from overlapping regions). 'nms' (Non-Maximum Suppression, default): Removes lower-confidence detections when IoU exceeds threshold, keeping only the highest confidence detection for each overlapping region. 'nmm' (Non-Maximum Merge): Combines overlapping detections instead of discarding them, merging detections that exceed IoU threshold. Use 'none' when you want to preserve all detections, 'nms' to remove duplicates (recommended for most cases), or 'nmm' to combine overlapping detections..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Intersection over Union (IoU) threshold for overlap filtering. Range: 0.0 to 1.0. When overlap filtering strategy is 'nms' or 'nmm', detections with IoU above this threshold are considered overlapping. For NMS: overlapping detections with IoU above threshold result in lower-confidence detection being removed. For NMM: overlapping detections with IoU above threshold are merged. Lower values (e.g., 0.2-0.3) are more aggressive, removing/merging more detections. Higher values (e.g., 0.5-0.7) are more permissive, only handling highly overlapping detections. Default 0.3 works well for most use cases with overlapping slices..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Detections Stitch` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_stitch@v1",
	    "reference_image": "$inputs.image",
	    "predictions": "$steps.object_detection.predictions",
	    "overlap_filtering_strategy": "none",
	    "iou_threshold": 0.2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

