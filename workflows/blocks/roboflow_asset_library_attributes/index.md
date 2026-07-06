
# Roboflow Asset Library Attributes



??? "Class: `RoboflowAssetLibraryAttributesBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/asset_library_attributes/v1.py">inference.core.workflows.core_steps.sinks.roboflow.asset_library_attributes.v1.RoboflowAssetLibraryAttributesBlockV1</a>
    



Submit attribute and tag updates for existing Asset Library images, enabling enrichment workflows where model outputs become filterable image fields.

## How This Block Works

This block submits key-value attributes and tags for existing Asset Library images in your Roboflow workspace. Attribute values are stored as image metadata. The block:

1. Receives Asset Library source image IDs, optional attributes, and optional tags
2. Resolves the target workspace from the configured Roboflow API key
3. Skips rows where both attributes and tags are empty
4. Merges duplicate source IDs using sequential semantics: later attribute values win, and tags are added as a de-duplicated set
5. Submits one batch update request and returns one submission status per input source ID, in input order

Re-running the same workflow against the same source IDs is safe: attribute keys are upserted (last write wins) and tags are unioned. There is no destructive write.

The block does not send image bytes and does not create new images. It only updates existing Asset Library source images. Removing attribute keys, removing tags, writing annotations, and creating images are intentionally out of scope for this workflow block.

## Requirements

This block requires a valid Roboflow API key. The API key determines the workspace whose Asset Library images can be updated.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/asset_library_attributes@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `source_id` | `str` | Asset Library source image ID to update. For batch workflows, provide one source ID per image.. | ✅ |
| `metadata` | `Dict[str, Union[bool, float, int, str]]` | Optional key-value attributes to set on the Asset Library image. Attributes are stored as image metadata. Either an inline dict whose values may be static or selector references (e.g. `$inputs.camera_id`), or a whole-field selector to a per-row dict produced by an upstream step.. | ✅ |
| `tags` | `List[str]` | Optional tags to add to the Asset Library image. Each entry may be a static string or a reference to a workflow input/step (e.g. `$inputs.label`).. | ✅ |
| `disable_sink` | `bool` | If True, the block execution is disabled and no Asset Library attribute writes occur.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Asset Library Attributes` in version `v1`.

    - inputs: [`Absolute Static Crop`](absolute_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Distance Measurement`](distance_measurement.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Delta Filter`](delta_filter.md), [`Blur Visualization`](blur_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Path Deviation`](path_deviation.md), [`QR Code Detection`](qr_code_detection.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Property Definition`](property_definition.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Cache Get`](cache_get.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Stitch Images`](stitch_images.md), [`Google Gemini`](google_gemini.md), [`QR Code Generator`](qr_code_generator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Line Counter`](line_counter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Rate Limiter`](rate_limiter.md), [`Switch Case`](switch_case.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Inner Workflow`](inner_workflow.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Expression`](expression.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Local File Sink`](local_file_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SORT Tracker`](sort_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Cache Get`](cache_get.md), [`Triangle Visualization`](triangle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SAM 3`](sam3.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Path Deviation`](path_deviation.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Current Time`](current_time.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Asset Library Attributes` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `source_id` (*[`string`](../kinds/string.md)*): Asset Library source image ID to update. For batch workflows, provide one source ID per image..
        - `metadata` (*Union[[`*`](../kinds/wildcard.md), [`dictionary`](../kinds/dictionary.md)]*): Optional key-value attributes to set on the Asset Library image. Attributes are stored as image metadata. Either an inline dict whose values may be static or selector references (e.g. `$inputs.camera_id`), or a whole-field selector to a per-row dict produced by an upstream step..
        - `tags` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): Optional tags to add to the Asset Library image. Each entry may be a static string or a reference to a workflow input/step (e.g. `$inputs.label`)..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, the block execution is disabled and no Asset Library attribute writes occur..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Asset Library Attributes` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/asset_library_attributes@v1",
	    "source_id": "$inputs.source_id",
	    "metadata": {
	        "color": "red",
	        "score": 0.98
	    },
	    "tags": [
	        "auto-labeled",
	        "red"
	    ],
	    "disable_sink": false
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

