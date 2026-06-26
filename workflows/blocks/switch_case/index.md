
# Switch Case



??? "Class: `SwitchCaseBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/flow_control/switch_case/v1.py">inference.core.workflows.core_steps.flow_control.switch_case.v1.SwitchCaseBlockV1</a>
    



Route workflow execution to one of several branches by matching an input value against a set of
case values, similar to a switch-case statement in programming, enabling multi-way branching,
value-based routing, and decision trees without chaining multiple Continue If blocks.

## How This Block Works

This block compares a single input value against the keys of a case mapping and directs execution
to the step associated with the first matching case. The block:

1. Takes a `value` input (typically a selector referencing a workflow input or a step output, e.g.
   a classification result) and converts it to a string
2. Looks the string up in the `cases` mapping, where each key is a case value and each value is the
   step to execute when that case matches (e.g. `{"red": "$steps.on_red", "blue": "$steps.on_blue"}`)
3. If `case_insensitive` is enabled, the comparison ignores letter case
4. If a case matches, execution continues to that case's step and all other branches terminate
5. If no case matches, execution continues to the steps listed in `default_next_steps`
6. If no case matches and `default_next_steps` is empty, the branch terminates

Because the input value is converted to a string before matching, non-string values match their
string representation: `True` matches the key `"True"`, `1.0` matches `"1.0"` (not `"1"`), and a
missing/None value matches `"None"`. Each target step may appear at most once across `cases` and
`default_next_steps` — to route several case values to the same logic, point each case at its own
step or normalize the value upstream (e.g. with an Expression block).

## Common Use Cases

- **Routing by classification result**: Send images down different processing paths based on the
  top class predicted by a classification model (e.g. "damaged" → alert branch, "ok" → logging
  branch, anything else → default review branch)
- **Mode-based pipelines**: Use a workflow input parameter (e.g. `$inputs.mode`) to select between
  alternative processing branches at runtime without editing the workflow
- **Multi-way alerting**: Route to different notification blocks (email, Slack, webhook) depending
  on a severity or category value computed earlier in the workflow
- **Replacing chained conditions**: Collapse a ladder of Continue If blocks comparing the same
  value against different constants into a single, easier-to-read block

## Connecting to Other Blocks

This block controls workflow execution flow and can be connected:

- **After classification or detection blocks** to branch on predicted classes, counts, or other
  prediction properties (often via a Property Definition or Expression block that extracts the
  value to switch on)
- **After workflow inputs** to select a branch from a runtime parameter
- **Before any downstream blocks** (models, notifications, sinks) that should only run for a
  specific case — each case target becomes the head of its own execution branch
- **With a default branch** wired via `default_next_steps` to handle unmatched values, or left
  empty to simply stop when nothing matches


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/switch_case@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `value` | `Union[bool, float, int, str]` | Value to match against the case keys. Typically a selector referencing a workflow input or a step output (e.g. $inputs.mode or $steps.classifier.top). The value is converted to a string before comparison, so booleans match keys 'True'/'False', 1.0 matches '1.0' and a None value matches 'None'.. | ✅ |
| `case_insensitive` | `bool` | When enabled, case values are matched ignoring letter case (e.g. value 'RED' matches case key 'red').. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Switch Case` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Slack Notification`](slack_notification.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Delta Filter`](delta_filter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Overlap Analysis`](overlap_analysis.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Switch Case`](switch_case.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Cache Set`](cache_set.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Property Definition`](property_definition.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Detections Filter`](detections_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Dimension Collapse`](dimension_collapse.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Continue If`](continue_if.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Switch Case` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `value` (*[`*`](../kinds/wildcard.md)*): Value to match against the case keys. Typically a selector referencing a workflow input or a step output (e.g. $inputs.mode or $steps.classifier.top). The value is converted to a string before comparison, so booleans match keys 'True'/'False', 1.0 matches '1.0' and a None value matches 'None'..
        - `cases` (*step*): Mapping of case value to the step that should execute when `value` matches it, e.g. {"red": "$steps.on_red", "blue": "$steps.on_blue"}. Each target step may appear at most once across `cases` and `default_next_steps`..
        - `default_next_steps` (*step*): Steps to execute when no case matches. Leave empty to terminate the branch when nothing matches..

    - output
    




??? tip "Example JSON definition of step `Switch Case` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/switch_case@v1",
	    "value": "$steps.classifier.top",
	    "cases": {
	        "blue": "$steps.on_blue",
	        "red": "$steps.on_red"
	    },
	    "case_insensitive": false,
	    "default_next_steps": [
	        "$steps.fallback"
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

