
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

    - inputs: [`Cache Set`](cache_set.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Delta Filter`](delta_filter.md), [`Expression`](expression.md), [`Detection Offset`](detection_offset.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Switch Case`](switch_case.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
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

