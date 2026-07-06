
# Continue If



??? "Class: `ContinueIfBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/flow_control/continue_if/v1.py">inference.core.workflows.core_steps.flow_control.continue_if.v1.ContinueIfBlockV1</a>
    



Conditionally control workflow execution by evaluating custom logic statements and either continuing to specified next steps or terminating the current branch based on the condition result, enabling dynamic branching, conditional processing, and workflow control flow.

## How This Block Works

This block evaluates a conditional statement and controls whether the workflow branch continues execution or stops. The block:

1. Takes a conditional statement (using a query language syntax) and evaluation parameters as input
2. Builds an evaluation function from the conditional statement definition
3. Evaluates the condition using the provided evaluation parameters (which can reference workflow inputs, step outputs, or other dynamic values)
4. If the condition evaluates to `true`:
   - Continues execution to the specified `next_steps` blocks
   - If a `stop_delay` is configured, records the current time to enable delayed termination
5. If the condition evaluates to `false`:
   - Terminates the current workflow branch (stops execution of downstream blocks in this branch)
   - If `stop_delay` was previously triggered and the delay period hasn't elapsed, continues execution to `next_steps` for the remaining delay duration
6. Returns flow control directives that either continue execution to the next steps or terminate the branch

The block uses a query language system that supports binary comparisons (equality, inequality, greater than, less than, etc.) between dynamic values (from workflow data) and static values. Conditions can check numeric values, string values, or other data types. The `stop_delay` feature allows the branch to remain active for a short period after a condition becomes false, which is useful for handling transient states or maintaining execution during brief condition fluctuations (e.g., keeping a workflow active for a few seconds after a detection count drops below threshold).

## Common Use Cases

- **Conditional Processing Based on Detection Counts**: Continue processing only when the number of detected objects exceeds a threshold (e.g., process alerts only when 3+ objects are detected, skip processing when count is below threshold)
- **Dynamic Quality Control**: Evaluate image quality metrics, detection confidence scores, or model outputs and continue workflow execution only when quality criteria are met, terminating branches that don't meet standards
- **Conditional Notifications**: Send notifications or trigger actions only when specific conditions are met (e.g., continue to notification blocks when confidence scores are above 0.9, or when specific object classes are detected)
- **Branch Filtering and Routing**: Route workflow execution to different branches based on dynamic conditions, allowing one path to continue while others terminate (e.g., continue video recording branch when motion is detected, terminate when no activity)
- **Threshold-Based Actions**: Execute downstream blocks only when values meet thresholds (e.g., continue to data storage when detection count > 5, terminate otherwise; continue processing when temperature > threshold, skip when below)
- **Transient State Handling**: Use `stop_delay` to handle brief condition changes by keeping branches active for a short period after conditions become false, preventing rapid on/off toggling in response to temporary fluctuations

## Connecting to Other Blocks

This block controls workflow execution flow and can be connected:

- **After detection or analysis blocks** (e.g., Object Detection, Classification, Keypoint Detection) to evaluate detection counts, confidence scores, class names, or other prediction results and conditionally continue processing based on the analysis results
- **After data processing blocks** (e.g., Property Definition, Expression, Delta Filter) to evaluate computed values, metrics, or processed data and control whether subsequent blocks execute based on the processed results
- **Before notification blocks** (e.g., Email Notification, Slack Notification, Twilio SMS Notification) to conditionally trigger notifications only when specific conditions are met, preventing unnecessary alerts
- **Before data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to conditionally save or send data only when certain criteria are satisfied, filtering what gets stored or transmitted
- **Between workflow stages** to create conditional processing paths, where different branches execute based on dynamic conditions, enabling complex workflow logic and decision trees
- **In parallel branches** to create multiple conditional paths, allowing different parts of a workflow to continue or terminate independently based on their respective conditions


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/continue_if@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `condition_statement` | `StatementGroup` | Define the conditional logic using the query language syntax. Specifies the condition to evaluate (e.g., comparisons, equality checks, numeric comparisons). The condition is built using StatementGroup syntax with binary statements that compare dynamic operands (referenced in evaluation_parameters) against static values using comparators like (Number) ==, (Number) >, (Number) <, (String) ==, etc. Example: Compare a dynamic value 'left' against static value 1 using (Number) ==.. | ❌ |
| `stop_delay` | `float` | Number of seconds to continue execution after the condition becomes false, before terminating the branch. If the condition was previously true and then becomes false, execution continues to next_steps for this delay duration before terminating. This is useful for handling transient state changes or preventing rapid on/off toggling. Must be greater than 0 to take effect. Set to 0 (default) to terminate immediately when condition becomes false.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Continue If` in version `v1`.

    - inputs: [`Detections Filter`](detections_filter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Cache Get`](cache_get.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Switch Case`](switch_case.md), [`Buffer`](buffer.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`LMM`](lmm.md), [`Inner Workflow`](inner_workflow.md), [`Velocity`](velocity.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Expression`](expression.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MQTT Writer`](mqtt_writer.md), [`Cache Set`](cache_set.md), [`Delta Filter`](delta_filter.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Path Deviation`](path_deviation.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Gemma API`](google_gemma_api.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Property Definition`](property_definition.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Data Aggregator`](data_aggregator.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Image Slicer`](image_slicer.md), [`Detections Merge`](detections_merge.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Continue If` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `evaluation_parameters` (*[`*`](../kinds/wildcard.md)*): Dictionary mapping operand names (used in condition_statement) to actual values from the workflow. These parameters provide the dynamic data that gets evaluated in the conditional statement. Keys match operand names in the condition (e.g., 'left', 'right'), and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'left': '$steps.detection.count', 'threshold': 5} where 'left' is referenced in the condition_statement..
        - `next_steps` (*step*): List of workflow steps to execute if the condition evaluates to true. These steps receive control flow when the condition is satisfied, allowing the workflow branch to continue execution. If empty, the branch terminates even when the condition is true. Each step selector references a block in the workflow that should execute when the condition passes..
        - `stop_delay` (*[`float`](../kinds/float.md)*): Number of seconds to continue execution after the condition becomes false, before terminating the branch. If the condition was previously true and then becomes false, execution continues to next_steps for this delay duration before terminating. This is useful for handling transient state changes or preventing rapid on/off toggling. Must be greater than 0 to take effect. Set to 0 (default) to terminate immediately when condition becomes false..

    - output
    




??? tip "Example JSON definition of step `Continue If` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/continue_if@v1",
	    "condition_statement": {
	        "statements": [
	            {
	                "comparator": {
	                    "type": "(Number) =="
	                },
	                "left_operand": {
	                    "operand_name": "left",
	                    "type": "DynamicOperand"
	                },
	                "right_operand": {
	                    "type": "StaticOperand",
	                    "value": 1
	                },
	                "type": "BinaryStatement"
	            }
	        ],
	        "type": "StatementGroup"
	    },
	    "evaluation_parameters": {
	        "left": "$inputs.some"
	    },
	    "next_steps": [
	        "$steps.on_true"
	    ],
	    "stop_delay": 5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

