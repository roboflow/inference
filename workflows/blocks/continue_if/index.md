
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

    - inputs: [`Cache Set`](cache_set.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Delta Filter`](delta_filter.md), [`Expression`](expression.md), [`Detection Offset`](detection_offset.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Switch Case`](switch_case.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
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

