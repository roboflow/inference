
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

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Overlap Filter`](overlap_filter.md), [`Rate Limiter`](rate_limiter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Expression`](expression.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Property Definition`](property_definition.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Inner Workflow`](inner_workflow.md), [`Detection Offset`](detection_offset.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Delta Filter`](delta_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)
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

