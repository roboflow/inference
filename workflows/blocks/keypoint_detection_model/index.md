
# Keypoint Detection Model



## v3

??? "Class: `RoboflowKeypointDetectionModelBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/keypoint_detection/v3.py">inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v3.RoboflowKeypointDetectionModelBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_keypoint_detection_model@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | not available. | ✅ |
| `custom_confidence` | `float` | not available. | ✅ |
| `keypoint_confidence` | `float` | Confidence threshold to predict a keypoint as visible.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Keypoint Detection Model` in version `v3`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`PLC Reader`](plc_reader.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)
    - outputs: [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Trace Visualization`](trace_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Filter`](detections_filter.md), [`Qwen3.5`](qwen3.5.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Keypoint Detection Model` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): not available.
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): not available.
        - `keypoint_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold to predict a keypoint as visible..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)): Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Keypoint Detection Model` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_keypoint_detection_model@v3",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": "<block_does_not_provide_example>",
	    "keypoint_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `RoboflowKeypointDetectionModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/keypoint_detection/v2.py">inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v2.RoboflowKeypointDetectionModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_keypoint_detection_model@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `keypoint_confidence` | `float` | Confidence threshold to predict a keypoint as visible.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Keypoint Detection Model` in version `v2`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Size Measurement`](size_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`PLC Reader`](plc_reader.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)
    - outputs: [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Trace Visualization`](trace_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Filter`](detections_filter.md), [`Qwen3.5`](qwen3.5.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Keypoint Detection Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `keypoint_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold to predict a keypoint as visible..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)): Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Keypoint Detection Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_keypoint_detection_model@v2",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "keypoint_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `RoboflowKeypointDetectionModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/keypoint_detection/v1.py">inference.core.workflows.core_steps.models.roboflow.keypoint_detection.v1.RoboflowKeypointDetectionModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a keypoint detection model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_keypoint_detection_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `keypoint_confidence` | `float` | Confidence threshold to predict a keypoint as visible.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Keypoint Detection Model` in version `v1`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Size Measurement`](size_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`PLC Reader`](plc_reader.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md)
    - outputs: [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Filter`](detections_filter.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Image Threshold`](image_threshold.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Transformation`](detections_transformation.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Keypoint Detection Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `keypoint_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold to predict a keypoint as visible..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `inference_id` ([`string`](../kinds/string.md)): String value.
        - `predictions` ([`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)): Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Keypoint Detection Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_keypoint_detection_model@v1",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "keypoint_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

