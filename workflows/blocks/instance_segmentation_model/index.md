
# Instance Segmentation Model



## v4

??? "Class: `RoboflowInstanceSegmentationModelBlockV4`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v4.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v4.RoboflowInstanceSegmentationModelBlockV4</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).

This version of block introduces breaking change in behaviour of mask construction - it uses 
`rle` format instead `polygon` making it possible to retrieve 
shapes of any kind from remote server.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v4`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How confidence thresholds are determined.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v4`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Distance Measurement`](distance_measurement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Path Deviation`](path_deviation.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Moondream2`](moondream2.md), [`Qwen3.5`](qwen3.5.md), [`Camera Focus`](camera_focus.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`GLM-OCR`](glmocr.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Detection Offset`](detection_offset.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v4`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How confidence thresholds are determined..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v4`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
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




## v3

??? "Class: `RoboflowInstanceSegmentationModelBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3.RoboflowInstanceSegmentationModelBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How confidence thresholds are determined.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v3`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Distance Measurement`](distance_measurement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Path Deviation`](path_deviation.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Moondream2`](moondream2.md), [`Qwen3.5`](qwen3.5.md), [`Camera Focus`](camera_focus.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`GLM-OCR`](glmocr.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Detection Offset`](detection_offset.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How confidence thresholds are determined..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `RoboflowInstanceSegmentationModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v2.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v2.RoboflowInstanceSegmentationModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v2`.

    - inputs: [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Distance Measurement`](distance_measurement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Path Deviation`](path_deviation.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Moondream2`](moondream2.md), [`Qwen3.5`](qwen3.5.md), [`Camera Focus`](camera_focus.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`GLM-OCR`](glmocr.md), [`Corner Visualization`](corner_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Detection Offset`](detection_offset.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `RoboflowInstanceSegmentationModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v1.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v1.RoboflowInstanceSegmentationModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v1`.

    - inputs: [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Distance Measurement`](distance_measurement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Cache Set`](cache_set.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`Overlap Filter`](overlap_filter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`GLM-OCR`](glmocr.md), [`Camera Focus`](camera_focus.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Consensus`](detections_consensus.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`LMM For Classification`](lmm_for_classification.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`string`](../kinds/string.md)): String value.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v1",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

