
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

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`GLM-OCR`](glmocr.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen-VL`](qwen_vl.md)

    
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

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`GLM-OCR`](glmocr.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Analysis`](overlap_analysis.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen-VL`](qwen_vl.md)

    
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

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Slack Notification`](slack_notification.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`S3 Sink`](s3_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`GLM-OCR`](glmocr.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Analysis`](overlap_analysis.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen-VL`](qwen_vl.md)

    
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

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Slack Notification`](slack_notification.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`S3 Sink`](s3_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Slack Notification`](slack_notification.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Path Deviation`](path_deviation.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Local File Sink`](local_file_sink.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Current Time`](current_time.md), [`Detections Filter`](detections_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
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

