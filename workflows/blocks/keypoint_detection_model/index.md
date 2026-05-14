
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

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Stack`](image_stack.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`GLM-OCR`](glmocr.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5`](qwen3.5.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Webhook Sink`](webhook_sink.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dot Visualization`](dot_visualization.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md)

    
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

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Image Stack`](image_stack.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Outliers`](identify_outliers.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`GLM-OCR`](glmocr.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5`](qwen3.5.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Webhook Sink`](webhook_sink.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dot Visualization`](dot_visualization.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md)

    
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

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Image Stack`](image_stack.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Outliers`](identify_outliers.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`GLM-OCR`](glmocr.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Object Detection Model`](object_detection_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Image Blur`](image_blur.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Seg Preview`](seg_preview.md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`Contrast Equalization`](contrast_equalization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Velocity`](velocity.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Google Gemma API`](google_gemma_api.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md)

    
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

