
# Single-Label Classification Model



## v3

??? "Class: `RoboflowClassificationModelBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_class_classification/v3.py">inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v3.RoboflowClassificationModelBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-class classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_classification_model@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How to determine the confidence threshold.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Single-Label Classification Model` in version `v3`.

    - inputs: [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Current Time`](current_time.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Moondream2`](moondream2.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Event Writer`](event_writer.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Single-Label Classification Model` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How to determine the confidence threshold..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Single-Label Classification Model` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_classification_model@v3",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3,
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

??? "Class: `RoboflowClassificationModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_class_classification/v2.py">inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v2.RoboflowClassificationModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-class classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_classification_model@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Single-Label Classification Model` in version `v2`.

    - inputs: [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Slack Notification`](slack_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`JSON Parser`](json_parser.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MQTT Writer`](mqtt_writer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`PLC Reader`](plc_reader.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Moondream2`](moondream2.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Event Writer`](event_writer.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Object Detection Model`](object_detection_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Single-Label Classification Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Single-Label Classification Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_classification_model@v2",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
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

??? "Class: `RoboflowClassificationModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_class_classification/v1.py">inference.core.workflows.core_steps.models.roboflow.multi_class_classification.v1.RoboflowClassificationModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-class classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_classification_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Single-Label Classification Model` in version `v1`.

    - inputs: [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Slack Notification`](slack_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`JSON Parser`](json_parser.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MQTT Writer`](mqtt_writer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`PLC Reader`](plc_reader.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Qwen-VL`](qwen_vl.md), [`Cache Get`](cache_get.md), [`Triangle Visualization`](triangle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Current Time`](current_time.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Single-Label Classification Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `inference_id` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Single-Label Classification Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_classification_model@v1",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
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

