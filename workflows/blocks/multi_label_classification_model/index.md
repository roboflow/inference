
# Multi-Label Classification Model



## v3

??? "Class: `RoboflowMultiLabelClassificationModelBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_label_classification/v3.py">inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v3.RoboflowMultiLabelClassificationModelBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_multi_label_classification_model@v3`to add the block as
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
    Check what blocks you can connect to `Multi-Label Classification Model` in version `v3`.

    - inputs: [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`CogVLM`](cog_vlm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Slack Notification`](slack_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Identify Changes`](identify_changes.md), [`OCR Model`](ocr_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen-VL`](qwen_vl.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Google Gemini`](google_gemini.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Object Detection Model`](object_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Qwen-VL`](qwen_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Multi-Label Classification Model` in version `v3`  has.

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



??? tip "Example JSON definition of step `Multi-Label Classification Model` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_multi_label_classification_model@v3",
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

??? "Class: `RoboflowMultiLabelClassificationModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_label_classification/v2.py">inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v2.RoboflowMultiLabelClassificationModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_multi_label_classification_model@v2`to add the block as
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
    Check what blocks you can connect to `Multi-Label Classification Model` in version `v2`.

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Slack Notification`](slack_notification.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Trace Visualization`](trace_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Local File Sink`](local_file_sink.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Visualization`](mask_visualization.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Object Detection Model`](object_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Qwen-VL`](qwen_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Multi-Label Classification Model` in version `v2`  has.

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



??? tip "Example JSON definition of step `Multi-Label Classification Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
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

??? "Class: `RoboflowMultiLabelClassificationModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/multi_label_classification/v1.py">inference.core.workflows.core_steps.models.roboflow.multi_label_classification.v1.RoboflowMultiLabelClassificationModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a multi-label classification model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_multi_label_classification_model@v1`to add the block as
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
    Check what blocks you can connect to `Multi-Label Classification Model` in version `v1`.

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Slack Notification`](slack_notification.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Trace Visualization`](trace_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Local File Sink`](local_file_sink.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Visualization`](mask_visualization.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemma`](google_gemma.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Cache Get`](cache_get.md), [`Heatmap Visualization`](heatmap_visualization.md), [`CogVLM`](cog_vlm.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Corner Visualization`](corner_visualization.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenRouter`](open_router.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Florence-2 Model`](florence2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Moondream2`](moondream2.md), [`Seg Preview`](seg_preview.md), [`Trace Visualization`](trace_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`QR Code Generator`](qr_code_generator.md), [`Anthropic Claude`](anthropic_claude.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dot Visualization`](dot_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Path Deviation`](path_deviation.md), [`Google Gemini`](google_gemini.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Threshold`](image_threshold.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Local File Sink`](local_file_sink.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Time in Zone`](timein_zone.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`GLM-OCR`](glmocr.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Segment Anything 2 Model`](segment_anything2_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Multi-Label Classification Model` in version `v1`  has.

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



??? tip "Example JSON definition of step `Multi-Label Classification Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_multi_label_classification_model@v1",
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

