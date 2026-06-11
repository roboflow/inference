
# Semantic Segmentation Model



## v2

??? "Class: `RoboflowSemanticSegmentationModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/semantic_segmentation/v2.py">inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2.RoboflowSemanticSegmentationModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a semantic segmentation model hosted on or uploaded to Roboflow.

Semantic segmentation assigns a class label to every pixel in the image, producing a
dense segmentation mask rather than per-object bounding boxes or instance masks.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_semantic_segmentation_model@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How confidence thresholds are determined.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Semantic Segmentation Model` in version `v2`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Object Detection Model`](object_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5`](qwen3.5.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Webhook Sink`](webhook_sink.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Semantic Segmentation Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How confidence thresholds are determined..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`semantic_segmentation_prediction`](../kinds/semantic_segmentation_prediction.md)): Prediction with per-pixel class label and confidence for semantic segmentation.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Semantic Segmentation Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `RoboflowSemanticSegmentationModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/semantic_segmentation/v1.py">inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v1.RoboflowSemanticSegmentationModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on a semantic segmentation model hosted on or uploaded to Roboflow.

Semantic segmentation assigns a class label to every pixel in the image, producing a
dense segmentation mask rather than per-object bounding boxes or instance masks.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_semantic_segmentation_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Semantic Segmentation Model` in version `v1`.

    - inputs: [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Contrast Equalization`](contrast_equalization.md), [`Relative Static Crop`](relative_static_crop.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Grid Visualization`](grid_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dot Visualization`](dot_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Threshold`](image_threshold.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md)
    - outputs: [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Object Detection Model`](object_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Moondream2`](moondream2.md), [`SAM 3`](sam3.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5`](qwen3.5.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`GLM-OCR`](glmocr.md), [`Webhook Sink`](webhook_sink.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Semantic Segmentation Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`semantic_segmentation_prediction`](../kinds/semantic_segmentation_prediction.md)): Prediction with per-pixel class label and confidence for semantic segmentation.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Semantic Segmentation Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
	    "images": "$inputs.image",
	    "model_id": "my_project/3"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

