
# Clip Comparison



## v2

??? "Class: `ClipComparisonBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/clip_comparison/v2.py">inference.core.workflows.core_steps.models.foundation.clip_comparison.v2.ClipComparisonBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Use the OpenAI CLIP zero-shot classification model to classify images.

This block accepts an image and a list of text prompts. The block then returns the 
similarity of each text label to the provided image.

This block is useful for classifying images without having to train a fine-tuned 
classification model. For example, you could use CLIP to classify the type of vehicle 
in an image, or if an image contains NSFW material.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/clip_comparison@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `classes` | `List[str]` | List of classes to calculate similarity against each input image. | ✅ |
| `version` | `str` | Variant of CLIP model. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Clip Comparison` in version `v2`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Qwen-VL`](qwen_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`S3 Sink`](s3_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`Buffer`](buffer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`OCR Model`](ocr_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`GLM-OCR`](glmocr.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`SAM 3`](sam3.md), [`Stitch Images`](stitch_images.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Cache Get`](cache_get.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Clip Comparison` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to calculate similarity against each input image.
        - `version` (*[`string`](../kinds/string.md)*): Variant of CLIP model.

    - output
    
        - `similarities` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.
        - `max_similarity` ([`float_zero_to_one`](../kinds/float_zero_to_one.md)): `float` value in range `[0.0, 1.0]`.
        - `most_similar_class` ([`string`](../kinds/string.md)): String value.
        - `min_similarity` ([`float_zero_to_one`](../kinds/float_zero_to_one.md)): `float` value in range `[0.0, 1.0]`.
        - `least_similar_class` ([`string`](../kinds/string.md)): String value.
        - `classification_predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.



??? tip "Example JSON definition of step `Clip Comparison` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/clip_comparison@v2",
	    "images": "$inputs.image",
	    "classes": [
	        "a",
	        "b",
	        "c"
	    ],
	    "version": "ViT-B-16"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `ClipComparisonBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/clip_comparison/v1.py">inference.core.workflows.core_steps.models.foundation.clip_comparison.v1.ClipComparisonBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Use the OpenAI CLIP zero-shot classification model to classify images.

This block accepts an image and a list of text prompts. The block then returns the 
similarity of each text label to the provided image.

This block is useful for classifying images without having to train a fine-tuned 
classification model. For example, you could use CLIP to classify the type of vehicle 
in an image, or if an image contains NSFW material.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/clip_comparison@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `texts` | `List[str]` | List of texts to calculate similarity against each input image. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Clip Comparison` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Circle Visualization`](circle_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenRouter`](open_router.md), [`Qwen-VL`](qwen_vl.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch Images`](stitch_images.md), [`Image Contours`](image_contours.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Camera Focus`](camera_focus.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma`](google_gemma.md), [`Buffer`](buffer.md), [`Contrast Equalization`](contrast_equalization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Set`](cache_set.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Florence-2 Model`](florence2_model.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Size Measurement`](size_measurement.md), [`Email Notification`](email_notification.md), [`Object Detection Model`](object_detection_model.md), [`LMM For Classification`](lmm_for_classification.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Clip Comparison` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `texts` (*[`list_of_values`](../kinds/list_of_values.md)*): List of texts to calculate similarity against each input image.

    - output
    
        - `similarity` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `prediction_type` ([`prediction_type`](../kinds/prediction_type.md)): String value with type of prediction.



??? tip "Example JSON definition of step `Clip Comparison` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/clip_comparison@v1",
	    "images": "$inputs.image",
	    "texts": [
	        "a",
	        "b",
	        "c"
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

