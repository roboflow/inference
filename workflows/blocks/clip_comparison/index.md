
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

    - inputs: [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PP-OCR`](ppocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`GeoTag Detection`](geo_tag_detection.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`CSV Formatter`](csv_formatter.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`S3 Sink`](s3_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Cache Get`](cache_get.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`PLC Reader`](plc_reader.md), [`Clip Comparison`](clip_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Identify Outliers`](identify_outliers.md), [`Cache Set`](cache_set.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Image Blur`](image_blur.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

    - inputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Image Slicer`](image_slicer.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Polygon Visualization`](polygon_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Crop Visualization`](crop_visualization.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Size Measurement`](size_measurement.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Google Gemini`](google_gemini.md), [`Depth Estimation`](depth_estimation.md)
    - outputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Frame Delay`](frame_delay.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PLC Reader`](plc_reader.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Clip Comparison`](clip_comparison.md), [`Polygon Visualization`](polygon_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenRouter`](open_router.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Line Counter`](line_counter.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

