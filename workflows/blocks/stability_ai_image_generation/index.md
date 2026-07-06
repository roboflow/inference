
# Stability AI Image Generation



??? "Class: `StabilityAIImageGenBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/stability_ai/image_gen/v1.py">inference.core.workflows.core_steps.models.foundation.stability_ai.image_gen.v1.StabilityAIImageGenBlockV1</a>
    



The block wraps [Stability AI image generation API](https://platform.stability.ai/docs/api-reference#tag/Generate) and let users generate new images from text, or create variations of existing images.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stability_ai_image_gen@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `strength` | `float` | controls how much influence the image parameter has on the generated image. A value of 0 would yield an image that is identical to the input. A value of 1 would be as if you passed in no image at all.. | ✅ |
| `prompt` | `str` | Prompt to generate new images from text (what you wish to see). | ✅ |
| `negative_prompt` | `str` | Negative prompt to image generation model (what you do not wish to see). | ✅ |
| `model` | `str` | choose one of {'core', 'ultra', 'sd3'}. Default 'core' . | ✅ |
| `api_key` | `str` | Your Stability AI API key. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stability AI Image Generation` in version `v1`.

    - inputs: [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen-VL`](qwen_vl.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Identify Changes`](identify_changes.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Current Time`](current_time.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`PLC Writer`](plc_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md)
    - outputs: [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`Dominant Color`](dominant_color.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Label Visualization`](label_visualization.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stability AI Image Generation` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to use as the starting point for the generation..
        - `strength` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): controls how much influence the image parameter has on the generated image. A value of 0 would yield an image that is identical to the input. A value of 1 would be as if you passed in no image at all..
        - `prompt` (*[`string`](../kinds/string.md)*): Prompt to generate new images from text (what you wish to see).
        - `negative_prompt` (*[`string`](../kinds/string.md)*): Negative prompt to image generation model (what you do not wish to see).
        - `model` (*[`string`](../kinds/string.md)*): choose one of {'core', 'ultra', 'sd3'}. Default 'core' .
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Your Stability AI API key.

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Stability AI Image Generation` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stability_ai_image_gen@v1",
	    "image": "$inputs.image",
	    "strength": 0.3,
	    "prompt": "my prompt",
	    "negative_prompt": "my prompt",
	    "model": "my prompt",
	    "api_key": "xxx-xxx"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

