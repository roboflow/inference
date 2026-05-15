
# Stability AI Inpainting



??? "Class: `StabilityAIInpaintingBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/stability_ai/inpainting/v1.py">inference.core.workflows.core_steps.models.foundation.stability_ai.inpainting.v1.StabilityAIInpaintingBlockV1</a>
    



The block wraps 
[Stability AI inpainting API](https://platform.stability.ai/docs/legacy/grpc-api/features/inpainting#Python) and 
let users use instance segmentation results to change the content of images in a creative way.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stability_ai_inpainting@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Prompt to inpainting model (what you wish to see).. | ✅ |
| `negative_prompt` | `str` | Negative prompt to inpainting model (what you do not wish to see).. | ✅ |
| `api_key` | `str` | Your Stability AI API key.. | ✅ |
| `invert_segmentation_mask` | `bool` | Invert segmentation mask to inpaint background instead of foreground.. | ✅ |
| `preset` | `StabilityAIPresets` | Optional preset to apply when outpainting the image (what you wish to see). If not provided, the image will be outpainted without any preset. Avaliable presets: 3d-model, analog-film, anime, cinematic, comic-book, digital-art, enhance, fantasy-art, isometric, line-art, low-poly, modeling-compound, neon-punk, origami, photographic, pixel-art, tile-texture. | ❌ |
| `seed` | `int` | A specific value that is used to guide the 'randomness' of the generation. If not provided, a random seed is used. Must be a number between 0 and 4294967294. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stability AI Inpainting` in version `v1`.

    - inputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`JSON Parser`](json_parser.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Detection Offset`](detection_offset.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Image Blur`](image_blur.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT`](sift.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stability AI Inpainting` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to inpaint..
        - `segmentation_mask` (*[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)*): Model predictions from segmentation model..
        - `prompt` (*[`string`](../kinds/string.md)*): Prompt to inpainting model (what you wish to see)..
        - `negative_prompt` (*[`string`](../kinds/string.md)*): Negative prompt to inpainting model (what you do not wish to see)..
        - `api_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Your Stability AI API key..
        - `invert_segmentation_mask` (*[`boolean`](../kinds/boolean.md)*): Invert segmentation mask to inpaint background instead of foreground..
        - `seed` (*[`integer`](../kinds/integer.md)*): A specific value that is used to guide the 'randomness' of the generation. If not provided, a random seed is used. Must be a number between 0 and 4294967294.

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Stability AI Inpainting` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stability_ai_inpainting@v1",
	    "image": "$inputs.image",
	    "segmentation_mask": "$steps.model.predictions",
	    "prompt": "my prompt",
	    "negative_prompt": "my prompt",
	    "api_key": "xxx-xxx",
	    "invert_segmentation_mask": "<block_does_not_provide_example>",
	    "preset": "3d-model",
	    "seed": 200
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

