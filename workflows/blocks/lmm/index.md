
# LMM

!!! warning "Deprecated"

    This block is deprecated and may be removed in a future release.



??? "Class: `LMMBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/lmm/v1.py">inference.core.workflows.core_steps.models.foundation.lmm.v1.LMMBlockV1</a>
    



Ask a question to a Large Multimodal Model (LMM) with an image and text.

You can specify arbitrary text prompts to an LMMBlock.

The LLMBlock supports two LMMs:

- OpenAI's GPT-4 with Vision;

You need to provide your OpenAI API key to use the GPT-4 with Vision model. 

_If you want to classify an image into one or more categories, we recommend using the 
dedicated LMMForClassificationBlock._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/lmm@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Holds unconstrained text prompt to LMM mode. | ✅ |
| `lmm_type` | `str` | Type of LMM to be used. | ✅ |
| `lmm_config` | `LMMConfig` | Configuration of LMM. | ❌ |
| `remote_api_key` | `str` | Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`.. | ✅ |
| `json_output` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `LMM` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`GLM-OCR`](glmocr.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Image Blur`](image_blur.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Pixelate Visualization`](pixelate_visualization.md), [`EasyOCR`](easy_ocr.md), [`SIFT`](sift.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`Grid Visualization`](grid_visualization.md), [`LMM`](lmm.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CSV Formatter`](csv_formatter.md), [`S3 Sink`](s3_sink.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen-VL`](qwen_vl.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Webhook Sink`](webhook_sink.md), [`Color Visualization`](color_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`CogVLM`](cog_vlm.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Camera Focus`](camera_focus.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Label Visualization`](label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Calibration`](camera_calibration.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Morphological Transformation`](morphological_transformation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Perspective Correction`](perspective_correction.md)
    - outputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Delta Filter`](delta_filter.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Consensus`](detections_consensus.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`JSON Parser`](json_parser.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Stitch Images`](stitch_images.md), [`OpenRouter`](open_router.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Time in Zone`](timein_zone.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Buffer`](buffer.md), [`EasyOCR`](easy_ocr.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detection Offset`](detection_offset.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dominant Color`](dominant_color.md), [`SIFT Comparison`](sift_comparison.md), [`Identify Changes`](identify_changes.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CSV Formatter`](csv_formatter.md), [`Object Detection Model`](object_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Qwen-VL`](qwen_vl.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Velocity`](velocity.md), [`Qwen3.5`](qwen3.5.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`Detection Event Log`](detection_event_log.md), [`CogVLM`](cog_vlm.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Inner Workflow`](inner_workflow.md), [`SmolVLM2`](smol_vlm2.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Merge`](detections_merge.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`Rate Limiter`](rate_limiter.md), [`LMM For Classification`](lmm_for_classification.md), [`Text Display`](text_display.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Detections Combine`](detections_combine.md), [`Image Stack`](image_stack.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Perspective Correction`](perspective_correction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Focus`](camera_focus.md), [`Seg Preview`](seg_preview.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Trace Visualization`](trace_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Byte Tracker`](byte_tracker.md), [`GLM-OCR`](glmocr.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Image Blur`](image_blur.md), [`Cache Get`](cache_get.md), [`Detections Stitch`](detections_stitch.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Local File Sink`](local_file_sink.md), [`Property Definition`](property_definition.md), [`Icon Visualization`](icon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Size Measurement`](size_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen3-VL`](qwen3_vl.md), [`Identify Outliers`](identify_outliers.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Image Slicer`](image_slicer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Webhook Sink`](webhook_sink.md), [`Color Visualization`](color_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Expression`](expression.md), [`Google Vision OCR`](google_vision_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Distance Measurement`](distance_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Moondream2`](moondream2.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Template Matching`](template_matching.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Line Counter`](line_counter.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Label Visualization`](label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Halo Visualization`](halo_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Clip Comparison`](clip_comparison.md), [`Continue If`](continue_if.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`OCR Model`](ocr_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Overlap Analysis`](overlap_analysis.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`LMM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Holds unconstrained text prompt to LMM mode.
        - `lmm_type` (*[`string`](../kinds/string.md)*): Type of LMM to be used.
        - `remote_api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Holds API key required to call LMM model - in current state of development, we require OpenAI key when `lmm_type=gpt_4v`..

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `LMM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/lmm@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "lmm_type": "gpt_4v",
	    "lmm_config": {
	        "gpt_image_detail": "low",
	        "gpt_model_version": "gpt-4o",
	        "max_tokens": 200
	    },
	    "remote_api_key": "xxx-xxx",
	    "json_output": {
	        "count": "number of cats in the picture"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

