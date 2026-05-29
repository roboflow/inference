
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

    - inputs: [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Blur Visualization`](blur_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Velocity`](velocity.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Detections Merge`](detections_merge.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Local File Sink`](local_file_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Camera Focus`](camera_focus.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Inner Workflow`](inner_workflow.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Cosine Similarity`](cosine_similarity.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Webhook Sink`](webhook_sink.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Overlap Filter`](overlap_filter.md), [`Image Threshold`](image_threshold.md), [`Cache Get`](cache_get.md), [`Line Counter`](line_counter.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemma`](google_gemma.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Rate Limiter`](rate_limiter.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Transformation`](detections_transformation.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Data Aggregator`](data_aggregator.md), [`EasyOCR`](easy_ocr.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Depth Estimation`](depth_estimation.md), [`Stitch Images`](stitch_images.md), [`Grid Visualization`](grid_visualization.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Qwen3-VL`](qwen3_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Property Definition`](property_definition.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Detection Offset`](detection_offset.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Image Blur`](image_blur.md), [`Expression`](expression.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Delta Filter`](delta_filter.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Size Measurement`](size_measurement.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`QR Code Detection`](qr_code_detection.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections Stitch`](detections_stitch.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md)

    
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

