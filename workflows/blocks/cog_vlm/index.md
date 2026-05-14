
# CogVLM



??? "Class: `CogVLMBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/cog_vlm/v1.py">inference.core.workflows.core_steps.models.foundation.cog_vlm.v1.CogVLMBlockV1</a>
    




!!! Warning "CogVLM reached **End Of Life**"

    Due to dependencies conflicts with newer models and security vulnerabilities discovered in `transformers`
    library patched in the versions of library incompatible with the model we announced End Of Life for CogVLM
    support in `inference`, effective since release `0.38.0`.
    
    We are leaving this block in ecosystem until release `0.42.0` for clients to get informed about change that 
    was introduced.
    
    Starting as of now, all Workflows using the block stop being functional (runtime error will be raised), 
    after inference release `0.42.0` - this block will be removed and Execution Engine will raise compilation 
    error seeing the block in Workflow definition. 


Ask a question to CogVLM, an open source vision-language model.

This model requires a GPU and can only be run on self-hosted devices, and is not available on the Roboflow Hosted API.

_This model was previously part of the LMM block._


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/cog_vlm@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Text prompt to the CogVLM model. | ✅ |
| `json_output_format` | `Dict[str, str]` | Holds dictionary that maps name of requested output field into its description. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `CogVLM` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`GLM-OCR`](glmocr.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Dot Visualization`](dot_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`OCR Model`](ocr_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Depth Estimation`](depth_estimation.md), [`LMM`](lmm.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Contours`](image_contours.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`Polygon Visualization`](polygon_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Detections Transformation`](detections_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Rate Limiter`](rate_limiter.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Barcode Detection`](barcode_detection.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Delta Filter`](delta_filter.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Merge`](detections_merge.md), [`SAM 3`](sam3.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Detections Filter`](detections_filter.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Distance Measurement`](distance_measurement.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Slack Notification`](slack_notification.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Cosine Similarity`](cosine_similarity.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Identify Outliers`](identify_outliers.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Property Definition`](property_definition.md), [`Image Threshold`](image_threshold.md), [`Dimension Collapse`](dimension_collapse.md), [`Corner Visualization`](corner_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Offset`](detection_offset.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Expression`](expression.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CogVLM` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the CogVLM model.

    - output
    
        - `parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `root_parent_id` ([`parent_id`](../kinds/parent_id.md)): Identifier of parent for step output.
        - `image` ([`image_metadata`](../kinds/image_metadata.md)): Dictionary with image metadata required by supervision.
        - `structured_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `raw_output` ([`string`](../kinds/string.md)): String value.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `CogVLM` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/cog_vlm@v1",
	    "images": "$inputs.image",
	    "prompt": "my prompt",
	    "json_output_format": {
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

