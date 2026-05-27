
# SAM 3



## v3

??? "Class: `SegmentAnything3BlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/segment_anything3/v3.py">inference.core.workflows.core_steps.models.foundation.segment_anything3.v3.SegmentAnything3BlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run Segment Anything 3 (SAM3), a zero-shot instance segmentation model, on an image.

You can use text prompts for open-vocabulary segmentation - just specify class names and SAM3 will
segment those objects in the image.

This block supports two output formats:
- **rle** (default): Returns masks in RLE (Run-Length Encoding) format, which is more memory-efficient
- **polygons**: Returns polygon coordinates for each mask

RLE format is recommended for high-resolution images or workflows with many detections.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sam3@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | model version. You only need to change this for fine tuned sam3 models.. | ✅ |
| `class_names` | `Optional[List[str], str]` | List of classes to recognise. | ✅ |
| `class_mapping` | `Dict[str, str]` | Maps class names in predictions to different output names. Applied after inference, e.g. {'cat': 'gato'} renames 'cat' predictions to 'gato'.. | ✅ |
| `confidence` | `float` | Minimum confidence threshold for predicted masks. | ✅ |
| `per_class_confidence` | `List[float]` | List of confidence thresholds per class (must match class_names length). | ✅ |
| `apply_nms` | `bool` | Whether to apply Non-Maximum Suppression across prompts. | ✅ |
| `nms_iou_threshold` | `float` | IoU threshold for cross-prompt NMS. Must be in [0.0, 1.0]. | ✅ |
| `output_format` | `str` | 'rle' returns efficient RLE encoding (recommended), 'polygons' returns polygon coordinates. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM 3` in version `v3`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Qwen3-VL`](qwen3_vl.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`JSON Parser`](json_parser.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Gaze Detection`](gaze_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Mask Edge Snap`](mask_edge_snap.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detections Transformation`](detections_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Detection Event Log`](detection_event_log.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Offset`](detection_offset.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM 3` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): model version. You only need to change this for fine tuned sam3 models..
        - `class_names` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): List of classes to recognise.
        - `class_mapping` (*[`dictionary`](../kinds/dictionary.md)*): Maps class names in predictions to different output names. Applied after inference, e.g. {'cat': 'gato'} renames 'cat' predictions to 'gato'..
        - `confidence` (*[`float`](../kinds/float.md)*): Minimum confidence threshold for predicted masks.
        - `per_class_confidence` (*[`list_of_values`](../kinds/list_of_values.md)*): List of confidence thresholds per class (must match class_names length).
        - `apply_nms` (*[`boolean`](../kinds/boolean.md)*): Whether to apply Non-Maximum Suppression across prompts.
        - `nms_iou_threshold` (*[`float`](../kinds/float.md)*): IoU threshold for cross-prompt NMS. Must be in [0.0, 1.0].

    - output
    
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `SAM 3` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sam3@v3",
	    "images": "$inputs.image",
	    "model_id": "sam3/sam3_final",
	    "class_names": [
	        "car",
	        "person"
	    ],
	    "class_mapping": {
	        "cat": "gato",
	        "dog": "perro"
	    },
	    "confidence": 0.3,
	    "per_class_confidence": [
	        0.3,
	        0.5,
	        0.7
	    ],
	    "apply_nms": "<block_does_not_provide_example>",
	    "nms_iou_threshold": 0.5,
	    "output_format": "rle"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `SegmentAnything3BlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/segment_anything3/v2.py">inference.core.workflows.core_steps.models.foundation.segment_anything3.v2.SegmentAnything3BlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run Segment Anything 3, a zero-shot instance segmentation model, on an image.

You can pass in boxes/predictions from other models as prompts, or use a text prompt for open-vocabulary segmentation.
If you pass in box detections from another model, the class names of the boxes will be forwarded to the predicted masks.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sam3@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | model version.  You only need to change this for fine tuned sam3 models.. | ✅ |
| `class_names` | `Optional[List[str], str]` | List of classes to recognise. | ✅ |
| `confidence` | `float` | Minimum confidence threshold for predicted masks. | ✅ |
| `per_class_confidence` | `List[float]` | List of confidence thresholds per class (must match class_names length). | ✅ |
| `apply_nms` | `bool` | Whether to apply Non-Maximum Suppression across prompts. | ✅ |
| `nms_iou_threshold` | `float` | IoU threshold for cross-prompt NMS. Must be in [0.0, 1.0]. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM 3` in version `v2`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`JSON Parser`](json_parser.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Gaze Detection`](gaze_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Mask Edge Snap`](mask_edge_snap.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detections Transformation`](detections_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Detection Event Log`](detection_event_log.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Offset`](detection_offset.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM 3` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): model version.  You only need to change this for fine tuned sam3 models..
        - `class_names` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): List of classes to recognise.
        - `confidence` (*[`float`](../kinds/float.md)*): Minimum confidence threshold for predicted masks.
        - `per_class_confidence` (*[`list_of_values`](../kinds/list_of_values.md)*): List of confidence thresholds per class (must match class_names length).
        - `apply_nms` (*[`boolean`](../kinds/boolean.md)*): Whether to apply Non-Maximum Suppression across prompts.
        - `nms_iou_threshold` (*[`float`](../kinds/float.md)*): IoU threshold for cross-prompt NMS. Must be in [0.0, 1.0].

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `SAM 3` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sam3@v2",
	    "images": "$inputs.image",
	    "model_id": "sam3/sam3_final",
	    "class_names": [
	        "car",
	        "person"
	    ],
	    "confidence": 0.3,
	    "per_class_confidence": [
	        0.3,
	        0.5,
	        0.7
	    ],
	    "apply_nms": "<block_does_not_provide_example>",
	    "nms_iou_threshold": 0.5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `SegmentAnything3BlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/segment_anything3/v1.py">inference.core.workflows.core_steps.models.foundation.segment_anything3.v1.SegmentAnything3BlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run Segment Anything 3, a zero-shot instance segmentation model, on an image.

You can pass in boxes/predictions from other models as prompts, or use a text prompt for open-vocabulary segmentation.
If you pass in box detections from another model, the class names of the boxes will be forwarded to the predicted masks.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sam3@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | model version.  You only need to change this for fine tuned sam3 models.. | ✅ |
| `class_names` | `Optional[List[str], str]` | List of classes to recognise. | ✅ |
| `threshold` | `float` | Threshold for predicted mask scores. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM 3` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Gaze Detection`](gaze_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Buffer`](buffer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Mask Edge Snap`](mask_edge_snap.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detections Transformation`](detections_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Detection Event Log`](detection_event_log.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Offset`](detection_offset.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM 3` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): model version.  You only need to change this for fine tuned sam3 models..
        - `class_names` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): List of classes to recognise.
        - `threshold` (*[`float`](../kinds/float.md)*): Threshold for predicted mask scores.

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `SAM 3` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sam3@v1",
	    "images": "$inputs.image",
	    "model_id": "sam3/sam3_final",
	    "class_names": [
	        "car",
	        "person"
	    ],
	    "threshold": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

