
# Instance Segmentation Model



## v4

??? "Class: `RoboflowInstanceSegmentationModelBlockV4`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v4.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v4.RoboflowInstanceSegmentationModelBlockV4</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).

This version of block introduces breaking change in behaviour of mask construction - it uses 
`rle` format instead `polygon` making it possible to retrieve 
shapes of any kind from remote server.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v4`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How confidence thresholds are determined.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v4`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Size Measurement`](size_measurement.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Distance Measurement`](distance_measurement.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Overlap Filter`](overlap_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Dot Visualization`](dot_visualization.md), [`Detections Merge`](detections_merge.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Moondream2`](moondream2.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter`](line_counter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Crop Visualization`](crop_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Perspective Correction`](perspective_correction.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v4`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How confidence thresholds are determined..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v4`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
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




## v3

??? "Class: `RoboflowInstanceSegmentationModelBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v3.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v3.RoboflowInstanceSegmentationModelBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this
block. To learn more about setting your Roboflow API key, [refer to the Inference
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence_mode` | `str` | How confidence thresholds are determined.. | ✅ |
| `custom_confidence` | `float` | Custom confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v3`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Size Measurement`](size_measurement.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Distance Measurement`](distance_measurement.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Overlap Filter`](overlap_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Moondream2`](moondream2.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Crop Visualization`](crop_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Perspective Correction`](perspective_correction.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence_mode` (*[`string`](../kinds/string.md)*): How confidence thresholds are determined..
        - `custom_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Custom confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v3",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence_mode": "<block_does_not_provide_example>",
	    "custom_confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `RoboflowInstanceSegmentationModelBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v2.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v2.RoboflowInstanceSegmentationModelBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v2`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`Local File Sink`](local_file_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Distance Measurement`](distance_measurement.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Overlap Filter`](overlap_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Moondream2`](moondream2.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Crop Visualization`](crop_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Perspective Correction`](perspective_correction.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `model_id` ([`roboflow_model_id`](../kinds/roboflow_model_id.md)): Roboflow model id.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `RoboflowInstanceSegmentationModelBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/roboflow/instance_segmentation/v1.py">inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v1.RoboflowInstanceSegmentationModelBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Run inference on an instance segmentation model hosted on or uploaded to Roboflow.

You can query any model that is private to your account, or any public model available 
on [Roboflow Universe](https://universe.roboflow.com).

You will need to set your Roboflow API key in your Inference environment to use this 
block. To learn more about setting your Roboflow API key, [refer to the Inference 
documentation](https://inference.roboflow.com/quickstart/configure_api_key/).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_instance_segmentation_model@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Roboflow model identifier.. | ✅ |
| `confidence` | `float` | Confidence threshold for predictions.. | ✅ |
| `class_filter` | `List[str]` | List of accepted classes. Classes must exist in the model's training set.. | ✅ |
| `iou_threshold` | `float` | Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/).. | ✅ |
| `max_detections` | `int` | Maximum number of detections to return.. | ✅ |
| `class_agnostic_nms` | `bool` | Boolean flag to specify if NMS is to be used in class-agnostic mode.. | ✅ |
| `max_candidates` | `int` | Maximum number of candidates as NMS input to be taken into account.. | ✅ |
| `mask_decode_mode` | `str` | Parameter of mask decoding in prediction post-processing.. | ✅ |
| `tradeoff_factor` | `float` | Post-processing parameter to dictate tradeoff between fast and accurate.. | ✅ |
| `disable_active_learning` | `bool` | Boolean flag to disable project-level active learning for this block.. | ✅ |
| `active_learning_target_dataset` | `str` | Target dataset for active learning, if enabled.. | ✅ |
| `enforce_dense_masks_in_inference_models` | `bool` | Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Instance Segmentation Model` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Identify Outliers`](identify_outliers.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`Local File Sink`](local_file_sink.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Changes`](identify_changes.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Distance Measurement`](distance_measurement.md), [`Velocity`](velocity.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Merge`](detections_merge.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Local File Sink`](local_file_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Focus`](camera_focus.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Offset`](detection_offset.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Overlap Filter`](overlap_filter.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`Cache Get`](cache_get.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Line Counter`](line_counter.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Trace Visualization`](trace_visualization.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Instance Segmentation Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model identifier..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for predictions..
        - `class_filter` (*[`list_of_values`](../kinds/list_of_values.md)*): List of accepted classes. Classes must exist in the model's training set..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap threshold between boxes to combine them into a single detection, used in NMS. [Learn more](https://blog.roboflow.com/how-to-code-non-maximum-suppression-nms-in-plain-numpy/)..
        - `max_detections` (*[`integer`](../kinds/integer.md)*): Maximum number of detections to return..
        - `class_agnostic_nms` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to specify if NMS is to be used in class-agnostic mode..
        - `max_candidates` (*[`integer`](../kinds/integer.md)*): Maximum number of candidates as NMS input to be taken into account..
        - `mask_decode_mode` (*[`string`](../kinds/string.md)*): Parameter of mask decoding in prediction post-processing..
        - `tradeoff_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Post-processing parameter to dictate tradeoff between fast and accurate..
        - `disable_active_learning` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable project-level active learning for this block..
        - `active_learning_target_dataset` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Target dataset for active learning, if enabled..
        - `enforce_dense_masks_in_inference_models` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to enforce dense masks when inference models backend is in use (irrelevant in other cases). Dense masks are faster to process, but require more memory. Users can't tweak this flag when running on Roboflow serverless platform..

    - output
    
        - `inference_id` ([`string`](../kinds/string.md)): String value.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Instance Segmentation Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_instance_segmentation_model@v1",
	    "images": "$inputs.image",
	    "model_id": "my_project/3",
	    "confidence": 0.3,
	    "class_filter": [
	        "a",
	        "b",
	        "c"
	    ],
	    "iou_threshold": 0.4,
	    "max_detections": 300,
	    "class_agnostic_nms": true,
	    "max_candidates": 3000,
	    "mask_decode_mode": "accurate",
	    "tradeoff_factor": 0.3,
	    "disable_active_learning": true,
	    "active_learning_target_dataset": "my_project",
	    "enforce_dense_masks_in_inference_models": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

