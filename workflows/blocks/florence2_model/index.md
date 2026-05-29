
# Florence-2 Model



## v2

??? "Class: `Florence2BlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/florence2/v2.py">inference.core.workflows.core_steps.models.foundation.florence2.v2.Florence2BlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



**Dedicated inference server required (GPU recommended) - you may want to use dedicated deployment**

This Workflow block introduces **Florence 2**, a Visual Language Model (VLM) capable of performing a 
wide range of tasks, including:

* Object Detection

* Instance Segmentation

* Image Captioning

* Optical Character Recognition (OCR)

* and more...


Below is a comprehensive list of tasks supported by the model, along with descriptions on 
how to utilize their outputs within the Workflows ecosystem:

**Task Descriptions:**

* **Custom Prompt** (`custom`) - Use free-form prompt to generate a response. Useful with finetuned models.

* **Text Recognition (OCR)** (`ocr`) - Model recognizes text in the image

* **Text Detection & Recognition (OCR)** (`ocr-with-text-detection`) - Model detects text regions in the image, and then performs OCR on each detected region

* **Captioning (short)** (`caption`) - Model provides a short description of the image

* **Captioning** (`detailed-caption`) - Model provides a long description of the image

* **Captioning (long)** (`more-detailed-caption`) - Model provides a very long description of the image

* **Unprompted Object Detection** (`object-detection`) - Model detects and returns the bounding boxes for prominent objects in the image

* **Object Detection** (`open-vocabulary-object-detection`) - Model detects and returns the bounding boxes for the provided classes

* **Detection & Captioning** (`object-detection-and-caption`) - Model detects prominent objects and captions them

* **Prompted Object Detection** (`phrase-grounded-object-detection`) - Based on the textual prompt, model detects objects matching the descriptions

* **Prompted Instance Segmentation** (`phrase-grounded-instance-segmentation`) - Based on the textual prompt, model segments objects matching the descriptions

* **Segment Bounding Box** (`detection-grounded-instance-segmentation`) - Model segments the object in the provided bounding box into a polygon

* **Classification of Bounding Box** (`detection-grounded-classification`) - Model classifies the object inside the provided bounding box

* **Captioning of Bounding Box** (`detection-grounded-caption`) - Model captions the object in the provided bounding box

* **Text Recognition (OCR) for Bounding Box** (`detection-grounded-ocr`) - Model performs OCR on the text inside the provided bounding box

* **Regions of Interest proposal** (`region-proposal`) - Model proposes Regions of Interest (Bounding Boxes) in the image


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/florence_2@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Florence-2 model. | ✅ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `grounding_detection` | `Optional[List[float], List[int]]` | Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`.. | ✅ |
| `grounding_selection_mode` | `str` | . | ❌ |
| `model_id` | `str` | Model to be used. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Florence-2 Model` in version `v2`.

    - inputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Velocity`](velocity.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemini`](google_gemini.md), [`Detections Filter`](detections_filter.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Seg Preview`](seg_preview.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Offset`](detection_offset.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Overlap Filter`](overlap_filter.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Size Measurement`](size_measurement.md), [`Moondream2`](moondream2.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Gaze Detection`](gaze_detection.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Distance Measurement`](distance_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Local File Sink`](local_file_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Motion Detection`](motion_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`Cache Get`](cache_get.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`Trace Visualization`](trace_visualization.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`CogVLM`](cog_vlm.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Florence-2 Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Florence-2 model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `grounding_detection` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`list_of_values`](../kinds/list_of_values.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Model to be used.

    - output
    
        - `raw_output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Florence-2 Model` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/florence_2@v2",
	    "images": "$inputs.image",
	    "task_type": "<block_does_not_provide_example>",
	    "prompt": "my prompt",
	    "classes": [
	        "class-a",
	        "class-b"
	    ],
	    "grounding_detection": "$steps.detection.predictions",
	    "grounding_selection_mode": "first",
	    "model_id": "florence-2-base"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `Florence2BlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/florence2/v1.py">inference.core.workflows.core_steps.models.foundation.florence2.v1.Florence2BlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



**Dedicated inference server required (GPU recommended) - you may want to use dedicated deployment**

This Workflow block introduces **Florence 2**, a Visual Language Model (VLM) capable of performing a 
wide range of tasks, including:

* Object Detection

* Instance Segmentation

* Image Captioning

* Optical Character Recognition (OCR)

* and more...


Below is a comprehensive list of tasks supported by the model, along with descriptions on 
how to utilize their outputs within the Workflows ecosystem:

**Task Descriptions:**

* **Custom Prompt** (`custom`) - Use free-form prompt to generate a response. Useful with finetuned models.

* **Text Recognition (OCR)** (`ocr`) - Model recognizes text in the image

* **Text Detection & Recognition (OCR)** (`ocr-with-text-detection`) - Model detects text regions in the image, and then performs OCR on each detected region

* **Captioning (short)** (`caption`) - Model provides a short description of the image

* **Captioning** (`detailed-caption`) - Model provides a long description of the image

* **Captioning (long)** (`more-detailed-caption`) - Model provides a very long description of the image

* **Unprompted Object Detection** (`object-detection`) - Model detects and returns the bounding boxes for prominent objects in the image

* **Object Detection** (`open-vocabulary-object-detection`) - Model detects and returns the bounding boxes for the provided classes

* **Detection & Captioning** (`object-detection-and-caption`) - Model detects prominent objects and captions them

* **Prompted Object Detection** (`phrase-grounded-object-detection`) - Based on the textual prompt, model detects objects matching the descriptions

* **Prompted Instance Segmentation** (`phrase-grounded-instance-segmentation`) - Based on the textual prompt, model segments objects matching the descriptions

* **Segment Bounding Box** (`detection-grounded-instance-segmentation`) - Model segments the object in the provided bounding box into a polygon

* **Classification of Bounding Box** (`detection-grounded-classification`) - Model classifies the object inside the provided bounding box

* **Captioning of Bounding Box** (`detection-grounded-caption`) - Model captions the object in the provided bounding box

* **Text Recognition (OCR) for Bounding Box** (`detection-grounded-ocr`) - Model performs OCR on the text inside the provided bounding box

* **Regions of Interest proposal** (`region-proposal`) - Model proposes Regions of Interest (Bounding Boxes) in the image


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/florence_2@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `task_type` | `str` | Task type to be performed by model. Value determines required parameters and output response.. | ❌ |
| `prompt` | `str` | Text prompt to the Florence-2 model. | ✅ |
| `classes` | `List[str]` | List of classes to be used. | ✅ |
| `grounding_detection` | `Optional[List[float], List[int]]` | Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`.. | ✅ |
| `grounding_selection_mode` | `str` | . | ❌ |
| `model_version` | `str` | Model to be used. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Florence-2 Model` in version `v1`.

    - inputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Velocity`](velocity.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemini`](google_gemini.md), [`Detections Filter`](detections_filter.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Seg Preview`](seg_preview.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Motion Detection`](motion_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Offset`](detection_offset.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Overlap Filter`](overlap_filter.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter`](line_counter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Relative Static Crop`](relative_static_crop.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Size Measurement`](size_measurement.md), [`Moondream2`](moondream2.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Camera Focus`](camera_focus.md), [`CogVLM`](cog_vlm.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Morphological Transformation`](morphological_transformation.md), [`Gaze Detection`](gaze_detection.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Transformation`](detections_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Triangle Visualization`](triangle_visualization.md), [`OCR Model`](ocr_model.md), [`Crop Visualization`](crop_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Template Matching`](template_matching.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)
    - outputs: [`Distance Measurement`](distance_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Morphological Transformation`](morphological_transformation.md), [`LMM`](lmm.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Buffer`](buffer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Grid Visualization`](grid_visualization.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Local File Sink`](local_file_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Motion Detection`](motion_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Color Visualization`](color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Object Detection Model`](object_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Label Visualization`](label_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Image Threshold`](image_threshold.md), [`Cache Get`](cache_get.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`Trace Visualization`](trace_visualization.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`CogVLM`](cog_vlm.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Perspective Correction`](perspective_correction.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Stability AI Inpainting`](stability_ai_inpainting.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Florence-2 Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Florence-2 model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `grounding_detection` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`list_of_values`](../kinds/list_of_values.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`..
        - `model_version` (*[`string`](../kinds/string.md)*): Model to be used.

    - output
    
        - `raw_output` (*Union[[`string`](../kinds/string.md), [`language_model_output`](../kinds/language_model_output.md)]*): String value if `string` or LLM / VLM output if `language_model_output`.
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.
        - `classes` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Florence-2 Model` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/florence_2@v1",
	    "images": "$inputs.image",
	    "task_type": "<block_does_not_provide_example>",
	    "prompt": "my prompt",
	    "classes": [
	        "class-a",
	        "class-b"
	    ],
	    "grounding_detection": "$steps.detection.predictions",
	    "grounding_selection_mode": "first",
	    "model_version": "florence-2-base"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

