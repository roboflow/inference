
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Florence-2 Model` in version `v2`.

    - inputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Transformation`](detections_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Bounding Rectangle`](bounding_rectangle.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Stitch`](detections_stitch.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Overlap Filter`](overlap_filter.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Florence-2 Model` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Florence-2 model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `grounding_detection` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`list_of_values`](../kinds/list_of_values.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`..
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Florence-2 Model` in version `v1`.

    - inputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Transformation`](detections_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Image Threshold`](image_threshold.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Bounding Rectangle`](bounding_rectangle.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Stitch`](detections_stitch.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Overlap Filter`](overlap_filter.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Florence-2 Model` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `prompt` (*[`string`](../kinds/string.md)*): Text prompt to the Florence-2 model.
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of classes to be used.
        - `grounding_detection` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`list_of_values`](../kinds/list_of_values.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection to ground Florence-2 model. May be statically provided bounding box `[left_top_x, left_top_y, right_bottom_x, right_bottom_y]` or result of object-detection model. If the latter is true, one box will be selected based on `grounding_selection_mode`..
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

