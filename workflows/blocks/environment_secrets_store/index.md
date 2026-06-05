
# Environment Secrets Store



??? "Class: `EnvironmentSecretsStoreBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/secrets_providers/environment_secrets_store/v1.py">inference.core.workflows.core_steps.secrets_providers.environment_secrets_store.v1.EnvironmentSecretsStoreBlockV1</a>
    



The **Environment Secrets Store** block is a secure and flexible solution for fetching secrets stored as 
**environmental variables**.

## How This Block Works

It is designed to enable Workflows to access sensitive information, 
such as API keys or service credentials, without embedding them directly into the Workflow definitions. 

This block simplifies the integration of external services while prioritizing security and adaptability. You can
use secrets fetched from environment (which can be set by system administrator to be available in self-hosted
`inference` server) to pass as inputs to other steps.

!!! Tip "Credentials security"

    It is strongly advised to use secrets providers (available when running self-hosted `inference` server)
    or workflows parameters to pass credentials. **Do not hardcode secrets in Workflows definitions.**
    
!!! Important "Blocks limitations"

    This block can only run on self-hosted `inference` server, we Roboflow does not allow exporting env
    variables from Hosted Platform due to security concerns. 

#### 🛠️ Block configuration

Block has configuration parameter `variables_storing_secrets` that must be filled with list of
environmental variables which will be exposed as block outputs. Thanks to that, you can
use them as inputs for other blocks. Please note that names of outputs will be lowercased. For example,
the following settings:
```
variables_storing_secrets=["MY_SECRET_A", "MY_SECRET_B"]
```
will generate the following outputs:

* `my_secret_a`

* `my_secret_b`


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/environment_secrets_store@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `variables_storing_secrets` | `List[str]` | List with names of environment variables to fetch. Each will create separate block output.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Environment Secrets Store` in version `v1`.

    - inputs: None
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Time in Zone`](timein_zone.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Cosine Similarity`](cosine_similarity.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`S3 Sink`](s3_sink.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Bounding Rectangle`](bounding_rectangle.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Slack Notification`](slack_notification.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`Image Stack`](image_stack.md), [`Delta Filter`](delta_filter.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Label Visualization`](label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`Overlap Filter`](overlap_filter.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Buffer`](buffer.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Dominant Color`](dominant_color.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Qwen3-VL`](qwen3_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Continue If`](continue_if.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`SmolVLM2`](smol_vlm2.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Expression`](expression.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Barcode Detection`](barcode_detection.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Google Gemma`](google_gemma.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`QR Code Detection`](qr_code_detection.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Equalization`](contrast_equalization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Image Contours`](image_contours.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Qwen3.5`](qwen3.5.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Property Definition`](property_definition.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Environment Secrets Store` in version `v1`  has.

???+ tip "Bindings"

    - input
    


    - output
    
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `Environment Secrets Store` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/environment_secrets_store@v1",
	    "variables_storing_secrets": [
	        "MY_API_KEY",
	        "OTHER_API_KEY"
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

