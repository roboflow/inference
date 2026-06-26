
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
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Property Definition`](property_definition.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Continue If`](continue_if.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Overlap Analysis`](overlap_analysis.md), [`Trace Visualization`](trace_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Crop`](dynamic_crop.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Switch Case`](switch_case.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Expression`](expression.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dominant Color`](dominant_color.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Detections Filter`](detections_filter.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
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

