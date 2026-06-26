
# Stability AI Outpainting



??? "Class: `StabilityAIOutpaintingBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/stability_ai/outpainting/v1.py">inference.core.workflows.core_steps.models.foundation.stability_ai.outpainting.v1.StabilityAIOutpaintingBlockV1</a>
    



The block wraps 
[Stability AI outpainting API](https://platform.stability.ai/docs/api-reference#tag/Edit/paths/~1v2beta~1stable-image~1edit~1outpaint/post) and 
let users use object detection results to change the content of images in a creative way.

The block sends crop of the image to the API together with directions where to outpaint.
As a result, the API returns the image with outpainted regions.
At least one of `left`, `right`, `up`, `down` must be provided, otherwise original image is returned.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stability_ai_outpainting@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `creativity` | `float` | Creativity parameter for outpainting. Higher values result in more creative outpainting.. | ✅ |
| `left` | `int` | Number of pixels to outpaint on the left side of the image. Max value is 2000.. | ✅ |
| `right` | `int` | Number of pixels to outpaint on the right side of the image. Max value is 2000.. | ✅ |
| `up` | `int` | Number of pixels to outpaint on the top side of the image. Max value is 2000.. | ✅ |
| `down` | `int` | Number of pixels to outpaint on the bottom side of the image. Max value is 2000.. | ✅ |
| `prompt` | `str` | Optional prompt to apply when outpainting the image (what you wish to see). If not provided, the image will be outpainted without any prompt.. | ✅ |
| `preset` | `StabilityAIPresets` | Optional preset to apply when outpainting the image (what you wish to see). If not provided, the image will be outpainted without any preset. Avaliable presets: 3d-model, analog-film, anime, cinematic, comic-book, digital-art, enhance, fantasy-art, isometric, line-art, low-poly, modeling-compound, neon-punk, origami, photographic, pixel-art, tile-texture. | ❌ |
| `seed` | `int` | A specific value that is used to guide the 'randomness' of the generation. If not provided, a random seed is used. Must be a number between 0 and 4294967294. | ✅ |
| `api_key` | `str` | Your Stability AI API key.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stability AI Outpainting` in version `v1`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Blur Visualization`](blur_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`Detection Event Log`](detection_event_log.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen3.5`](qwen3.5.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Stability AI Outpainting` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to outpaint..
        - `creativity` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Creativity parameter for outpainting. Higher values result in more creative outpainting..
        - `left` (*[`integer`](../kinds/integer.md)*): Number of pixels to outpaint on the left side of the image. Max value is 2000..
        - `right` (*[`integer`](../kinds/integer.md)*): Number of pixels to outpaint on the right side of the image. Max value is 2000..
        - `up` (*[`integer`](../kinds/integer.md)*): Number of pixels to outpaint on the top side of the image. Max value is 2000..
        - `down` (*[`integer`](../kinds/integer.md)*): Number of pixels to outpaint on the bottom side of the image. Max value is 2000..
        - `prompt` (*[`string`](../kinds/string.md)*): Optional prompt to apply when outpainting the image (what you wish to see). If not provided, the image will be outpainted without any prompt..
        - `seed` (*[`integer`](../kinds/integer.md)*): A specific value that is used to guide the 'randomness' of the generation. If not provided, a random seed is used. Must be a number between 0 and 4294967294.
        - `api_key` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Your Stability AI API key..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Stability AI Outpainting` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stability_ai_outpainting@v1",
	    "image": "$inputs.image",
	    "creativity": 0.5,
	    "left": 200,
	    "right": 200,
	    "up": 200,
	    "down": 200,
	    "prompt": "my prompt",
	    "preset": "3d-model",
	    "seed": 200,
	    "api_key": "xxx-xxx"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

