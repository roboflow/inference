
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Stability AI Outpainting` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Dominant Color`](dominant_color.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Visualization`](mask_visualization.md), [`SIFT`](sift.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Buffer`](buffer.md), [`VLM As Detector`](vlm_as_detector.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Background Color Visualization`](background_color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
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

