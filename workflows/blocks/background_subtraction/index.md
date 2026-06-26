
# Background Subtraction



??? "Class: `BackgroundSubtractionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/background_subtraction/v1.py">inference.core.workflows.core_steps.classical_cv.background_subtraction.v1.BackgroundSubtractionBlockV1</a>
    



Create motion masks from video streams using OpenCV's background subtraction algorithm.

## How This Block Works

This block uses background subtraction (specifically the MOG2 algorithm) to identify pixels that differ from a learned background model and outputs a mask image highlighting motion areas. The block maintains state across frames to build and update the background model:

1. **Initializes background model** - on the first frame, creates a background subtractor using the specified history and threshold parameters
2. **Processes each frame** - applies background subtraction to identify pixels that differ from the learned background model
3. **Creates motion mask** - generates a foreground mask where white pixels represent motion areas and black pixels represent the background
4. **Converts to image format** - converts the single-channel mask to a 3-channel image format required by workflows
5. **Returns mask image** - outputs the motion mask as an image that can be visualized or processed further

The output mask image shows motion areas as white pixels against a black background, making it easy to visualize where motion occurred in the frame. This mask can be used for further analysis, visualization, or as input to other processing steps.

## Common Use Cases

- **Motion Visualization**: Create visual motion masks to see where movement occurs in video streams for monitoring, analysis, or debugging purposes
- **Preprocessing for Motion Models**: Generate motion masks as input data for training or inference with motion-based models that require mask data
- **Motion Area Extraction**: Extract regions of motion from video frames for further processing, analysis, or feature extraction
- **Video Analysis**: Analyze motion patterns by processing mask images to identify movement trends, activity levels, or motion characteristics
- **Background Removal**: Use motion masks to separate foreground (moving) objects from static background for segmentation or isolation tasks
- **Motion-based Filtering**: Use motion masks to filter or focus processing on areas where motion occurs, ignoring static background regions

## Connecting to Other Blocks

The motion mask image from this block can be connected to:

- **Visualization blocks** to display the motion mask overlayed on original images or as standalone visualizations
- **Object detection blocks** to run detection models only on motion regions identified by the mask
- **Image processing blocks** to apply additional transformations, filters, or analysis to motion mask images
- **Data storage blocks** (e.g., Local File Sink, Roboflow Dataset Upload) to save motion masks for training data, analysis, or documentation
- **Conditional logic blocks** to route workflow execution based on the presence or absence of motion in mask images
- **Model training blocks** to use motion masks as training data for motion-based models or segmentation tasks


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/background_subtraction@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `threshold` | `int` | Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16.. | ✅ |
| `history` | `int` | Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Background Subtraction` in version `v1`.

    - inputs: [`Heatmap Visualization`](heatmap_visualization.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter`](line_counter.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`Blur Visualization`](blur_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Depth Estimation`](depth_estimation.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Stitch Images`](stitch_images.md), [`Halo Visualization`](halo_visualization.md), [`Image Blur`](image_blur.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Contrast Equalization`](contrast_equalization.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Line Counter`](line_counter.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Image Slicer`](image_slicer.md), [`Text Display`](text_display.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Polygon Visualization`](polygon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Camera Focus`](camera_focus.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Template Matching`](template_matching.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OCR Model`](ocr_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Blur Visualization`](blur_visualization.md), [`Barcode Detection`](barcode_detection.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SIFT`](sift.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen3.5`](qwen3.5.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Background Subtraction` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The input image or video frame to process for background subtraction. The block processes frames sequentially to build a background model - each frame updates the background model and creates a motion mask showing areas that differ from the learned background. Can be connected from workflow inputs or previous steps..
        - `threshold` (*[`integer`](../kinds/integer.md)*): Threshold value for the squared Mahalanobis distance used by the MOG2 background subtraction algorithm. Controls sensitivity to motion - smaller values increase sensitivity (detect smaller changes) but may produce more false positives, larger values decrease sensitivity (only detect significant changes) but may miss subtle motion. Recommended range is 8-32. Default is 16..
        - `history` (*[`integer`](../kinds/integer.md)*): Number of previous frames used to build the background model. Controls how quickly the background adapts to changes - larger values (e.g., 50-100) create a more stable background model that's less sensitive to temporary changes but adapts slowly to permanent background changes. Smaller values (e.g., 10-20) allow faster adaptation but may treat moving objects as background if they stop moving. Default is 30 frames..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Background Subtraction` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/background_subtraction@v1",
	    "image": "$inputs.image",
	    "threshold": 16,
	    "history": 30
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

