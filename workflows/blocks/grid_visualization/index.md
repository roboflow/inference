
# Grid Visualization



??? "Class: `GridVisualizationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/grid/v1.py">inference.core.workflows.core_steps.visualizations.grid.v1.GridVisualizationBlockV1</a>
    



Arrange multiple images in a grid layout, automatically organizing a list of images into a square grid pattern with automatic resizing and cell-based positioning for side-by-side comparison, thumbnail displays, or batch visualization.

## How This Block Works

This block takes a list of images and arranges them into a grid layout within a single output image. The block:

1. Takes a list of images and output dimensions (width and height) as input
2. Calculates the grid size based on the number of images (creates a square grid with dimensions equal to the square root of the image count, rounded up)
3. Divides the output canvas into equal-sized cells based on the grid dimensions
4. Resizes each input image to fit within its assigned cell while maintaining aspect ratio (images are scaled to fit the cell dimensions without distortion)
5. Places images in the grid starting from the top-left corner, filling left-to-right and top-to-bottom (row-major order)
6. Centers each resized image within its cell, creating evenly spaced grid layout
7. Returns a single output image containing all input images arranged in the grid

The block automatically organizes multiple images into a grid for easy comparison or batch viewing. Each image is resized to fit its grid cell while preserving aspect ratio, and images are centered within their cells. The grid dimensions are automatically calculated to create a roughly square grid (e.g., 4 images = 2x2, 9 images = 3x3, 10 images = 4x4). This creates a compact, organized layout ideal for comparing multiple images, displaying thumbnails, or creating batch visualization outputs. The block uses caching to optimize performance when the same images are reused.

## Common Use Cases

- **Batch Image Comparison**: Arrange multiple images side-by-side in a grid for easy comparison, allowing you to visualize results from different models, time periods, or processing steps simultaneously
- **Thumbnail Gallery Creation**: Create thumbnail grids from collections of images for gallery displays, image browsers, or preview interfaces where multiple images need to be shown in a compact layout
- **Multi-Image Workflow Results**: Display results from multi-image workflows (like batch processing, image slicer outputs, or buffer collections) in an organized grid format for overview visualization
- **Before/After Comparisons**: Arrange before and after images, original and processed versions, or multiple workflow outputs in a grid for comparison and validation workflows
- **Time-Series Visualization**: Display images from different time points, frames, or snapshots in a grid to visualize temporal changes, sequences, or progression over time
- **Quality Control and Review**: Create grid layouts for quality control workflows, batch review, or inspection processes where multiple images need to be viewed together for evaluation or validation

## Connecting to Other Blocks

The grid output image from this block can be connected to:

- **Image processing blocks** (e.g., Buffer, Image Slicer, Dynamic Crop) to receive lists of images that are arranged into grid layouts
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save grid images for documentation, reporting, or batch review purposes
- **Webhook blocks** to send grid visualizations to external systems, APIs, or web applications for display in dashboards, galleries, or batch viewing interfaces
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send grid images as visual evidence in alerts or reports containing multiple images
- **Video output blocks** to create video streams or recordings with grid layouts for live multi-image monitoring or batch visualization workflows
- **Other visualization blocks** that can accept single images, allowing grid outputs to be further processed or combined with additional annotations


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/grid_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `width` | `int` | Width of the output grid image in pixels. Controls the total width of the canvas where the image grid will be arranged. The width is divided into equal-sized cells based on the grid dimensions. Typical values range from 1280 to 3840 pixels depending on desired output size and number of images.. | ✅ |
| `height` | `int` | Height of the output grid image in pixels. Controls the total height of the canvas where the image grid will be arranged. The height is divided into equal-sized cells based on the grid dimensions. Typical values range from 720 to 2160 pixels depending on desired output size and number of images.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Grid Visualization` in version `v1`.

    - inputs: [`Camera Focus`](camera_focus.md), [`Size Measurement`](size_measurement.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`Distance Measurement`](distance_measurement.md), [`Line Counter`](line_counter.md), [`Image Contours`](image_contours.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Line Counter`](line_counter.md), [`Buffer`](buffer.md), [`Detection Event Log`](detection_event_log.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Camera Focus`](camera_focus.md), [`Background Color Visualization`](background_color_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Barcode Detection`](barcode_detection.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Preprocessing`](image_preprocessing.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dominant Color`](dominant_color.md), [`Gaze Detection`](gaze_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Grid Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`list_of_values`](../kinds/list_of_values.md)*): List of images to arrange in a grid layout. Can be a list of image outputs from blocks like Buffer, Image Slicer, Dynamic Crop, or other blocks that output multiple images. Images will be automatically arranged in a square grid (calculated from the number of images) and resized to fit their grid cells while maintaining aspect ratio..
        - `width` (*[`integer`](../kinds/integer.md)*): Width of the output grid image in pixels. Controls the total width of the canvas where the image grid will be arranged. The width is divided into equal-sized cells based on the grid dimensions. Typical values range from 1280 to 3840 pixels depending on desired output size and number of images..
        - `height` (*[`integer`](../kinds/integer.md)*): Height of the output grid image in pixels. Controls the total height of the canvas where the image grid will be arranged. The height is divided into equal-sized cells based on the grid dimensions. Typical values range from 720 to 2160 pixels depending on desired output size and number of images..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Grid Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/grid_visualization@v1",
	    "images": "$steps.buffer.output",
	    "width": 2560,
	    "height": 1440
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

