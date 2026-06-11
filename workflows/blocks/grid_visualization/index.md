
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

    - inputs: [`Florence-2 Model`](florence2_model.md), [`Google Gemma API`](google_gemma_api.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Image Contours`](image_contours.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Florence-2 Model`](florence2_model.md), [`Template Matching`](template_matching.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Distance Measurement`](distance_measurement.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Motion Detection`](motion_detection.md), [`SIFT Comparison`](sift_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`OpenRouter`](open_router.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Size Measurement`](size_measurement.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Google Gemma`](google_gemma.md), [`Detection Event Log`](detection_event_log.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Pixel Color Count`](pixel_color_count.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md)

    
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

