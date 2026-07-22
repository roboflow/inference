
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

    - inputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Pixel Color Count`](pixel_color_count.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Size Measurement`](size_measurement.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Motion Detection`](motion_detection.md), [`Buffer`](buffer.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detection Event Log`](detection_event_log.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Image Contours`](image_contours.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`Distance Measurement`](distance_measurement.md), [`Florence-2 Model`](florence2_model.md), [`Frame Delay`](frame_delay.md), [`GeoTag Detection`](geo_tag_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Zone`](dynamic_zone.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`Corner Visualization`](corner_visualization.md), [`PP-OCR`](ppocr.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`Image Slicer`](image_slicer.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5`](qwen3.5.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
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

