
# Polygon Visualization



## v2

??? "Class: `PolygonVisualizationBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/polygon/v2.py">inference.core.workflows.core_steps.visualizations.polygon.v2.PolygonVisualizationBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Draw polygon outlines around detected objects that follow the exact shape of object masks, providing precise boundary visualization for instance segmentation results.

## How This Block Works

This block takes an image and instance segmentation predictions (which include segmentation masks) and draws polygon outlines that precisely follow the shape of each detected object. The block:

1. Takes an image and instance segmentation predictions as input (predictions must include mask data)
2. Converts segmentation masks to polygon coordinates that trace the object boundaries
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws polygon outlines with the specified thickness using the PolygonAnnotator
5. Returns an annotated image with polygon outlines overlaid on the original image

The block extracts the exact shape of each object from its segmentation mask and draws polygon outlines that follow these precise boundaries. This provides much more accurate visualization than bounding boxes, as polygons conform to the actual object shape rather than enclosing them in rectangles. If mask data is not available, the block falls back to drawing bounding boxes. The polygon outlines can be customized with different thickness values and color palettes, allowing you to clearly distinguish between different objects or object classes.

## Common Use Cases

- **Precise Object Boundary Visualization**: Visualize the exact shape and boundaries of segmented objects for applications requiring accurate object outlines, such as medical imaging, manufacturing quality control, or precise measurement workflows
- **Instance Segmentation Model Validation**: Verify and debug instance segmentation model performance by visualizing how well polygon predictions match object boundaries, identify segmentation errors, and validate mask quality
- **Irregular Shape Analysis**: Visualize objects with irregular or non-rectangular shapes (e.g., people, animals, complex machinery parts) where bounding boxes would be inaccurate or misleading
- **Overlapping Object Visualization**: Clearly show object boundaries when multiple objects overlap, as polygons accurately represent each object's shape without the ambiguity of overlapping bounding boxes
- **Shape-Based Quality Control**: Inspect object shapes and boundaries in manufacturing, agriculture, or quality assurance workflows where precise object contours are critical for defect detection or classification
- **Scientific and Medical Imaging**: Visualize segmented regions in medical imaging, microscopy, or scientific analysis where accurate boundary representation is essential for measurement, analysis, or diagnosis

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Mask Visualization, Bounding Box Visualization) to combine polygon outlines with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with polygon outlines for documentation, reporting, or training data validation
- **Webhook blocks** to send visualized results with polygon outlines to external systems, APIs, or web applications for display in dashboards or analysis tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with polygon outlines as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with polygon outlines for live monitoring, tracking, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/polygon_visualization@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `color_palette` | `str` | Select a color palette for the visualised elements.. | ✅ |
| `palette_size` | `int` | Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes.. | ✅ |
| `custom_colors` | `List[str]` | Define a list of custom colors for bounding boxes in HEX format.. | ✅ |
| `color_axis` | `str` | Choose how bounding box colors are assigned.. | ✅ |
| `thickness` | `int` | Thickness of the polygon outline in pixels. Higher values create thicker, more visible outlines.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Polygon Visualization` in version `v2`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`Distance Measurement`](distance_measurement.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Path Deviation`](path_deviation.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md)
    - outputs: [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`SORT Tracker`](sort_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Blur Visualization`](blur_visualization.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Polygon Visualization` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Instance segmentation predictions containing mask data. The block converts masks to polygon outlines that follow the exact shape of each detected object..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the polygon outline in pixels. Higher values create thicker, more visible outlines..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Polygon Visualization` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/polygon_visualization@v2",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.instance_segmentation_model.predictions",
	    "color_palette": "DEFAULT",
	    "palette_size": 10,
	    "custom_colors": [
	        "#FF0000",
	        "#00FF00",
	        "#0000FF"
	    ],
	    "color_axis": "CLASS",
	    "thickness": 2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `PolygonVisualizationBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/visualizations/polygon/v1.py">inference.core.workflows.core_steps.visualizations.polygon.v1.PolygonVisualizationBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Draw polygon outlines around detected objects that follow the exact shape of object masks, providing precise boundary visualization for instance segmentation results.

## How This Block Works

This block takes an image and instance segmentation predictions (which include segmentation masks) and draws polygon outlines that precisely follow the shape of each detected object. The block:

1. Takes an image and instance segmentation predictions as input (predictions must include mask data)
2. Converts segmentation masks to polygon coordinates that trace the object boundaries
3. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
4. Draws polygon outlines with the specified thickness using the PolygonAnnotator
5. Returns an annotated image with polygon outlines overlaid on the original image

The block extracts the exact shape of each object from its segmentation mask and draws polygon outlines that follow these precise boundaries. This provides much more accurate visualization than bounding boxes, as polygons conform to the actual object shape rather than enclosing them in rectangles. If mask data is not available, the block falls back to drawing bounding boxes. The polygon outlines can be customized with different thickness values and color palettes, allowing you to clearly distinguish between different objects or object classes.

## Common Use Cases

- **Precise Object Boundary Visualization**: Visualize the exact shape and boundaries of segmented objects for applications requiring accurate object outlines, such as medical imaging, manufacturing quality control, or precise measurement workflows
- **Instance Segmentation Model Validation**: Verify and debug instance segmentation model performance by visualizing how well polygon predictions match object boundaries, identify segmentation errors, and validate mask quality
- **Irregular Shape Analysis**: Visualize objects with irregular or non-rectangular shapes (e.g., people, animals, complex machinery parts) where bounding boxes would be inaccurate or misleading
- **Overlapping Object Visualization**: Clearly show object boundaries when multiple objects overlap, as polygons accurately represent each object's shape without the ambiguity of overlapping bounding boxes
- **Shape-Based Quality Control**: Inspect object shapes and boundaries in manufacturing, agriculture, or quality assurance workflows where precise object contours are critical for defect detection or classification
- **Scientific and Medical Imaging**: Visualize segmented regions in medical imaging, microscopy, or scientific analysis where accurate boundary representation is essential for measurement, analysis, or diagnosis

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Mask Visualization, Bounding Box Visualization) to combine polygon outlines with additional annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with polygon outlines for documentation, reporting, or training data validation
- **Webhook blocks** to send visualized results with polygon outlines to external systems, APIs, or web applications for display in dashboards or analysis tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with polygon outlines as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with polygon outlines for live monitoring, tracking, or post-processing analysis


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/polygon_visualization@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `copy_image` | `bool` | Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.. | ✅ |
| `color_palette` | `str` | Select a color palette for the visualised elements.. | ✅ |
| `palette_size` | `int` | Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes.. | ✅ |
| `custom_colors` | `List[str]` | Define a list of custom colors for bounding boxes in HEX format.. | ✅ |
| `color_axis` | `str` | Choose how bounding box colors are assigned.. | ✅ |
| `thickness` | `int` | Thickness of the polygon outline in pixels. Higher values create thicker, more visible outlines.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Polygon Visualization` in version `v1`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`CSV Formatter`](csv_formatter.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detection Offset`](detection_offset.md), [`Distance Measurement`](distance_measurement.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Path Deviation`](path_deviation.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md)
    - outputs: [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stitch`](detections_stitch.md), [`SORT Tracker`](sort_tracker.md), [`Barcode Detection`](barcode_detection.md), [`Text Display`](text_display.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Blur Visualization`](blur_visualization.md), [`Moondream2`](moondream2.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Polygon Visualization` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image to visualize on..
        - `copy_image` (*[`boolean`](../kinds/boolean.md)*): Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations..
        - `predictions` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Instance segmentation predictions containing mask data. The block converts masks to polygon outlines that follow the exact shape of each detected object..
        - `color_palette` (*[`string`](../kinds/string.md)*): Select a color palette for the visualised elements..
        - `palette_size` (*[`integer`](../kinds/integer.md)*): Specify the number of colors in the palette. This applies when using custom or Matplotlib palettes..
        - `custom_colors` (*[`list_of_values`](../kinds/list_of_values.md)*): Define a list of custom colors for bounding boxes in HEX format..
        - `color_axis` (*[`string`](../kinds/string.md)*): Choose how bounding box colors are assigned..
        - `thickness` (*[`integer`](../kinds/integer.md)*): Thickness of the polygon outline in pixels. Higher values create thicker, more visible outlines..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.



??? tip "Example JSON definition of step `Polygon Visualization` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/polygon_visualization@v1",
	    "image": "$inputs.image",
	    "copy_image": true,
	    "predictions": "$steps.instance_segmentation_model.predictions",
	    "color_palette": "DEFAULT",
	    "palette_size": 10,
	    "custom_colors": [
	        "#FF0000",
	        "#00FF00",
	        "#0000FF"
	    ],
	    "color_axis": "CLASS",
	    "thickness": 2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

