
# Distance Measurement



??? "Class: `DistanceMeasurementBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/distance_measurement/v1.py">inference.core.workflows.core_steps.classical_cv.distance_measurement.v1.DistanceMeasurementBlockV1</a>
    



Calculate the distance between two detected objects on a 2D plane using bounding box coordinates, supporting horizontal or vertical distance measurement along a specified axis, with two calibration methods (reference object with known dimensions or pixel-to-centimeter ratio) to convert pixel distances to real-world measurements for spatial analysis, object spacing assessment, safety monitoring, and measurement workflows.

## How This Block Works

This block measures the distance between two detected objects by analyzing their bounding box positions and converting pixel distances to real-world units (centimeters). The block:

1. Receives detection predictions containing bounding boxes and class names for objects in the image
2. Identifies the two target objects using their class names (object_1_class_name and object_2_class_name):
   - Searches through all detections to find bounding boxes matching the specified class names
   - Extracts bounding box coordinates (x_min, y_min, x_max, y_max) for both objects
   - Validates that both objects are found in the detections
3. Validates object positioning for distance measurement:
   - Checks if bounding boxes overlap (if they overlap, distance is set to 0)
   - Verifies objects have a gap along the specified reference axis (horizontal or vertical)
   - Returns 0 distance if objects overlap or are positioned incorrectly for the selected axis
4. Determines the calibration method and performs calibration:

   **For Reference Object Calibration:**
   - Searches detections for a reference object with known real-world dimensions (reference_object_class_name)
   - Extracts the reference object's bounding box coordinates
   - Measures reference object dimensions in pixels (width and height)
   - Calculates pixel-to-centimeter ratios:
     - Width ratio: reference_width_pixels / reference_width (cm)
     - Height ratio: reference_height_pixels / reference_height (cm)
   - Computes average pixel ratio from width and height ratios for more accurate scaling
   - Uses the average ratio to convert all pixel measurements to centimeters

   **For Pixel-to-Centimeter Ratio Calibration:**
   - Uses the provided pixel_ratio directly (e.g., 100 pixels = 1 centimeter)
   - Applies the ratio to convert pixel distances to centimeter distances
   - Suitable when the pixel-to-real-world scale is already known or calibrated

5. Measures pixel distance between the two objects along the specified axis:
   - **For Vertical Distance**: Calculates distance along the Y-axis (vertical separation)
     - Finds the gap between bounding boxes vertically
     - Measures distance from bottom of upper object to top of lower object (or vice versa)
     - Accounts for bounding box positions to find the actual gap distance
   - **For Horizontal Distance**: Calculates distance along the X-axis (horizontal separation)
     - Finds the gap between bounding boxes horizontally
     - Measures distance from right edge of left object to left edge of right object (or vice versa)
     - Accounts for bounding box positions to find the actual gap distance
6. Converts pixel distance to centimeter distance:
   - Divides pixel distance by the pixel-to-centimeter ratio (from calibration)
   - Produces real-world distance measurement in centimeters
7. Returns both pixel distance and centimeter distance values

The block assumes a perpendicular camera view (top-down or frontal view) where perspective distortion is minimal, ensuring accurate 2D distance measurements. Distance is measured as the gap between bounding boxes along the specified axis (horizontal or vertical), not the diagonal distance between object centers. The calibration process converts pixel measurements to real-world units using either a reference object with known dimensions (more flexible, works with different scales) or a direct pixel ratio (simpler, requires pre-calibration). This enables accurate spatial measurements for monitoring, analysis, and control applications.

## Common Use Cases

- **Safety Monitoring**: Measure distances between objects to ensure safe spacing (e.g., measure distance between people for social distancing, monitor spacing between vehicles, ensure safe gaps in industrial settings), enabling safety monitoring workflows
- **Warehouse Management**: Measure spacing between items or objects in storage and logistics (e.g., measure gaps between packages, assess shelf spacing, monitor object placement), enabling warehouse management workflows
- **Quality Control**: Verify spacing and positioning of objects in manufacturing and assembly (e.g., measure gaps between components, verify spacing in assembly lines, check positioning accuracy), enabling quality control workflows
- **Traffic Analysis**: Measure distances between vehicles or objects in traffic monitoring (e.g., measure vehicle spacing, assess safe following distances, monitor traffic gaps), enabling traffic analysis workflows
- **Retail Analytics**: Measure spacing between products or customers in retail environments (e.g., measure product spacing on shelves, assess customer spacing, monitor display arrangements), enabling retail analytics workflows
- **Agricultural Monitoring**: Measure spacing between crops, plants, or agricultural objects (e.g., measure crop spacing, assess plant gaps, monitor field arrangements), enabling agricultural monitoring workflows

## Connecting to Other Blocks

This block receives detection predictions and produces distance_cm and distance_pixel values:

- **After object detection or instance segmentation blocks** to measure distances between detected objects (e.g., measure distance between detected objects, calculate spacing from detections, analyze object relationships), enabling detection-to-measurement workflows
- **Before logic blocks** like Continue If to make decisions based on distance measurements (e.g., continue if distance is safe, filter based on spacing requirements, make decisions using distance thresholds), enabling distance-based decision workflows
- **Before analysis blocks** to analyze spatial relationships between objects (e.g., analyze object spacing, process distance measurements, work with spatial data), enabling spatial analysis workflows
- **Before notification blocks** to alert when distances violate thresholds (e.g., send alerts when spacing is too close, notify on distance violations, trigger actions based on measurements), enabling distance-based notification workflows
- **Before data storage blocks** to record distance measurements (e.g., store distance measurements, log spacing data, record spatial metrics), enabling distance measurement logging workflows
- **In measurement pipelines** where distance calculation is part of a larger spatial analysis workflow (e.g., measure distances in analysis pipelines, calculate spacing in monitoring systems, process spatial measurements in chains), enabling spatial measurement pipeline workflows

## Requirements

This block requires detection predictions with bounding boxes and class names. The image should be captured from a perpendicular camera view (top-down or frontal) to minimize perspective distortion and ensure accurate 2D distance measurements. For reference object calibration, a reference object with known dimensions must be present in the detections. For pixel-to-centimeter ratio calibration, the pixel ratio must be pre-calibrated or known for the camera setup. Objects must not overlap and must have a gap along the specified measurement axis (horizontal or vertical). The block assumes objects are on the same plane for accurate 2D measurement.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/distance_measurement@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `object_1_class_name` | `str` | Class name of the first object to measure distance from. Must match exactly the class name in the detection predictions. The block searches for this class name in the detections and uses its bounding box for distance calculation. Example: if detections contain objects labeled 'person', 'car', 'bicycle', use 'person' to measure distance from a person to another object. The class name is case-sensitive and must match exactly.. | ❌ |
| `object_2_class_name` | `str` | Class name of the second object to measure distance to. Must match exactly the class name in the detection predictions. The block searches for this class name in the detections and uses its bounding box for distance calculation. Example: if detections contain objects labeled 'person', 'car', 'bicycle', use 'person' to measure distance to a person from another object. The class name is case-sensitive and must match exactly. The block measures the gap between object_1 and object_2 along the specified reference_axis.. | ❌ |
| `reference_axis` | `str` | Axis along which to measure the distance between the two objects. Options: 'horizontal' measures distance along the X-axis (left-right gap between objects, useful when objects are side-by-side), or 'vertical' measures distance along the Y-axis (top-bottom gap between objects, useful when objects are stacked vertically). The distance is measured as the gap between bounding boxes along the selected axis. Objects must have a gap along this axis (not overlap) for accurate measurement. Choose based on object orientation: horizontal for side-by-side objects, vertical for stacked objects.. | ❌ |
| `calibration_method` | `str` | Method to calibrate pixel measurements to real-world units (centimeters). Options: 'reference object' (uses a reference object with known dimensions in the image to calculate pixel-to-centimeter ratio automatically, more flexible for different scales), or 'pixel to centimeter' (uses a pre-calibrated pixel ratio directly, simpler but requires known scale). For reference object method, a reference object must be present in detections with known width and height. For pixel ratio method, the pixel_ratio must be pre-calibrated for your camera setup.. | ❌ |
| `reference_object_class_name` | `str` | Class name of the reference object used for calibration (only used when calibration_method is 'reference object'). Must match exactly the class name in the detection predictions. The reference object must have known real-world dimensions (reference_width and reference_height). The block measures the reference object's pixel dimensions and calculates a pixel-to-centimeter ratio to convert all distance measurements. Default is 'reference-object'. The reference object must be present in the detections and should be clearly visible and correctly detected.. | ✅ |
| `reference_width` | `float` | Real-world width of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical width of the reference object. The block measures the reference object's width in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm wide card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling.. | ✅ |
| `reference_height` | `float` | Real-world height of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical height of the reference object. The block measures the reference object's height in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm tall card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling.. | ✅ |
| `pixel_ratio` | `float` | Pixel-to-centimeter conversion ratio for the image (only used when calibration_method is 'pixel to centimeter'). Must be greater than 0. This value represents how many pixels equal 1 centimeter. Example: if 100 pixels = 1 centimeter, use 100. The block divides pixel distances by this ratio to convert to centimeters. This ratio must be pre-calibrated for your specific camera setup, viewing distance, and image resolution. Typical values range from 10-500 depending on camera distance and resolution. A higher ratio means more pixels per centimeter (objects appear larger, camera is closer), a lower ratio means fewer pixels per centimeter (objects appear smaller, camera is farther).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Distance Measurement` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)
    - outputs: [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Email Notification`](email_notification.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Offset`](detection_offset.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stitch Images`](stitch_images.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Distance Measurement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions containing bounding boxes and class names for objects in the image. Must include detections for the two objects to measure (object_1_class_name and object_2_class_name) and optionally a reference object (if using reference object calibration method). The bounding boxes will be used to calculate distances between objects. Both object detection and instance segmentation predictions are supported. The detections must contain class_name information to identify objects..
        - `reference_object_class_name` (*[`string`](../kinds/string.md)*): Class name of the reference object used for calibration (only used when calibration_method is 'reference object'). Must match exactly the class name in the detection predictions. The reference object must have known real-world dimensions (reference_width and reference_height). The block measures the reference object's pixel dimensions and calculates a pixel-to-centimeter ratio to convert all distance measurements. Default is 'reference-object'. The reference object must be present in the detections and should be clearly visible and correctly detected..
        - `reference_width` (*[`float`](../kinds/float.md)*): Real-world width of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical width of the reference object. The block measures the reference object's width in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm wide card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling..
        - `reference_height` (*[`float`](../kinds/float.md)*): Real-world height of the reference object in centimeters (only used when calibration_method is 'reference object'). Must be greater than 0. This is the actual physical height of the reference object. The block measures the reference object's height in pixels and divides by this value to calculate the pixel-to-centimeter ratio. Use accurate measurements for best results. Example: if your reference object is a 2.5cm tall card, use 2.5. The reference_width and reference_height are used to calculate separate width and height ratios, then averaged for more accurate scaling..
        - `pixel_ratio` (*[`float`](../kinds/float.md)*): Pixel-to-centimeter conversion ratio for the image (only used when calibration_method is 'pixel to centimeter'). Must be greater than 0. This value represents how many pixels equal 1 centimeter. Example: if 100 pixels = 1 centimeter, use 100. The block divides pixel distances by this ratio to convert to centimeters. This ratio must be pre-calibrated for your specific camera setup, viewing distance, and image resolution. Typical values range from 10-500 depending on camera distance and resolution. A higher ratio means more pixels per centimeter (objects appear larger, camera is closer), a lower ratio means fewer pixels per centimeter (objects appear smaller, camera is farther)..

    - output
    
        - `distance_cm` ([`integer`](../kinds/integer.md)): Integer value.
        - `distance_pixel` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Distance Measurement` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/distance_measurement@v1",
	    "predictions": "$steps.model.predictions",
	    "object_1_class_name": "car",
	    "object_2_class_name": "person",
	    "reference_axis": "vertical",
	    "calibration_method": "<block_does_not_provide_example>",
	    "reference_object_class_name": "reference-object",
	    "reference_width": 2.5,
	    "reference_height": 2.5,
	    "pixel_ratio": 100
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

