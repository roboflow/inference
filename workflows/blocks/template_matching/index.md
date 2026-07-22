
# Template Matching



??? "Class: `TemplateMatchingBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/template_matching/v1.py">inference.core.workflows.core_steps.classical_cv.template_matching.v1.TemplateMatchingBlockV1</a>
    



Locate instances of a template image within a larger image using template matching with normalized cross-correlation, finding exact or near-exact matches of the template pattern at any location in the image, outputting bounding box detections with optional NMS filtering for object detection, logo detection, pattern recognition, and template-based object localization workflows.

## How This Block Works

This block searches for occurrences of a template image within a larger input image using normalized cross-correlation template matching. The block:

1. Receives an input image and a template image (smaller pattern to search for)
2. Converts both images to grayscale for template matching (template matching typically works on grayscale images for efficiency and robustness)
3. Performs template matching using OpenCV's matchTemplate with TM_CCOEFF_NORMED method:
   - Slides the template across the input image at every possible position
   - Computes normalized cross-correlation coefficient at each position (measures similarity between template and image region)
   - Generates a similarity map showing how well the template matches at each location
4. Identifies match locations where similarity exceeds the matching_threshold:
   - Finds all positions where the correlation coefficient is greater than or equal to the threshold
   - Threshold values range from 0.0 to 1.0, with higher values requiring closer matches
   - Lower thresholds find more potential matches (including partial matches), higher thresholds find only very similar matches
5. Creates bounding boxes for each match:
   - Each match location becomes a detection with a bounding box matching the template's dimensions
   - All detections have confidence of 1.0 (they met the threshold requirement)
   - All detections are assigned class "template_match" and class_id 0
   - Each detection gets a unique detection ID for tracking
6. Optionally applies Non-Maximum Suppression (NMS) to filter overlapping detections:
   - Template matching often produces many overlapping detections at the same location (duplicate matches)
   - NMS removes overlapping detections, keeping only the best match in each area
   - NMS threshold controls how much overlap is allowed before removing detections
   - Can be disabled (apply_nms=False) if NMS becomes computationally intractable with very large numbers of matches
7. Attaches metadata to detections:
   - Sets parent_id to reference the input image
   - Sets prediction_type to "object-detection"
   - Stores image dimensions for coordinate reference
   - Attaches parent coordinate information for workflow tracking
8. Returns detection predictions in sv.Detections format along with the total number of matches found

The block uses normalized cross-correlation which is effective for finding exact or near-exact template matches. It works best when the template appears in the image at the same scale, rotation, and lighting conditions. The method tends to produce many overlapping detections for the same match location, which is why NMS filtering is important. However, in cases with extremely large numbers of matches (e.g., repeating patterns), NMS may become computationally expensive and can be disabled if needed.

## Common Use Cases

- **Logo and Brand Detection**: Find specific logos or brand elements within images (e.g., detect company logos in photos, find brand markers in images, locate specific logo patterns in scenes), enabling logo detection workflows
- **Exact Pattern Matching**: Locate specific patterns or objects that appear identically in images (e.g., find specific UI elements in screenshots, detect exact patterns in images, locate specific visual elements), enabling exact pattern detection workflows
- **Quality Control and Inspection**: Find reference patterns or features for quality inspection (e.g., detect specific features in manufacturing images, find reference markers for alignment, locate inspection targets), enabling quality control workflows
- **Object Localization**: Locate specific objects or regions when exact appearance is known (e.g., find specific objects with known appearance, locate reference objects in images, detect specific visual elements), enabling template-based object localization
- **Document Processing**: Find specific elements or regions in documents (e.g., locate form fields in documents, detect specific document elements, find reference markers in scanned documents), enabling document processing workflows
- **UI Element Detection**: Detect specific UI components or elements in interface images (e.g., find buttons in UI screenshots, locate specific UI elements, detect interface components), enabling UI analysis workflows

## Connecting to Other Blocks

This block receives an image and template, and produces detection predictions:

- **After image input blocks** to find template patterns in input images (e.g., search for templates in input images, locate patterns in camera feeds, find templates in image streams), enabling template matching workflows
- **After preprocessing blocks** to find templates in preprocessed images (e.g., match templates after image enhancement, find patterns in filtered images, locate templates in normalized images), enabling preprocessed template matching
- **Before visualization blocks** to visualize template match locations (e.g., visualize detected template matches, display bounding boxes for matches, show template match results), enabling template match visualization workflows
- **Before filtering blocks** to filter template matches by criteria (e.g., filter matches by location, select specific match regions, refine template match results), enabling filtered template matching workflows
- **Before crop blocks** to extract regions around template matches (e.g., crop areas around matches, extract match regions for analysis, crop template match locations), enabling template-based region extraction
- **In quality control workflows** where template matching is used for inspection or alignment (e.g., find reference markers for alignment, detect inspection targets, locate quality control features), enabling quality control template matching workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/template_matching@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `matching_threshold` | `float` | Minimum similarity threshold (0.0 to 1.0) required for a template match. Higher values (closer to 1.0) require very close matches and find fewer but more precise matches. Lower values (closer to 0.0) allow more lenient matches and find more potential matches including partial matches. Default is 0.8, which requires fairly close matches. Use lower thresholds (0.6-0.7) to find more matches or handle slight variations. Use higher thresholds (0.85-0.95) for exact matches only. The threshold compares normalized cross-correlation coefficients from template matching.. | ✅ |
| `apply_nms` | `bool` | Whether to apply Non-Maximum Suppression (NMS) to filter overlapping detections. Template matching often produces many overlapping detections at the same location. NMS removes overlapping detections, keeping only the best match in each area. Default is True (recommended for most cases). Set to False if: (1) the number of matches is extremely large (NMS may become computationally expensive), (2) you want to see all raw matches without filtering, or (3) matches are intentionally close together and should all be kept. When disabled, you may see many duplicate detections for the same match location.. | ✅ |
| `nms_threshold` | `float` | Intersection over Union (IoU) threshold for Non-Maximum Suppression. Only relevant when apply_nms is True. Detections with IoU overlap greater than this threshold are considered duplicates, and only the detection with highest confidence is kept. Lower values (0.3-0.4) are more aggressive at removing overlaps, removing detections that are only slightly overlapping. Higher values (0.6-0.7) are more lenient, only removing heavily overlapping detections. Default is 0.5, which provides balanced overlap filtering. Adjust based on how much overlap you expect between template matches and how close together valid matches can be.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Template Matching` in version `v1`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Cosine Similarity`](cosine_similarity.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`PLC Reader`](plc_reader.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PLC Writer`](plc_writer.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Motion Detection`](motion_detection.md), [`Detections Consensus`](detections_consensus.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`VLM As Detector`](vlm_as_detector.md), [`JSON Parser`](json_parser.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Identify Outliers`](identify_outliers.md), [`Identify Changes`](identify_changes.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SIFT Comparison`](sift_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Event Writer`](event_writer.md), [`S3 Sink`](s3_sink.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`Image Preprocessing`](image_preprocessing.md), [`Time in Zone`](timein_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Dominant Color`](dominant_color.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`Image Threshold`](image_threshold.md), [`Detections Merge`](detections_merge.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Identify Outliers`](identify_outliers.md), [`Event Writer`](event_writer.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Combine`](detections_combine.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Blur`](image_blur.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Polygon Visualization`](polygon_visualization.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Template Matching` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Large image in which to search for the template pattern. The template will be searched across this entire image at all possible positions. The image is converted to grayscale internally for template matching. Template matching works best when the image and template have similar lighting conditions and the template appears at similar scale and orientation in the image..
        - `template` (*[`image`](../kinds/image.md)*): Small template image pattern to search for within the input image. The template should be smaller than the input image. The template is converted to grayscale internally for matching. Template matching finds exact or near-exact matches of this template at any location in the input image. Works best when the template appears in the image at the same scale, rotation, and lighting conditions. The template's dimensions determine the size of the detection bounding boxes..
        - `matching_threshold` (*[`float`](../kinds/float.md)*): Minimum similarity threshold (0.0 to 1.0) required for a template match. Higher values (closer to 1.0) require very close matches and find fewer but more precise matches. Lower values (closer to 0.0) allow more lenient matches and find more potential matches including partial matches. Default is 0.8, which requires fairly close matches. Use lower thresholds (0.6-0.7) to find more matches or handle slight variations. Use higher thresholds (0.85-0.95) for exact matches only. The threshold compares normalized cross-correlation coefficients from template matching..
        - `apply_nms` (*[`boolean`](../kinds/boolean.md)*): Whether to apply Non-Maximum Suppression (NMS) to filter overlapping detections. Template matching often produces many overlapping detections at the same location. NMS removes overlapping detections, keeping only the best match in each area. Default is True (recommended for most cases). Set to False if: (1) the number of matches is extremely large (NMS may become computationally expensive), (2) you want to see all raw matches without filtering, or (3) matches are intentionally close together and should all be kept. When disabled, you may see many duplicate detections for the same match location..
        - `nms_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Intersection over Union (IoU) threshold for Non-Maximum Suppression. Only relevant when apply_nms is True. Detections with IoU overlap greater than this threshold are considered duplicates, and only the detection with highest confidence is kept. Lower values (0.3-0.4) are more aggressive at removing overlaps, removing detections that are only slightly overlapping. Higher values (0.6-0.7) are more lenient, only removing heavily overlapping detections. Default is 0.5, which provides balanced overlap filtering. Adjust based on how much overlap you expect between template matches and how close together valid matches can be..

    - output
    
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `number_of_matches` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Template Matching` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/template_matching@v1",
	    "image": "$inputs.image",
	    "template": "$inputs.template",
	    "matching_threshold": 0.8,
	    "apply_nms": "$inputs.apply_nms",
	    "nms_threshold": "$inputs.nms_threshold"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

