
# Dynamic Zone



??? "Class: `DynamicZonesBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/dynamic_zones/v1.py">inference.core.workflows.core_steps.transformations.dynamic_zones.v1.DynamicZonesBlockV1</a>
    



Generate simplified polygon zones from instance segmentation detections by converting masks to contours, computing convex hulls, reducing polygon vertices to a specified count using Douglas-Peucker approximation, optionally applying least squares edge fitting, and scaling polygons to create geometric zones based on detected object shapes for zone-based analytics, spatial filtering, and region-of-interest definition workflows.

## How This Block Works

This block creates simplified polygon zones from instance segmentation detections by converting complex mask shapes into geometrically convex polygons with a specified number of vertices. The block:

1. Receives instance segmentation predictions containing masks (polygon representations) for detected objects
2. Converts masks to contours:
   - Extracts contours from each detection mask using mask-to-polygon conversion
   - Selects the largest contour from each detection (handles multiple contours per mask)
3. Computes convex hull:
   - Calculates the convex hull of the largest contour using OpenCV's convex hull algorithm
   - Ensures the resulting polygon is geometrically convex (no inward-facing angles)
   - Creates a simplified outer boundary that encompasses all points in the contour
4. Simplifies polygon to required vertex count:
   - Uses Douglas-Peucker polygon approximation algorithm to reduce vertices
   - Iteratively adjusts epsilon parameter to achieve the target number of vertices
   - Uses binary search to find the optimal epsilon value that produces the requested vertex count
   - Handles convergence: if exact vertex count cannot be achieved, pads or truncates vertices
5. Optionally applies least squares edge fitting:
   - If `apply_least_squares` is enabled, refines polygon edges by fitting lines to original contour points
   - Selects contour points between polygon vertices
   - Optionally filters to midpoint fraction (e.g., uses only central portion of each edge) to avoid edge effects
   - Fits least squares lines to selected contour points for each edge
   - Calculates intersections of fitted lines to create refined vertex positions
   - Produces a polygon that better aligns with the original contour shape
6. Scales polygon (if `scale_ratio` != 1):
   - Calculates polygon centroid (center of mass)
   - Scales polygon relative to centroid by the specified scale ratio
   - Expands or contracts polygon outward from center (scale > 1 expands, scale < 1 contracts)
   - Useful for creating buffer zones or adjusting zone boundaries
7. Updates detections with simplified polygons:
   - Stores simplified polygons in detection metadata under the polygon key
   - Regenerates masks from simplified polygons for updated detection representation
8. Returns simplified zones and updated detections:
   - `zones`: List of simplified polygons (one per detection) as coordinate lists
   - `predictions`: Updated detections with simplified polygons and masks
   - `simplification_converged`: Boolean indicating if all polygons converged to exact vertex count

The block enables creation of geometric zones from complex object shapes detected by segmentation models. It's particularly useful when zones need to be created based on detected object shapes (e.g., basketball courts, road segments, parking lots, fields) where the zone should match the object's outline but be simplified for performance and ease of use.

## Common Use Cases

- **Zone Creation from Detections**: Create polygon zones based on detected object shapes (e.g., create basketball court zones from court detections, generate road segment zones from road detections, create field zones from sports field detections), enabling detection-based zone workflows
- **Geometric Zone Simplification**: Simplify complex object shapes into geometrically convex polygons with controlled vertex counts (e.g., simplify irregular shapes to rectangles/quadrilaterals, reduce complex polygons to manageable vertex counts, create geometric zones from masks), enabling zone simplification workflows
- **Dynamic Zone Definition**: Dynamically define zones based on detected objects in images (e.g., define zones from detected regions, create zones from object shapes, generate zones from segmentation results), enabling dynamic zone workflows
- **Zone-Based Analytics Setup**: Prepare zones for zone-based analytics and filtering (e.g., prepare zones for time-in-zone analytics, create zones for zone-based filtering, set up zones for spatial analytics), enabling zone-based analytics workflows
- **Region-of-Interest Definition**: Define regions of interest based on detected object boundaries (e.g., define ROIs from object detections, create ROI zones from segmentation, generate interest regions from masks), enabling ROI definition workflows
- **Spatial Filtering and Analysis**: Create zones for spatial filtering and analysis operations (e.g., create zones for spatial filtering, prepare zones for area calculations, generate zones for spatial queries), enabling spatial analysis workflows

## Connecting to Other Blocks

This block receives instance segmentation predictions and produces simplified polygon zones:

- **After instance segmentation models** to create zones from detected object shapes (e.g., segmentation model to zones, masks to simplified polygons, detections to geometric zones), enabling segmentation-to-zone workflows
- **After detection filtering blocks** to create zones from filtered detections (e.g., filter detections then create zones, create zones from specific classes, generate zones from filtered results), enabling filter-to-zone workflows
- **Before zone-based analytics blocks** to provide simplified zones for analytics (e.g., zones for time-in-zone, zones for zone analytics, polygons for zone filtering), enabling zone-to-analytics workflows
- **Before visualization blocks** to display simplified zones (e.g., visualize zone polygons, display geometric zones, show simplified regions), enabling zone visualization workflows
- **Before spatial filtering blocks** to provide zones for spatial operations (e.g., zones for overlap filtering, polygons for spatial queries, regions for area calculations), enabling zone-to-filter workflows
- **In workflow outputs** to provide simplified zones as final output (e.g., zone generation workflows, polygon extraction workflows, geometric zone outputs), enabling zone output workflows

## Requirements

This block requires instance segmentation predictions with masks (polygon data). Input detections should be filtered to contain only the desired classes of interest before processing. The `required_number_of_vertices` parameter specifies the target vertex count for simplified polygons (e.g., 4 for rectangles/quadrilaterals, 3 for triangles). The block uses iterative Douglas-Peucker approximation with binary search to achieve the target vertex count, with a maximum of 1000 iterations. If convergence to exact vertex count fails, vertices are padded or truncated. The `scale_ratio` parameter (default 1) scales polygons relative to their centroid. The `apply_least_squares` parameter (default False) enables edge fitting to better align polygon edges with original contours. The `midpoint_fraction` parameter (0-1, default 1) controls which portion of contour points are used for least squares fitting (1 = all points, lower values use central portions of edges). The block outputs simplified polygons as lists of coordinate pairs, updated detections with simplified polygons, and a convergence flag.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/dynamic_zone@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `required_number_of_vertices` | `int` | Target number of vertices for simplified polygons. The block uses Douglas-Peucker polygon approximation with iterative binary search to reduce polygon vertices to this count. Common values: 4 for rectangles/quadrilaterals, 3 for triangles, 6+ for more complex shapes. The algorithm attempts to converge to this exact count; if convergence fails (within iteration limit), vertices are padded or truncated to match the count.. | ✅ |
| `scale_ratio` | `float` | Scale factor to expand or contract resulting polygons relative to their centroid. Values > 1 expand polygons outward from center (create buffer zones), values < 1 contract polygons inward. Value of 1 (default) means no scaling. Scaling is applied after polygon simplification. Useful for creating buffer zones or adjusting zone boundaries.. | ✅ |
| `apply_least_squares` | `bool` | If True, applies least squares line fitting to refine polygon edges by aligning them with original contour points. For each edge of the simplified polygon, fits a line to contour points between vertices, then calculates intersections of fitted lines to create refined vertex positions. Produces polygons that better match the original contour shape, especially useful when simplified polygon vertices don't align well with contour edges.. | ✅ |
| `midpoint_fraction` | `float` | Fraction (0-1) of contour points to use for least squares fitting on each edge. Value of 1 (default) uses all contour points between vertices. Lower values use only the central portion of each edge (e.g., 0.9 uses 90% of points, centered). Useful when convex polygon vertices are not well-aligned with edges, as it focuses fitting on the central portion of edges rather than edge effects near vertices. Only applies when apply_least_squares is True.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Dynamic Zone` in version `v1`.

    - inputs: [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Image Contours`](image_contours.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Gaze Detection`](gaze_detection.md), [`Velocity`](velocity.md), [`S3 Sink`](s3_sink.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Camera Focus`](camera_focus.md), [`Detections Combine`](detections_combine.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`JSON Parser`](json_parser.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Webhook Sink`](webhook_sink.md), [`Identify Changes`](identify_changes.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Detection Event Log`](detection_event_log.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SIFT Comparison`](sift_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Template Matching`](template_matching.md), [`Detections Filter`](detections_filter.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Rectangle`](bounding_rectangle.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Webhook Sink`](webhook_sink.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Cache Set`](cache_set.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Dynamic Zone` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)*): Instance segmentation predictions containing masks (polygon data) for detected objects. Detections should be filtered to contain only desired classes of interest. Each detection's mask is converted to contours, and the largest contour is used to generate a simplified polygon zone. Supports instance segmentation format with mask data..
        - `required_number_of_vertices` (*[`integer`](../kinds/integer.md)*): Target number of vertices for simplified polygons. The block uses Douglas-Peucker polygon approximation with iterative binary search to reduce polygon vertices to this count. Common values: 4 for rectangles/quadrilaterals, 3 for triangles, 6+ for more complex shapes. The algorithm attempts to converge to this exact count; if convergence fails (within iteration limit), vertices are padded or truncated to match the count..
        - `scale_ratio` (*[`float`](../kinds/float.md)*): Scale factor to expand or contract resulting polygons relative to their centroid. Values > 1 expand polygons outward from center (create buffer zones), values < 1 contract polygons inward. Value of 1 (default) means no scaling. Scaling is applied after polygon simplification. Useful for creating buffer zones or adjusting zone boundaries..
        - `apply_least_squares` (*[`boolean`](../kinds/boolean.md)*): If True, applies least squares line fitting to refine polygon edges by aligning them with original contour points. For each edge of the simplified polygon, fits a line to contour points between vertices, then calculates intersections of fitted lines to create refined vertex positions. Produces polygons that better match the original contour shape, especially useful when simplified polygon vertices don't align well with contour edges..
        - `midpoint_fraction` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Fraction (0-1) of contour points to use for least squares fitting on each edge. Value of 1 (default) uses all contour points between vertices. Lower values use only the central portion of each edge (e.g., 0.9 uses 90% of points, centered). Useful when convex polygon vertices are not well-aligned with edges, as it focuses fitting on the central portion of edges rather than edge effects near vertices. Only applies when apply_least_squares is True..

    - output
    
        - `zones` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `simplification_converged` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `Dynamic Zone` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/dynamic_zone@v1",
	    "predictions": "$steps.instance_segmentation_model.predictions",
	    "required_number_of_vertices": 4,
	    "scale_ratio": 1.0,
	    "apply_least_squares": false,
	    "midpoint_fraction": 1.0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

