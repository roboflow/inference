
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

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Transformation`](detections_transformation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`S3 Sink`](s3_sink.md), [`Perspective Correction`](perspective_correction.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Email Notification`](email_notification.md), [`Velocity`](velocity.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Motion Detection`](motion_detection.md), [`Detections Filter`](detections_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Clip Comparison`](clip_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Slack Notification`](slack_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Track Class Lock`](track_class_lock.md), [`Local File Sink`](local_file_sink.md), [`Time in Zone`](timein_zone.md), [`Cosine Similarity`](cosine_similarity.md), [`MQTT Writer`](mqtt_writer.md), [`Image Contours`](image_contours.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixel Color Count`](pixel_color_count.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Template Matching`](template_matching.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Motion Detection`](motion_detection.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Filter`](detections_filter.md), [`Blur Visualization`](blur_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`Overlap Filter`](overlap_filter.md), [`Time in Zone`](timein_zone.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Time in Zone`](timein_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Camera Calibration`](camera_calibration.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Detections Transformation`](detections_transformation.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Combine`](detections_combine.md), [`Size Measurement`](size_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Path Deviation`](path_deviation.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Stitch`](detections_stitch.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`PLC Writer`](plc_writer.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
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

