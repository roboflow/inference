
# Time in Zone



## v3

??? "Class: `TimeInZoneBlockV3`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/time_in_zone/v3.py">inference.core.workflows.core_steps.analytics.time_in_zone.v3.TimeInZoneBlockV3</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Calculate and track the time spent by tracked objects within one or more defined polygon zones, measure duration of object presence in specific areas (supporting multiple zones where objects are considered 'in zone' if present in any zone), filter detections based on zone membership, reset time tracking when objects leave zones, and enable zone-based analytics, dwell time analysis, and presence monitoring workflows.

## How This Block Works

This block measures how long each tracked object has been inside one or more defined polygon zones by tracking entry and exit times for each unique track ID. The block supports multiple zones, treating objects as 'in zone' if they are present in any of the defined zones. The block:

1. Receives tracked detection predictions with track IDs, an image with embedded video metadata, and polygon zone definition(s) (single zone or list of zones)
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts fps, frame_number, frame_timestamp, video_identifier, and video source information
   - Uses video_identifier to maintain separate tracking state for different videos
3. Validates that detections have track IDs (tracker_id must be present):
   - Requires detections to come from a tracking block (e.g., Byte Tracker)
   - Each object must have a unique tracker_id that persists across frames
   - Raises an error if tracker_id is missing
4. Normalizes zone input to a list of polygons:
   - Accepts a single polygon zone or a list of polygon zones
   - Automatically wraps single polygons in a list for consistent processing
   - Validates nesting depth and coordinate format for all zones
   - Enables flexible zone input formats (single zone or multiple zones)
5. Initializes or retrieves polygon zones for the video:
   - Creates a list of PolygonZone objects from zone coordinates for each unique zone combination
   - Validates zone coordinates (each zone must be a list of at least 3 points, each with 2 coordinates)
   - Stores zone configurations in an OrderedDict with zone cache (max 100 zone combinations)
   - Uses zone key combining video_identifier and zone coordinates for cache lookup
   - Implements FIFO eviction when cache exceeds 100 zone combinations
   - Configures triggering anchor point (e.g., CENTER, BOTTOM_CENTER) for zone detection
6. Initializes or retrieves time tracking state for the video:
   - Maintains a dictionary tracking when each track_id entered any zone
   - Stores entry timestamps per video using video_identifier
   - Maintains separate tracking state for each video
7. Calculates current timestamp for time measurement:
   - For video files: Calculates timestamp as frame_number / fps
   - For streamed video: Uses frame_timestamp from metadata
   - Provides accurate time measurement for duration calculation
8. Checks which detections are in any zone:
   - Tests each detection against all polygon zones using polygon zone triggers
   - Creates a matrix of zone membership (zones x detections)
   - Uses logical OR operation: objects are considered 'in zone' if they're in ANY of the zones
   - The triggering_anchor determines which point on the bounding box is checked (CENTER, BOTTOM_CENTER, etc.)
   - Returns boolean for each detection indicating zone membership in any zone
9. Updates time tracking for each tracked object:
   - **For objects entering any zone**: Records entry timestamp if not already tracked
   - **For objects in any zone**: Calculates time spent as current_timestamp - entry_timestamp
   - **For objects leaving all zones**: 
     - If reset_out_of_zone_detections is True: Removes entry timestamp (resets to 0)
     - If reset_out_of_zone_detections is False: Keeps entry timestamp (continues tracking)
10. Handles out-of-zone detections:
    - **If remove_out_of_zone_detections is True**: Filters out detections outside all zones from output
    - **If remove_out_of_zone_detections is False**: Includes out-of-zone detections with time = 0
11. Adds time_in_zone information to each detection:
    - Attaches time_in_zone value (in seconds) to each detection as metadata
    - Objects in any zone: Time represents duration spent in any zone
    - Objects outside all zones: Time is 0 (if not reset) or undefined (if removed)
12. Returns detections with time_in_zone information:
    - Outputs tracked detections enhanced with time_in_zone metadata
    - Filtered or unfiltered based on remove_out_of_zone_detections setting
    - Maintains all original detection properties plus time tracking information

The block maintains persistent tracking state across frames, allowing accurate cumulative time measurement for objects that remain in any zone over multiple frames. Time is measured from when an object first enters any zone (based on its track_id) until the current frame, providing real-time duration tracking. When multiple zones are provided, objects are considered 'in zone' if their anchor point is inside any of the zones, allowing tracking across multiple areas as a single combined zone. The zone cache efficiently manages multiple zone configurations per video using FIFO eviction to limit memory usage. The triggering anchor determines which part of the bounding box is used for zone detection, enabling different zone entry/exit behaviors based on object position.

## Common Use Cases

- **Multi-Zone Dwell Time Analysis**: Measure how long objects remain in any of multiple areas for behavior analysis (e.g., measure customer time in any store section, track time spent in multiple parking areas, analyze time in overlapping zones), enabling multi-zone dwell time analytics workflows
- **Zone-Based Monitoring**: Monitor object presence across multiple defined areas for security and safety (e.g., detect loitering in any restricted area, monitor time in multiple danger zones, track presence across secure zones), enabling multi-zone monitoring workflows
- **Retail Analytics**: Track customer time across multiple store sections for retail insights (e.g., measure time in any product aisle, analyze shopping patterns across departments, track engagement in multiple zones), enabling multi-zone retail analytics workflows
- **Occupancy Management**: Measure time objects spend in any of multiple spaces for space utilization (e.g., track vehicle parking duration in multiple lots, measure table occupancy across zones, analyze space usage in multiple areas), enabling multi-zone occupancy management workflows
- **Safety Compliance**: Monitor time violations across multiple restricted or time-limited zones (e.g., detect extended stays in any hazardous area, monitor time limit violations across zones, track safety compliance in multiple areas), enabling multi-zone safety monitoring workflows
- **Traffic Analysis**: Measure time vehicles spend in any of multiple traffic zones or intersections (e.g., track time at multiple intersections, measure queue waiting time across zones, analyze traffic flow in multiple areas), enabling multi-zone traffic analytics workflows

## Connecting to Other Blocks

This block receives an image with embedded video metadata, tracked detections, and zone coordinates (single or multiple zones), and produces timed_detections with time_in_zone metadata:

- **After Byte Tracker blocks** to measure time for tracked objects across multiple zones (e.g., track time in multiple zones for tracked objects, measure dwell time with consistent IDs across areas, analyze tracked object presence in multiple zones), enabling tracking-to-time workflows
- **After zone definition blocks** to apply time tracking to multiple defined areas (e.g., measure time across multiple polygon zones, track duration in custom multi-zone configurations, analyze zone-based presence across areas), enabling zone-to-time workflows
- **Before logic blocks** like Continue If to make decisions based on time in any zone (e.g., continue if time exceeds threshold in any zone, filter based on dwell time across zones, trigger actions on time violations in multiple areas), enabling time-based decision workflows
- **Before analysis blocks** to analyze time-based metrics across multiple zones (e.g., analyze dwell time patterns across zones, process time-in-zone data for multiple areas, work with duration metrics across zones), enabling time analysis workflows
- **Before notification blocks** to alert on time violations or thresholds in any zone (e.g., alert on extended stays in any zone, notify on time limit violations across areas, trigger time-based alerts for multiple zones), enabling time-based notification workflows
- **Before data storage blocks** to record time metrics across multiple zones (e.g., store dwell time data for multiple areas, log time-in-zone metrics across zones, record duration measurements for multiple zones), enabling time metrics logging workflows

## Version Differences

**Enhanced from v2:**

- **Multiple Zone Support**: Supports tracking time across multiple polygon zones simultaneously, where objects are considered 'in zone' if they're present in any of the defined zones, enabling multi-zone time tracking and analysis
- **Flexible Zone Input**: Accepts either a single polygon zone or a list of polygon zones, automatically normalizing the input to handle both formats seamlessly
- **Zone Cache Management**: Implements a zone cache with FIFO eviction (max 100 zone combinations) to efficiently manage multiple zone configurations per video while limiting memory usage
- **Combined Zone Logic**: Uses logical OR operation across all zones, allowing tracking across multiple areas as a unified zone system for comprehensive presence monitoring
- **Enhanced Zone Key System**: Uses combined zone keys (video_identifier + zone coordinates) for cache lookup, enabling efficient storage and retrieval of zone configurations

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The zone can be a single polygon or a list of polygons, where each polygon must be defined as a list of at least 3 points, with each point being a list or tuple of exactly 2 coordinates (x, y). The image's video_metadata should include frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time measurements. The block maintains persistent tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate time measurement, detections should be provided consistently across frames with valid track IDs. When multiple zones are provided, objects are considered 'in zone' if they're present in any of the zones.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/time_in_zone@v3`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `zone` | `List[Any]` | Polygon zone coordinates defining one or more areas for time measurement. Can be a single polygon zone or a list of polygon zones. Each zone must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example for single zone: [(100, 100), (100, 200), (300, 200), (300, 100)]. Example for multiple zones: [[(100, 100), (100, 200), (300, 200), (300, 100)], [(400, 400), (400, 500), (600, 500), (600, 400)]]. Objects are considered 'in zone' if their triggering_anchor point is inside ANY of the provided zones. Zone coordinates are validated and PolygonZone objects are created for each zone. Zone configurations are cached (max 100 combinations) with FIFO eviction.. | ✅ |
| `triggering_anchor` | `str` | Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon(s). When multiple zones are provided, the object is considered 'in zone' if its anchor point is inside ANY of the zones. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'.. | ✅ |
| `remove_out_of_zone_detections` | `bool` | If True (default), detections found outside all zones are filtered out and not included in the output. Only detections inside at least one zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside all zones. Use True to focus analysis only on objects in any zone, or False to maintain all detections with zone status. When multiple zones are provided, objects are considered 'in zone' if present in any zone. Default is True for cleaner output focused on zone activity.. | ✅ |
| `reset_out_of_zone_detections` | `bool` | If True (default), when a tracked object leaves all zones, its time tracking is reset (entry timestamp is cleared). When the object re-enters any zone, time tracking starts from 0 again. If False, time tracking continues even after leaving all zones, and re-entry maintains cumulative time. Use True to measure current continuous time in any zone (resets on exit from all zones), or False to measure cumulative time across multiple entries. When multiple zones are provided, time is reset only when the object leaves all zones. Default is True for measuring continuous presence duration.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Time in Zone` in version `v3`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`JSON Parser`](json_parser.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Time in Zone` in version `v3`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image for the current video frame containing embedded video metadata (fps, frame_number, frame_timestamp, video_identifier, video source) required for time calculation and state management. The block extracts video_metadata from the WorkflowImageData object. The fps and frame_number are used for video files to calculate timestamps (timestamp = frame_number / fps). For streamed video, frame_timestamp is used directly. The video_identifier is used to maintain separate tracking state and zone configurations for different videos. Used for zone visualization and reference. The image dimensions are used to validate zone coordinates. This version supports multiple zones per video with efficient zone cache management..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked detection predictions (object detection or instance segmentation) with tracker_id information. Detections must come from a tracking block (e.g., Byte Tracker) that has assigned unique tracker_id values that persist across frames. Each detection must have a tracker_id to enable time tracking. The block calculates time_in_zone for each tracked object based on when its track_id first entered any of the zones. Objects are considered 'in zone' if their anchor point is inside any of the provided zones. The output will include the same detections enhanced with time_in_zone metadata (duration in seconds). If remove_out_of_zone_detections is True, only detections inside any zone are included in the output..
        - `zone` (*[`list_of_values`](../kinds/list_of_values.md)*): Polygon zone coordinates defining one or more areas for time measurement. Can be a single polygon zone or a list of polygon zones. Each zone must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example for single zone: [(100, 100), (100, 200), (300, 200), (300, 100)]. Example for multiple zones: [[(100, 100), (100, 200), (300, 200), (300, 100)], [(400, 400), (400, 500), (600, 500), (600, 400)]]. Objects are considered 'in zone' if their triggering_anchor point is inside ANY of the provided zones. Zone coordinates are validated and PolygonZone objects are created for each zone. Zone configurations are cached (max 100 combinations) with FIFO eviction..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon(s). When multiple zones are provided, the object is considered 'in zone' if its anchor point is inside ANY of the zones. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'..
        - `remove_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), detections found outside all zones are filtered out and not included in the output. Only detections inside at least one zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside all zones. Use True to focus analysis only on objects in any zone, or False to maintain all detections with zone status. When multiple zones are provided, objects are considered 'in zone' if present in any zone. Default is True for cleaner output focused on zone activity..
        - `reset_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), when a tracked object leaves all zones, its time tracking is reset (entry timestamp is cleared). When the object re-enters any zone, time tracking starts from 0 again. If False, time tracking continues even after leaving all zones, and re-entry maintains cumulative time. Use True to measure current continuous time in any zone (resets on exit from all zones), or False to measure cumulative time across multiple entries. When multiple zones are provided, time is reset only when the object leaves all zones. Default is True for measuring continuous presence duration..

    - output
    
        - `timed_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Time in Zone` in version `v3`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/time_in_zone@v3",
	    "image": "$inputs.image",
	    "detections": "$steps.object_detection_model.predictions",
	    "zone": [
	        [
	            100,
	            100
	        ],
	        [
	            100,
	            200
	        ],
	        [
	            300,
	            200
	        ],
	        [
	            300,
	            100
	        ]
	    ],
	    "triggering_anchor": "CENTER",
	    "remove_out_of_zone_detections": true,
	    "reset_out_of_zone_detections": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v2

??? "Class: `TimeInZoneBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/time_in_zone/v2.py">inference.core.workflows.core_steps.analytics.time_in_zone.v2.TimeInZoneBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Calculate and track the time spent by tracked objects within a defined polygon zone, measure duration of object presence in specific areas, filter detections based on zone membership, reset time tracking when objects leave zones, and enable zone-based analytics, dwell time analysis, and presence monitoring workflows.

## How This Block Works

This block measures how long each tracked object has been inside a defined polygon zone by tracking entry and exit times for each unique track ID. The block:

1. Receives tracked detection predictions with track IDs, an image with embedded video metadata, and a polygon zone definition
2. Extracts video metadata from the image:
   - Accesses video_metadata from the WorkflowImageData object
   - Extracts fps, frame_number, frame_timestamp, video_identifier, and video source information
   - Uses video_identifier to maintain separate tracking state for different videos
3. Validates that detections have track IDs (tracker_id must be present):
   - Requires detections to come from a tracking block (e.g., Byte Tracker)
   - Each object must have a unique tracker_id that persists across frames
   - Raises an error if tracker_id is missing
4. Initializes or retrieves a polygon zone for the video:
   - Creates a PolygonZone object from zone coordinates for each unique video
   - Validates zone coordinates (must be a list of at least 3 points, each with 2 coordinates)
   - Stores zone configuration per video using video_identifier
   - Configures triggering anchor point (e.g., CENTER, BOTTOM_CENTER) for zone detection
5. Initializes or retrieves time tracking state for the video:
   - Maintains a dictionary tracking when each track_id entered the zone
   - Stores entry timestamps per video using video_identifier
   - Maintains separate tracking state for each video
6. Calculates current timestamp for time measurement:
   - For video files: Calculates timestamp as frame_number / fps
   - For streamed video: Uses frame_timestamp from metadata
   - Provides accurate time measurement for duration calculation
7. Checks which detections are in the zone:
   - Uses polygon zone trigger to test if each detection's anchor point is inside the zone
   - The triggering_anchor determines which point on the bounding box is checked (CENTER, BOTTOM_CENTER, etc.)
   - Returns boolean for each detection indicating zone membership
8. Updates time tracking for each tracked object:
   - **For objects entering the zone**: Records entry timestamp if not already tracked
   - **For objects in the zone**: Calculates time spent as current_timestamp - entry_timestamp
   - **For objects leaving the zone**: 
     - If reset_out_of_zone_detections is True: Removes entry timestamp (resets to 0)
     - If reset_out_of_zone_detections is False: Keeps entry timestamp (continues tracking)
9. Handles out-of-zone detections:
   - **If remove_out_of_zone_detections is True**: Filters out detections outside the zone from output
   - **If remove_out_of_zone_detections is False**: Includes out-of-zone detections with time = 0
10. Adds time_in_zone information to each detection:
    - Attaches time_in_zone value (in seconds) to each detection as metadata
    - Objects in zone: Time represents duration spent in zone
    - Objects outside zone: Time is 0 (if not reset) or undefined (if removed)
11. Returns detections with time_in_zone information:
    - Outputs tracked detections enhanced with time_in_zone metadata
    - Filtered or unfiltered based on remove_out_of_zone_detections setting
    - Maintains all original detection properties plus time tracking information

The block maintains persistent tracking state across frames, allowing accurate cumulative time measurement for objects that remain in the zone over multiple frames. Time is measured from when an object first enters the zone (based on its track_id) until the current frame, providing real-time duration tracking. The zone is defined as a polygon with multiple points, allowing flexible area definitions. The triggering anchor determines which part of the bounding box is used for zone detection, enabling different zone entry/exit behaviors based on object position.

## Common Use Cases

- **Dwell Time Analysis**: Measure how long objects remain in specific areas for behavior analysis (e.g., measure customer dwell time in store sections, track time spent in parking spaces, analyze time in waiting areas), enabling dwell time analytics workflows
- **Zone-Based Monitoring**: Monitor object presence in defined areas for security and safety (e.g., detect loitering in restricted areas, monitor time in danger zones, track presence in secure zones), enabling zone monitoring workflows
- **Retail Analytics**: Track customer time in different store sections for retail insights (e.g., measure time in product aisles, analyze shopping patterns, track department engagement), enabling retail analytics workflows
- **Occupancy Management**: Measure time objects spend in spaces for space utilization (e.g., track vehicle parking duration, measure table occupancy time, analyze space usage patterns), enabling occupancy management workflows
- **Safety Compliance**: Monitor time violations in restricted or time-limited zones (e.g., detect extended stays in hazardous areas, monitor time limit violations, track safety compliance), enabling safety monitoring workflows
- **Traffic Analysis**: Measure time vehicles spend in traffic zones or intersections (e.g., track time at intersections, measure queue waiting time, analyze traffic flow patterns), enabling traffic analytics workflows

## Connecting to Other Blocks

This block receives an image with embedded video metadata, tracked detections, and zone coordinates, and produces timed_detections with time_in_zone metadata:

- **After Byte Tracker blocks** to measure time for tracked objects (e.g., track time in zones for tracked objects, measure dwell time with consistent IDs, analyze tracked object presence), enabling tracking-to-time workflows
- **After zone definition blocks** to apply time tracking to defined areas (e.g., measure time in polygon zones, track duration in custom zones, analyze zone-based presence), enabling zone-to-time workflows
- **Before logic blocks** like Continue If to make decisions based on time in zone (e.g., continue if time exceeds threshold, filter based on dwell time, trigger actions on time violations), enabling time-based decision workflows
- **Before analysis blocks** to analyze time-based metrics (e.g., analyze dwell time patterns, process time-in-zone data, work with duration metrics), enabling time analysis workflows
- **Before notification blocks** to alert on time violations or thresholds (e.g., alert on extended stays, notify on time limit violations, trigger time-based alerts), enabling time-based notification workflows
- **Before data storage blocks** to record time metrics (e.g., store dwell time data, log time-in-zone metrics, record duration measurements), enabling time metrics logging workflows

## Version Differences

**Enhanced from v1:**

- **Simplified Input**: Uses `image` input that contains embedded video metadata instead of requiring a separate `metadata` field, simplifying workflow connections and reducing input complexity
- **Improved Integration**: Better integration with image-based workflows since video metadata is accessed directly from the image object rather than requiring separate metadata input
- **Streamlined Workflow**: Reduces the number of inputs needed, making it easier to connect in workflows where image and metadata come from the same source

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The zone must be defined as a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The image's video_metadata should include frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time measurements. The block maintains persistent tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate time measurement, detections should be provided consistently across frames with valid track IDs.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/time_in_zone@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `zone` | `List[Any]` | Polygon zone coordinates defining the area for time measurement. Must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example: [(100, 100), (100, 200), (300, 200), (300, 100)] for a quadrilateral zone. The zone defines the polygon area where time tracking occurs. Objects are considered 'in zone' when their triggering_anchor point is inside this polygon. Zone coordinates are validated and a PolygonZone object is created for each video.. | ✅ |
| `triggering_anchor` | `str` | Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'.. | ✅ |
| `remove_out_of_zone_detections` | `bool` | If True (default), detections found outside the zone are filtered out and not included in the output. Only detections inside the zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside the zone. Use True to focus analysis only on objects in the zone, or False to maintain all detections with zone status. Default is True for cleaner output focused on zone activity.. | ✅ |
| `reset_out_of_zone_detections` | `bool` | If True (default), when a tracked object leaves the zone, its time tracking is reset (entry timestamp is cleared). When the object re-enters the zone, time tracking starts from 0 again. If False, time tracking continues even after leaving the zone, and re-entry maintains cumulative time. Use True to measure current continuous time in zone (resets on exit), or False to measure cumulative time across multiple entries. Default is True for measuring continuous presence duration.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Time in Zone` in version `v2`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SORT Tracker`](sort_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`JSON Parser`](json_parser.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Time in Zone` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image for the current video frame containing embedded video metadata (fps, frame_number, frame_timestamp, video_identifier, video source) required for time calculation and state management. The block extracts video_metadata from the WorkflowImageData object. The fps and frame_number are used for video files to calculate timestamps (timestamp = frame_number / fps). For streamed video, frame_timestamp is used directly. The video_identifier is used to maintain separate tracking state and zone configurations for different videos. Used for zone visualization and reference. The image dimensions are used to validate zone coordinates. This version simplifies input by embedding metadata in the image object rather than requiring a separate metadata field..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked detection predictions (object detection or instance segmentation) with tracker_id information. Detections must come from a tracking block (e.g., Byte Tracker) that has assigned unique tracker_id values that persist across frames. Each detection must have a tracker_id to enable time tracking. The block calculates time_in_zone for each tracked object based on when its track_id first entered the zone. The output will include the same detections enhanced with time_in_zone metadata (duration in seconds). If remove_out_of_zone_detections is True, only detections inside the zone are included in the output..
        - `zone` (*[`list_of_values`](../kinds/list_of_values.md)*): Polygon zone coordinates defining the area for time measurement. Must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example: [(100, 100), (100, 200), (300, 200), (300, 100)] for a quadrilateral zone. The zone defines the polygon area where time tracking occurs. Objects are considered 'in zone' when their triggering_anchor point is inside this polygon. Zone coordinates are validated and a PolygonZone object is created for each video..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'..
        - `remove_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), detections found outside the zone are filtered out and not included in the output. Only detections inside the zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside the zone. Use True to focus analysis only on objects in the zone, or False to maintain all detections with zone status. Default is True for cleaner output focused on zone activity..
        - `reset_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), when a tracked object leaves the zone, its time tracking is reset (entry timestamp is cleared). When the object re-enters the zone, time tracking starts from 0 again. If False, time tracking continues even after leaving the zone, and re-entry maintains cumulative time. Use True to measure current continuous time in zone (resets on exit), or False to measure cumulative time across multiple entries. Default is True for measuring continuous presence duration..

    - output
    
        - `timed_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Time in Zone` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/time_in_zone@v2",
	    "image": "$inputs.image",
	    "detections": "$steps.object_detection_model.predictions",
	    "zone": [
	        [
	            100,
	            100
	        ],
	        [
	            100,
	            200
	        ],
	        [
	            300,
	            200
	        ],
	        [
	            300,
	            100
	        ]
	    ],
	    "triggering_anchor": "CENTER",
	    "remove_out_of_zone_detections": true,
	    "reset_out_of_zone_detections": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `TimeInZoneBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/time_in_zone/v1.py">inference.core.workflows.core_steps.analytics.time_in_zone.v1.TimeInZoneBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Calculate and track the time spent by tracked objects within a defined polygon zone, measure duration of object presence in specific areas, filter detections based on zone membership, reset time tracking when objects leave zones, and enable zone-based analytics, dwell time analysis, and presence monitoring workflows.

## How This Block Works

This block measures how long each tracked object has been inside a defined polygon zone by tracking entry and exit times for each unique track ID. The block:

1. Receives tracked detection predictions with track IDs, an image, video metadata, and a polygon zone definition
2. Validates that detections have track IDs (tracker_id must be present):
   - Requires detections to come from a tracking block (e.g., Byte Tracker)
   - Each object must have a unique tracker_id that persists across frames
   - Raises an error if tracker_id is missing
3. Initializes or retrieves a polygon zone for the video:
   - Creates a PolygonZone object from zone coordinates for each unique video
   - Validates zone coordinates (must be a list of at least 3 points, each with 2 coordinates)
   - Stores zone configuration per video using video_identifier
   - Configures triggering anchor point (e.g., CENTER, BOTTOM_CENTER) for zone detection
4. Initializes or retrieves time tracking state for the video:
   - Maintains a dictionary tracking when each track_id entered the zone
   - Stores entry timestamps per video using video_identifier
   - Maintains separate tracking state for each video
5. Calculates current timestamp for time measurement:
   - For video files: Calculates timestamp as frame_number / fps
   - For streamed video: Uses frame_timestamp from metadata
   - Provides accurate time measurement for duration calculation
6. Checks which detections are in the zone:
   - Uses polygon zone trigger to test if each detection's anchor point is inside the zone
   - The triggering_anchor determines which point on the bounding box is checked (CENTER, BOTTOM_CENTER, etc.)
   - Returns boolean for each detection indicating zone membership
7. Updates time tracking for each tracked object:
   - **For objects entering the zone**: Records entry timestamp if not already tracked
   - **For objects in the zone**: Calculates time spent as current_timestamp - entry_timestamp
   - **For objects leaving the zone**: 
     - If reset_out_of_zone_detections is True: Removes entry timestamp (resets to 0)
     - If reset_out_of_zone_detections is False: Keeps entry timestamp (continues tracking)
8. Handles out-of-zone detections:
   - **If remove_out_of_zone_detections is True**: Filters out detections outside the zone from output
   - **If remove_out_of_zone_detections is False**: Includes out-of-zone detections with time = 0
9. Adds time_in_zone information to each detection:
   - Attaches time_in_zone value (in seconds) to each detection as metadata
   - Objects in zone: Time represents duration spent in zone
   - Objects outside zone: Time is 0 (if not reset) or undefined (if removed)
10. Returns detections with time_in_zone information:
    - Outputs tracked detections enhanced with time_in_zone metadata
    - Filtered or unfiltered based on remove_out_of_zone_detections setting
    - Maintains all original detection properties plus time tracking information

The block maintains persistent tracking state across frames, allowing accurate cumulative time measurement for objects that remain in the zone over multiple frames. Time is measured from when an object first enters the zone (based on its track_id) until the current frame, providing real-time duration tracking. The zone is defined as a polygon with multiple points, allowing flexible area definitions. The triggering anchor determines which part of the bounding box is used for zone detection, enabling different zone entry/exit behaviors based on object position.

## Common Use Cases

- **Dwell Time Analysis**: Measure how long objects remain in specific areas for behavior analysis (e.g., measure customer dwell time in store sections, track time spent in parking spaces, analyze time in waiting areas), enabling dwell time analytics workflows
- **Zone-Based Monitoring**: Monitor object presence in defined areas for security and safety (e.g., detect loitering in restricted areas, monitor time in danger zones, track presence in secure zones), enabling zone monitoring workflows
- **Retail Analytics**: Track customer time in different store sections for retail insights (e.g., measure time in product aisles, analyze shopping patterns, track department engagement), enabling retail analytics workflows
- **Occupancy Management**: Measure time objects spend in spaces for space utilization (e.g., track vehicle parking duration, measure table occupancy time, analyze space usage patterns), enabling occupancy management workflows
- **Safety Compliance**: Monitor time violations in restricted or time-limited zones (e.g., detect extended stays in hazardous areas, monitor time limit violations, track safety compliance), enabling safety monitoring workflows
- **Traffic Analysis**: Measure time vehicles spend in traffic zones or intersections (e.g., track time at intersections, measure queue waiting time, analyze traffic flow patterns), enabling traffic analytics workflows

## Connecting to Other Blocks

This block receives tracked detections, image, video metadata, and zone coordinates, and produces timed_detections with time_in_zone metadata:

- **After Byte Tracker blocks** to measure time for tracked objects (e.g., track time in zones for tracked objects, measure dwell time with consistent IDs, analyze tracked object presence), enabling tracking-to-time workflows
- **After zone definition blocks** to apply time tracking to defined areas (e.g., measure time in polygon zones, track duration in custom zones, analyze zone-based presence), enabling zone-to-time workflows
- **Before logic blocks** like Continue If to make decisions based on time in zone (e.g., continue if time exceeds threshold, filter based on dwell time, trigger actions on time violations), enabling time-based decision workflows
- **Before analysis blocks** to analyze time-based metrics (e.g., analyze dwell time patterns, process time-in-zone data, work with duration metrics), enabling time analysis workflows
- **Before notification blocks** to alert on time violations or thresholds (e.g., alert on extended stays, notify on time limit violations, trigger time-based alerts), enabling time-based notification workflows
- **Before data storage blocks** to record time metrics (e.g., store dwell time data, log time-in-zone metrics, record duration measurements), enabling time metrics logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The zone must be defined as a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates (x, y). The block requires video metadata with frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time measurements. The block maintains persistent tracking state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For accurate time measurement, detections should be provided consistently across frames with valid track IDs.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/time_in_zone@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `zone` | `List[Any]` | Polygon zone coordinates defining the area for time measurement. Must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] for a quadrilateral zone. The zone defines the polygon area where time tracking occurs. Objects are considered 'in zone' when their triggering_anchor point is inside this polygon. Zone coordinates are validated and a PolygonZone object is created for each video.. | ✅ |
| `triggering_anchor` | `str` | Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'.. | ✅ |
| `remove_out_of_zone_detections` | `bool` | If True (default), detections found outside the zone are filtered out and not included in the output. Only detections inside the zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside the zone. Use True to focus analysis only on objects in the zone, or False to maintain all detections with zone status. Default is True for cleaner output focused on zone activity.. | ✅ |
| `reset_out_of_zone_detections` | `bool` | If True (default), when a tracked object leaves the zone, its time tracking is reset (entry timestamp is cleared). When the object re-enters the zone, time tracking starts from 0 again. If False, time tracking continues even after leaving the zone, and re-entry maintains cumulative time. Use True to measure current continuous time in zone (resets on exit), or False to measure cumulative time across multiple entries. Default is True for measuring continuous presence duration.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Time in Zone` in version `v1`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Contrast Equalization`](contrast_equalization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemma API`](google_gemma_api.md), [`Email Notification`](email_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Image Preprocessing`](image_preprocessing.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Moondream2`](moondream2.md), [`Keypoint Visualization`](keypoint_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Time in Zone` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image for the current video frame. Used for zone visualization and reference. The block uses the image dimensions to validate zone coordinates. The image metadata may be used for time calculation if frame timestamps are needed..
        - `metadata` (*[`video_metadata`](../kinds/video_metadata.md)*): Video metadata containing frame rate (fps), frame number, frame timestamp, video identifier, and video source information required for time calculation and state management. The fps and frame_number are used for video files to calculate timestamps (timestamp = frame_number / fps). For streamed video, frame_timestamp is used directly. The video_identifier is used to maintain separate tracking state and zone configurations for different videos. The metadata must include valid fps for video files or frame_timestamp for streams to enable accurate time measurement..
        - `detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked detection predictions (object detection or instance segmentation) with tracker_id information. Detections must come from a tracking block (e.g., Byte Tracker) that has assigned unique tracker_id values that persist across frames. Each detection must have a tracker_id to enable time tracking. The block calculates time_in_zone for each tracked object based on when its track_id first entered the zone. The output will include the same detections enhanced with time_in_zone metadata (duration in seconds). If remove_out_of_zone_detections is True, only detections inside the zone are included in the output..
        - `zone` (*[`list_of_values`](../kinds/list_of_values.md)*): Polygon zone coordinates defining the area for time measurement. Must be a list of at least 3 points, where each point is a list or tuple of exactly 2 coordinates [x, y] or (x, y). Coordinates should be in pixel space matching the image dimensions. Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] for a quadrilateral zone. The zone defines the polygon area where time tracking occurs. Objects are considered 'in zone' when their triggering_anchor point is inside this polygon. Zone coordinates are validated and a PolygonZone object is created for each video..
        - `triggering_anchor` (*[`string`](../kinds/string.md)*): Point on the detection bounding box that must be inside the zone to consider the object 'in zone'. Options include: 'CENTER' (default, center of bounding box), 'BOTTOM_CENTER' (bottom center point), 'TOP_CENTER' (top center point), 'CENTER_LEFT' (center left point), 'CENTER_RIGHT' (center right point), and other Position enum values. The triggering anchor determines which part of the object's bounding box is checked against the zone polygon. Use CENTER for standard zone detection, BOTTOM_CENTER for ground-level zones (e.g., tracking feet/vehicle base), or other anchors based on detection needs. Default is 'CENTER'..
        - `remove_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), detections found outside the zone are filtered out and not included in the output. Only detections inside the zone are returned. If False, all detections are included in the output, with time_in_zone = 0 for objects outside the zone. Use True to focus analysis only on objects in the zone, or False to maintain all detections with zone status. Default is True for cleaner output focused on zone activity..
        - `reset_out_of_zone_detections` (*[`boolean`](../kinds/boolean.md)*): If True (default), when a tracked object leaves the zone, its time tracking is reset (entry timestamp is cleared). When the object re-enters the zone, time tracking starts from 0 again. If False, time tracking continues even after leaving the zone, and re-entry maintains cumulative time. Use True to measure current continuous time in zone (resets on exit), or False to measure cumulative time across multiple entries. Default is True for measuring continuous presence duration..

    - output
    
        - `timed_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Time in Zone` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/time_in_zone@v1",
	    "image": "$inputs.image",
	    "metadata": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "zone": "$inputs.zones",
	    "triggering_anchor": "CENTER",
	    "remove_out_of_zone_detections": true,
	    "reset_out_of_zone_detections": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

