
# GeoTag Detection



??? "Class: `GeoTagDetectionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/geotag_detection/v1.py">inference.core.workflows.core_steps.transformations.geotag_detection.v1.GeoTagDetectionBlockV1</a>
    



Convert object detection bounding boxes to real-world GPS coordinates using camera position metadata.

## How This Block Works

This block takes object detection predictions and camera GPS metadata (latitude, longitude, altitude) and projects each detection's pixel position to a real-world ground coordinate. The projection uses the camera's field of view and altitude to compute a ground footprint, then maps pixel offsets from image center to geographic offsets from the camera position.

1. **Receives detection predictions** from an upstream object detection block (any model that outputs bounding boxes)
2. **Takes camera GPS metadata** as inputs: latitude, longitude, altitude above ground, and optionally horizontal field of view
3. **Computes ground footprint** from altitude and FOV using basic trigonometry
4. **Projects each detection center** from pixel coordinates to lat/lon offset from camera position
5. **Outputs GeoJSON-ready records** with class, confidence, and geographic coordinates for each detection

The projection assumes a nadir (straight-down) camera orientation, which is accurate for most drone survey flights. For oblique angles, accuracy decreases with distance from image center.

## Common Use Cases

- **Drone Survey Analysis**: Process drone footage to map detected objects (vehicles, people, animals, structures) at their real-world locations for survey, inspection, or monitoring applications
- **Agricultural Monitoring**: Map crop damage, equipment, or livestock positions from aerial imagery for precision agriculture workflows
- **Security and Surveillance**: Create geospatial awareness from aerial camera feeds, mapping detected activity to real-world coordinates for situational awareness
- **Wildlife Conservation**: Track and map animal detections from drone surveys to monitor populations, migration patterns, or habitat usage
- **Construction Site Monitoring**: Map equipment, materials, and personnel positions from aerial imagery for site management and safety compliance
- **Search and Rescue**: Rapidly map detected persons or objects across large areas from drone footage to coordinate response efforts

## Connecting to Other Blocks

This block receives detections and produces geospatial data:

- **After object detection blocks** (YOLO, RF-DETR, etc.) to geotag their predictions with real-world coordinates
- **After tracking blocks** (ByteTrack, OC-SORT) to produce geotagged tracks with movement paths
- **Before data sink blocks** (CSV, JSON, Webhook) to export detection locations for GIS analysis
- **Before visualization blocks** to annotate frames with GPS coordinate labels
- **In video processing pipelines** where each frame's GPS comes from drone telemetry or EXIF metadata


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/geotag_detection@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `latitude` | `float` | GPS latitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are North, negative are South.. | ✅ |
| `longitude` | `float` | GPS longitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are East, negative are West.. | ✅ |
| `altitude` | `float` | Camera altitude above ground level in meters. Used with field of view to compute the ground footprint. For drones, this is the relative altitude reported by the flight controller, not absolute altitude.. | ✅ |
| `horizontal_fov` | `float` | Horizontal field of view of the camera in degrees. Default 73.7 covers most DJI consumer drones (Mini, Air, Mavic series). Adjust for other cameras. Wider FOV = larger ground footprint per frame.. | ✅ |
| `heading` | `float` | Compass bearing that the top of the image points toward, in degrees clockwise from true north. 0 means image-up is North (the default). When the gimbal does not report yaw, derive this from the flight course (bearing between successive GPS fixes) for nose-forward flight. Rotates the ground footprint so detections land on the correct real-world bearing instead of being pinned to North.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `GeoTag Detection` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Trace Visualization`](trace_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Filter`](detections_filter.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`Stitch Images`](stitch_images.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Cosine Similarity`](cosine_similarity.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`EasyOCR`](easy_ocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Gaze Detection`](gaze_detection.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Image Slicer`](image_slicer.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PP-OCR`](ppocr.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Blur`](image_blur.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Image Contours`](image_contours.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Path Deviation`](path_deviation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Text Display`](text_display.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Identify Changes`](identify_changes.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md)
    - outputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Frame Delay`](frame_delay.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PLC Reader`](plc_reader.md), [`Clip Comparison`](clip_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Clip Comparison`](clip_comparison.md), [`Polygon Visualization`](polygon_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Grid Visualization`](grid_visualization.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenRouter`](open_router.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Line Counter`](line_counter.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`GeoTag Detection` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): The image that detections were generated from. Used to determine image dimensions for coordinate projection..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Object detection predictions to geotag. Each detection's bounding box center will be projected to a GPS coordinate..
        - `latitude` (*[`float`](../kinds/float.md)*): GPS latitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are North, negative are South..
        - `longitude` (*[`float`](../kinds/float.md)*): GPS longitude of the camera position in decimal degrees. For drone imagery, this comes from the flight controller GPS. Positive values are East, negative are West..
        - `altitude` (*[`float`](../kinds/float.md)*): Camera altitude above ground level in meters. Used with field of view to compute the ground footprint. For drones, this is the relative altitude reported by the flight controller, not absolute altitude..
        - `horizontal_fov` (*[`float`](../kinds/float.md)*): Horizontal field of view of the camera in degrees. Default 73.7 covers most DJI consumer drones (Mini, Air, Mavic series). Adjust for other cameras. Wider FOV = larger ground footprint per frame..
        - `heading` (*[`float`](../kinds/float.md)*): Compass bearing that the top of the image points toward, in degrees clockwise from true north. 0 means image-up is North (the default). When the gimbal does not report yaw, derive this from the flight course (bearing between successive GPS fixes) for nose-forward flight. Rotates the ground footprint so detections land on the correct real-world bearing instead of being pinned to North..

    - output
    
        - `geo_detections` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.
        - `geojson` ([`dictionary`](../kinds/dictionary.md)): Dictionary.



??? tip "Example JSON definition of step `GeoTag Detection` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/geotag_detection@v1",
	    "image": "$inputs.image",
	    "predictions": "$steps.detection.predictions",
	    "latitude": 47.428681,
	    "longitude": -105.279125,
	    "altitude": 69.0,
	    "horizontal_fov": 73.7,
	    "heading": 0.0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

