
# SAM 3 Interactive



??? "Class: `SegmentAnything3InteractiveBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/segment_anything3_interactive/v1.py">inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1.SegmentAnything3InteractiveBlockV1</a>
    



Run the interactive (promptable visual segmentation) head of Segment Anything 3 (SAM3) on an image.

Unlike the SAM 3 concept segmentation block (which takes text or exemplar prompts and returns
ALL instances of a concept), this block performs SAM2-style interactive segmentation: each prompt
targets ONE object and the model returns a single mask for it.

Two prompt inputs are supported (at least one must be provided):
- **points**: a list of labeled 2D points defining a single object. Positive points mark the
  object to segment, negative points mark regions to exclude (useful to refine the mask).
- **boxes**: detections from another model. Each bounding box becomes a separate prompt and
  the model segments the object inside it. Class names of the boxes are forwarded to the
  predicted masks.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/sam3_interactive@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `points` | `List[Any]` | Labeled points defining a single object to segment. Each point is {'x': ..., 'y': ..., 'positive': ...} in absolute pixel coordinates - positive points mark the object, negative points mark regions to exclude. Plain (x, y) or (x, y, positive) sequences are also accepted.. | ✅ |
| `threshold` | `float` | Minimum confidence threshold for predicted masks. | ✅ |
| `multimask_output` | `bool` | Flag to determine whether to use SAM3 internal multimask or single mask mode. For ambiguous prompts (like a single point) setting to True is recommended.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM 3 Interactive` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Cosine Similarity`](cosine_similarity.md), [`EasyOCR`](easy_ocr.md), [`Track Class Lock`](track_class_lock.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Byte Tracker`](byte_tracker.md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PP-OCR`](ppocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Camera Focus`](camera_focus.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Identify Outliers`](identify_outliers.md), [`Bounding Rectangle`](bounding_rectangle.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Event Writer`](event_writer.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Dot Visualization`](dot_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Combine`](detections_combine.md), [`Image Slicer`](image_slicer.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`PLC Writer`](plc_writer.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Blur`](image_blur.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Grid Visualization`](grid_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SIFT`](sift.md), [`Text Display`](text_display.md), [`JSON Parser`](json_parser.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Filter`](detections_filter.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Velocity`](velocity.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Detections Transformation`](detections_transformation.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Camera Focus`](camera_focus.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Byte Tracker`](byte_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM 3 Interactive` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `points` (*[`labeled_points`](../kinds/labeled_points.md)*): Labeled points defining a single object to segment. Each point is {'x': ..., 'y': ..., 'positive': ...} in absolute pixel coordinates - positive points mark the object, negative points mark regions to exclude. Plain (x, y) or (x, y, positive) sequences are also accepted..
        - `boxes` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Bounding boxes (from another model) to use as prompts - the model segments the object inside each box.
        - `threshold` (*[`float`](../kinds/float.md)*): Minimum confidence threshold for predicted masks.
        - `multimask_output` (*[`boolean`](../kinds/boolean.md)*): Flag to determine whether to use SAM3 internal multimask or single mask mode. For ambiguous prompts (like a single point) setting to True is recommended..

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `SAM 3 Interactive` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/sam3_interactive@v1",
	    "images": "$inputs.image",
	    "points": [
	        {
	            "positive": true,
	            "x": 320,
	            "y": 240
	        }
	    ],
	    "boxes": "$steps.object_detection_model.predictions",
	    "threshold": 0.3,
	    "multimask_output": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

