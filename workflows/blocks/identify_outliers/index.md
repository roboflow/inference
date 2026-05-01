
# Identify Outliers



??? "Class: `IdentifyOutliersBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sampling/identify_outliers/v1.py">inference.core.workflows.core_steps.sampling.identify_outliers.v1.IdentifyOutliersBlockV1</a>
    



Identify outlier embeddings compared to prior data using von Mises-Fisher statistical distribution analysis to detect anomalies, unusual patterns, or deviations from normal behavior by comparing current embedding vectors against a sliding window of historical embeddings for quality control, anomaly detection, and data sampling workflows.

## How This Block Works

This block detects outliers by statistically comparing embedding vectors against historical data using directional statistics. The block:

1. Receives an embedding vector representing the current data point's features
2. Normalizes the embedding to unit length:
   - Converts the embedding to a unit vector (length = 1) for directional analysis
   - Enables comparison using angular/directional statistics rather than distance-based metrics
   - Handles zero vectors gracefully by skipping normalization
3. Tracks sample count and warmup status:
   - Increments sample counter for each processed embedding
   - Determines if still in warmup period (samples < warmup parameter)
   - During warmup, no outliers are identified to allow baseline establishment
4. Maintains a sliding window of historical embeddings:
   - Stores normalized embeddings in a buffer that grows up to window_size
   - When buffer exceeds window_size, removes oldest embeddings (FIFO)
   - Creates a rolling history of recent data for statistical comparison
5. Fits von Mises-Fisher (vMF) distribution parameters during warmup completion:
   - **Mean Direction (mu)**: Calculates the average direction of all historical embeddings
   - **Concentration Parameter (kappa)**: Measures how tightly clustered the embeddings are around the mean
   - Uses statistical estimation to model the distribution of embedding directions
   - vMF distribution is ideal for directional data on a hypersphere (unit vectors)
6. Computes alignment score for current embedding:
   - Calculates dot product between current normalized embedding and mean direction vector
   - Measures how well the current embedding aligns with the typical direction
   - Higher values indicate closer alignment to the norm, lower values indicate deviation
7. Calculates empirical percentile of current embedding:
   - Computes alignment scores for all historical embeddings against the mean direction
   - Ranks the current embedding's alignment score among historical scores
   - Determines percentile position (0.0 = lowest, 1.0 = highest) of current embedding
8. Determines outlier status based on percentile thresholds:
   - Flags as outlier if percentile is below threshold_percentile (e.g., bottom 5%)
   - Flags as outlier if percentile is above (1 - threshold_percentile) (e.g., top 5%)
   - Detects both extreme low and extreme high deviations from the norm
9. Returns three outputs:
   - **is_outlier**: Boolean flag indicating if the current embedding is an outlier
   - **percentile**: Float value (0.0-1.0) representing where the embedding ranks among historical data
   - **warming_up**: Boolean flag indicating if still in warmup period (always False after warmup)

The block uses von Mises-Fisher distribution analysis, which is designed for directional data on a hypersphere (unit vectors). This makes it well-suited for high-dimensional embeddings where direction matters more than magnitude. The sliding window approach ensures the statistical model adapts to recent trends while the percentile-based detection identifies embeddings that are unusually different from the historical pattern. Lower percentiles indicate embeddings that are less aligned with typical patterns, while higher percentiles indicate embeddings that are unusually well-aligned or different in a positive direction.

## Common Use Cases

- **Anomaly Detection**: Detect unusual images, objects, or patterns that deviate from normal data (e.g., identify unusual product variations, detect anomalous behavior, flag unexpected patterns), enabling anomaly detection workflows
- **Quality Control**: Identify defective or unusual items in manufacturing or production (e.g., detect product defects, identify quality issues, flag manufacturing anomalies), enabling quality control workflows
- **Data Sampling**: Identify interesting or unusual data points for manual review or further analysis (e.g., sample unusual images for labeling, identify edge cases for model improvement, select interesting data for analysis), enabling intelligent data sampling workflows
- **Change Detection**: Detect when data patterns change significantly from historical norms (e.g., detect scene changes, identify pattern shifts, flag significant variations), enabling change detection workflows
- **Model Monitoring**: Monitor model performance by detecting when embeddings deviate from training distribution (e.g., detect distribution shift, identify out-of-distribution data, monitor model drift), enabling model monitoring workflows
- **Content Filtering**: Identify unusual or inappropriate content that differs from expected patterns (e.g., detect unusual content, flag inappropriate material, identify content anomalies), enabling content filtering workflows

## Connecting to Other Blocks

This block receives embeddings and produces is_outlier, percentile, and warming_up outputs:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to analyze embedding outliers (e.g., identify outliers from CLIP embeddings, analyze Perception Encoder outliers, detect anomalies from embeddings), enabling embedding-to-outlier workflows
- **After classification or detection blocks** with embeddings to identify unusual predictions (e.g., identify unusual detections, flag anomalous classifications, detect outlier predictions), enabling prediction-to-outlier workflows
- **Before logic blocks** like Continue If to make decisions based on outlier detection (e.g., continue if outlier detected, filter based on outlier status, trigger actions on anomalies), enabling outlier-based decision workflows
- **Before notification blocks** to alert on outlier detection (e.g., alert on anomalies, notify about unusual data, trigger alerts on outliers), enabling outlier-based notification workflows
- **Before data storage blocks** to record outlier information (e.g., log outlier data, store anomaly statistics, record unusual data points), enabling outlier data logging workflows
- **In quality control pipelines** where outlier detection is part of quality assurance (e.g., filter outliers in quality pipelines, identify issues in production workflows, detect problems in processing chains), enabling quality control workflows

## Requirements

This block requires embeddings as input (typically from embedding model blocks like CLIP or Perception Encoder). The block maintains internal state across workflow executions, accumulating a sliding window of historical embeddings. During the warmup period (first `warmup` samples), no outliers are identified and the block returns is_outlier=False and percentile=0.5. After warmup, the block uses at least `warmup` embeddings (up to `window_size` embeddings) to establish statistical baselines. The threshold_percentile parameter (0.0-1.0) controls sensitivity - lower values (e.g., 0.01) detect only extreme outliers, while higher values (e.g., 0.1) detect more moderate deviations. The block works best with consistent embedding models and may need adjustment of threshold_percentile based on expected variation in your data.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/identify_outliers@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `threshold_percentile` | `float` | Percentile threshold for outlier detection, range 0.0-1.0. Embeddings below this percentile or above (1 - threshold_percentile) are flagged as outliers. Lower values (e.g., 0.01) detect only extreme outliers - very strict. Higher values (e.g., 0.1) detect more moderate deviations - more sensitive. Default 0.05 means bottom 5% and top 5% are outliers. Adjust based on expected variation in your data.. | ✅ |
| `warmup` | `int` | Number of initial data points required before outlier detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay outlier detection. Lower values enable faster detection but may be less accurate initially.. | ✅ |
| `window_size` | `int` | Maximum number of historical embeddings to maintain in sliding window. The block keeps the most recent window_size embeddings for statistical comparison. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to distribution changes. Smaller windows adapt faster but may be less stable. Set to None for unlimited window (uses all historical data). Typical range: 10-100 embeddings.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Identify Outliers` in version `v1`.

    - inputs: [`CLIP Embedding Model`](clip_embedding_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Contours`](image_contours.md), [`Template Matching`](template_matching.md), [`Identify Outliers`](identify_outliers.md), [`Line Counter`](line_counter.md), [`SIFT Comparison`](sift_comparison.md), [`Distance Measurement`](distance_measurement.md), [`Detection Event Log`](detection_event_log.md), [`Clip Comparison`](clip_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Perspective Correction`](perspective_correction.md), [`Line Counter`](line_counter.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`SIFT Comparison`](sift_comparison.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Identify Outliers` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `embedding` (*[`embedding`](../kinds/embedding.md)*): Embedding vector representing the current data point's features. Typically from embedding models like CLIP or Perception Encoder. The embedding is normalized to unit length for directional statistical analysis using von Mises-Fisher distribution. Must be a numerical vector of any dimension..
        - `threshold_percentile` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Percentile threshold for outlier detection, range 0.0-1.0. Embeddings below this percentile or above (1 - threshold_percentile) are flagged as outliers. Lower values (e.g., 0.01) detect only extreme outliers - very strict. Higher values (e.g., 0.1) detect more moderate deviations - more sensitive. Default 0.05 means bottom 5% and top 5% are outliers. Adjust based on expected variation in your data..
        - `warmup` (*[`integer`](../kinds/integer.md)*): Number of initial data points required before outlier detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay outlier detection. Lower values enable faster detection but may be less accurate initially..
        - `window_size` (*[`integer`](../kinds/integer.md)*): Maximum number of historical embeddings to maintain in sliding window. The block keeps the most recent window_size embeddings for statistical comparison. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to distribution changes. Smaller windows adapt faster but may be less stable. Set to None for unlimited window (uses all historical data). Typical range: 10-100 embeddings..

    - output
    
        - `is_outlier` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `percentile` ([`float_zero_to_one`](../kinds/float_zero_to_one.md)): `float` value in range `[0.0, 1.0]`.
        - `warming_up` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `Identify Outliers` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/identify_outliers@v1",
	    "embedding": "$steps.clip.embedding",
	    "threshold_percentile": 0.05,
	    "warmup": 3,
	    "window_size": 32
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

