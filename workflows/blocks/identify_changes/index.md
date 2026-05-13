
# Identify Changes



??? "Class: `IdentifyChangesBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sampling/identify_changes/v1.py">inference.core.workflows.core_steps.sampling.identify_changes.v1.IdentifyChangesBlockV1</a>
    



Identify changes and detect when data patterns change at unusual rates compared to historical norms by tracking embedding vectors over time, measuring cosine similarity changes, computing rate-of-change statistics, and flagging anomalies when changes occur faster or slower than expected for change detection, anomaly monitoring, rate-of-change analysis, and temporal pattern detection workflows.

## How This Block Works

This block detects changes by monitoring how quickly embeddings change over time and comparing the current rate of change against historical patterns. The block:

1. Receives an embedding vector representing the current data point's features
2. Normalizes the embedding to unit length:
   - Converts the embedding to a unit vector (length = 1) for cosine similarity calculations
   - Enables comparison using angular similarity rather than distance-based metrics
   - Handles zero vectors gracefully by skipping normalization
3. Tracks sample count and warmup status:
   - Increments sample counter for each processed embedding
   - Determines if still in warmup period (samples < warmup parameter)
   - During warmup, no outliers are identified to allow baseline establishment
4. Maintains running statistics for embedding averages and standard deviations using one of three strategies:

   **For Exponential Moving Average (EMA) strategy:**
   - Updates average and variance using exponential moving average with smoothing_factor
   - More recent embeddings have greater weight (controlled by smoothing_factor)
   - Adapts quickly to recent trends while maintaining historical context
   - Smoothing factor determines responsiveness (higher = more responsive to recent changes)

   **For Simple Moving Average (SMA) strategy:**
   - Uses Welford's method to calculate running mean and variance
   - All historical samples contribute equally to statistics
   - Provides stable, unbiased estimates over time
   - Well-suited for consistent, long-term tracking

   **For Sliding Window strategy:**
   - Maintains a fixed-size window of recent embeddings (window_size)
   - Removes oldest embeddings when window exceeds size (FIFO)
   - Calculates mean and standard deviation from window contents only
   - Adapts quickly to recent trends, discarding older information

5. Calculates cosine similarity between current embedding and running average:
   - Measures how similar the current embedding is to the typical embedding pattern
   - Cosine similarity ranges from -1 (opposite) to 1 (identical)
   - Values close to 1 indicate the embedding is similar to the norm
   - Values further from 1 indicate the embedding differs from the norm
6. Tracks rate of change by monitoring cosine similarity statistics:
   - Maintains running average and standard deviation of cosine similarity values
   - Uses the same strategy (EMA, SMA, or Sliding Window) for cosine similarity tracking
   - Measures how quickly embeddings are changing compared to historical change rates
   - Tracks both the average change rate and variability in change rates
7. Calculates z-score for current cosine similarity:
   - Measures how many standard deviations the current cosine similarity is from the average
   - Z-score = (current_cosine_similarity - average_cosine_similarity) / std_cosine_similarity
   - Positive z-scores indicate faster-than-normal changes
   - Negative z-scores indicate slower-than-normal changes
8. Converts z-score to percentile:
   - Uses error function (erf) to convert z-score to percentile position
   - Percentile represents where the current change rate ranks among historical rates
   - Values near 0.0 indicate unusually slow changes (low percentiles)
   - Values near 1.0 indicate unusually fast changes (high percentiles)
9. Determines outlier status based on percentile thresholds:
   - Flags as outlier if percentile is below threshold_percentile/2 (unusually slow changes)
   - Flags as outlier if percentile is above (1 - threshold_percentile/2) (unusually fast changes)
   - Detects both abnormally fast and abnormally slow rates of change
10. Updates running statistics for next iteration:
    - Updates embedding average and standard deviation using selected strategy
    - Updates cosine similarity average and standard deviation using selected strategy
    - Maintains state across workflow executions
11. Returns six outputs:
    - **is_outlier**: Boolean flag indicating if the rate of change is anomalous
    - **percentile**: Float value (0.0-1.0) representing where the change rate ranks historically
    - **z_score**: Float value representing standard deviations from average change rate
    - **average**: Current average embedding vector (running average of historical embeddings)
    - **std**: Current standard deviation vector for embeddings (variability per dimension)
    - **warming_up**: Boolean flag indicating if still in warmup period

The block monitors the **rate of change** rather than just detecting outliers. It tracks how quickly embeddings are changing and compares this to historical change patterns. When embeddings change much faster or slower than they have in the past, the block flags this as anomalous. This makes it ideal for detecting sudden pattern shifts, unexpected changes in scenes, or unusual behavior patterns. The three strategies (EMA, SMA, Sliding Window) offer different trade-offs between responsiveness and stability.

## Common Use Cases

- **Change Detection**: Detect when scenes, environments, or patterns change unexpectedly (e.g., detect scene changes, identify sudden pattern shifts, flag unexpected environmental changes), enabling change detection workflows
- **Anomaly Monitoring**: Monitor for unusual changes in behavior or patterns (e.g., detect abnormal behavior changes, monitor unusual pattern variations, flag unexpected rate changes), enabling anomaly monitoring workflows
- **Rate-of-Change Analysis**: Analyze and detect unusual rates of change in data streams (e.g., detect unusually fast changes, identify unusually slow changes, monitor change rate patterns), enabling rate-of-change analysis workflows
- **Temporal Pattern Detection**: Identify when temporal patterns deviate from expected change rates (e.g., detect pattern disruptions, identify timeline anomalies, flag temporal inconsistencies), enabling temporal pattern detection workflows
- **Quality Monitoring**: Monitor for unexpected changes in quality or characteristics (e.g., detect quality degradation, identify unexpected quality changes, monitor characteristic variations), enabling quality monitoring workflows
- **Event Detection**: Detect significant events based on unusual change rates (e.g., detect significant events, identify important changes, flag notable pattern shifts), enabling event detection workflows

## Connecting to Other Blocks

This block receives embeddings and produces is_outlier, percentile, z_score, average, std, and warming_up outputs:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to analyze change rates from embeddings (e.g., detect changes from CLIP embeddings, analyze Perception Encoder change rates, monitor embedding-based changes), enabling embedding-to-change workflows
- **After classification or detection blocks** with embeddings to identify unusual change patterns (e.g., detect unusual detection changes, flag anomalous classification changes, monitor prediction pattern changes), enabling prediction-to-change workflows
- **Before logic blocks** like Continue If to make decisions based on change detection (e.g., continue if change detected, filter based on change rate, trigger actions on unusual changes), enabling change-based decision workflows
- **Before notification blocks** to alert on change detection (e.g., alert on significant changes, notify about pattern shifts, trigger alerts on rate anomalies), enabling change-based notification workflows
- **Before data storage blocks** to record change information (e.g., log change data, store change statistics, record rate-of-change metrics), enabling change data logging workflows
- **In monitoring pipelines** where change detection is part of continuous monitoring (e.g., monitor changes in observation systems, track pattern variations, detect anomalies in monitoring workflows), enabling change monitoring workflows

## Requirements

This block requires embeddings as input (typically from embedding model blocks like CLIP or Perception Encoder). The block maintains internal state across workflow executions, tracking running statistics for both embeddings and cosine similarity values. During the warmup period (first `warmup` samples), no outliers are identified and the block returns is_outlier=False, percentile=0.5, and warming_up=True. After warmup, the block uses the selected strategy (EMA, SMA, or Sliding Window) to track statistics and detect rate-of-change anomalies. The threshold_percentile parameter (0.0-1.0) controls sensitivity - lower values detect only extreme rate changes, while higher values detect more moderate rate deviations. The strategy choice affects responsiveness: EMA adapts quickly to recent trends, SMA provides stable long-term tracking, and Sliding Window adapts quickly but discards older information. The block works best with consistent embedding models and may need adjustment of threshold_percentile and strategy based on expected variation and change patterns in your data.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/identify_changes@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |
| `strategy` | `str` | Statistical strategy for tracking embedding and change rate statistics. 'Exponential Moving Average (EMA)': Adapts quickly to recent trends, more weight on recent data. 'Simple Moving Average (SMA)': Stable long-term tracking, all data contributes equally. 'Sliding Window': Fast adaptation, uses only recent window_size samples. EMA is best for adaptive monitoring, SMA for stable tracking, Sliding Window for rapid adaptation.. | ❌ |
| `threshold_percentile` | `float` | Percentile threshold for change rate anomaly detection, range 0.0-1.0. Change rates below threshold_percentile/2 or above (1 - threshold_percentile/2) are flagged as outliers. Lower values (e.g., 0.05) detect only extreme rate changes - very strict. Higher values (e.g., 0.3) detect more moderate rate deviations - more sensitive. Default 0.2 means bottom 10% and top 10% of change rates are outliers. Adjust based on expected variation in change rates.. | ✅ |
| `warmup` | `int` | Number of initial data points required before change detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment for change rates. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay change detection. Lower values enable faster detection but may be less accurate initially.. | ✅ |
| `smoothing_factor` | `float` | Smoothing factor (alpha) for Exponential Moving Average strategy, range 0.0-1.0. Controls responsiveness to recent data - higher values make statistics more responsive to recent changes, lower values maintain more historical context. Example: 0.1 means 10% weight on current value, 90% on historical average. Typical range: 0.05-0.3. Only used when strategy is 'Exponential Moving Average (EMA)'.. | ✅ |
| `window_size` | `int` | Maximum number of recent embeddings to maintain in sliding window. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to changes. Smaller windows adapt faster but may be less stable. Only used when strategy is 'Sliding Window'. Must be at least 2. Typical range: 5-50 embeddings.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Identify Changes` in version `v1`.

    - inputs: [`Detection Event Log`](detection_event_log.md), [`Identify Outliers`](identify_outliers.md), [`Template Matching`](template_matching.md), [`Clip Comparison`](clip_comparison.md), [`Identify Changes`](identify_changes.md), [`Perspective Correction`](perspective_correction.md), [`Image Contours`](image_contours.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Pixel Color Count`](pixel_color_count.md), [`Distance Measurement`](distance_measurement.md), [`Line Counter`](line_counter.md), [`Line Counter`](line_counter.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Consensus`](detections_consensus.md)
    - outputs: [`Slack Notification`](slack_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Velocity`](velocity.md), [`Detection Event Log`](detection_event_log.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Color Visualization`](color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Cosine Similarity`](cosine_similarity.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Identify Outliers`](identify_outliers.md), [`Template Matching`](template_matching.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Continue If`](continue_if.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Gaze Detection`](gaze_detection.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Identify Changes` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `embedding` (*[`embedding`](../kinds/embedding.md)*): Embedding vector representing the current data point's features. Typically from embedding models like CLIP or Perception Encoder. The embedding is normalized to unit length for cosine similarity calculations. The block compares current embedding to running average and tracks rate of change over time..
        - `threshold_percentile` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Percentile threshold for change rate anomaly detection, range 0.0-1.0. Change rates below threshold_percentile/2 or above (1 - threshold_percentile/2) are flagged as outliers. Lower values (e.g., 0.05) detect only extreme rate changes - very strict. Higher values (e.g., 0.3) detect more moderate rate deviations - more sensitive. Default 0.2 means bottom 10% and top 10% of change rates are outliers. Adjust based on expected variation in change rates..
        - `warmup` (*[`integer`](../kinds/integer.md)*): Number of initial data points required before change detection begins. During warmup, no outliers are identified (is_outlier=False) to allow baseline establishment for change rates. Must be at least 2 for statistical analysis. Typical range: 3-100 samples. Higher values provide more stable baselines but delay change detection. Lower values enable faster detection but may be less accurate initially..
        - `smoothing_factor` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Smoothing factor (alpha) for Exponential Moving Average strategy, range 0.0-1.0. Controls responsiveness to recent data - higher values make statistics more responsive to recent changes, lower values maintain more historical context. Example: 0.1 means 10% weight on current value, 90% on historical average. Typical range: 0.05-0.3. Only used when strategy is 'Exponential Moving Average (EMA)'..
        - `window_size` (*[`integer`](../kinds/integer.md)*): Maximum number of recent embeddings to maintain in sliding window. When exceeded, oldest embeddings are removed (FIFO). Larger windows provide more stable statistics but adapt slower to changes. Smaller windows adapt faster but may be less stable. Only used when strategy is 'Sliding Window'. Must be at least 2. Typical range: 5-50 embeddings..

    - output
    
        - `is_outlier` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `percentile` ([`float_zero_to_one`](../kinds/float_zero_to_one.md)): `float` value in range `[0.0, 1.0]`.
        - `z_score` ([`float`](../kinds/float.md)): Float value.
        - `average` ([`embedding`](../kinds/embedding.md)): A list of floating point numbers representing a vector embedding..
        - `std` ([`embedding`](../kinds/embedding.md)): A list of floating point numbers representing a vector embedding..
        - `warming_up` ([`boolean`](../kinds/boolean.md)): Boolean flag.



??? tip "Example JSON definition of step `Identify Changes` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/identify_changes@v1",
	    "strategy": "Exponential Moving Average (EMA)",
	    "embedding": "$steps.clip.embedding",
	    "threshold_percentile": 0.2,
	    "warmup": 3,
	    "smoothing_factor": 0.1,
	    "window_size": 10
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

