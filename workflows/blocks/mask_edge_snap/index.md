
# Mask Edge Snap



??? "Class: `MaskEdgeSnapBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/mask_edge_snap/v1.py">inference.core.workflows.core_steps.classical_cv.mask_edge_snap.v1.MaskEdgeSnapBlockV1</a>
    



Refine instance segmentation masks by snapping contour points to Sobel edges within a band around the predicted boundary. This block improves segmentation accuracy by adjusting mask edges to align with detected image features.

## How This Block Works

This block refines segmentation masks through a sophisticated multi-step pipeline:

1. **Edge Detection**: Computes Sobel gradient magnitudes from the input image to detect edges
2. **Adaptive Thresholding**: Uses per-pixel adaptive thresholding (local mean + sigma * local std) to identify significant edges
3. **Morphological Processing**: Applies closing (dilation + erosion) to bridge small gaps in edge segments
4. **Thinning**: Applies Zhang-Suen single-iteration thinning to reduce edge width to 1-2 pixels while preserving connectivity
5. **Boundary Band Creation**: Builds a search band around each predicted mask's contour
6. **Area Filtering**: Removes small edge components below a minimum area threshold
7. **Contour Snapping**: For each original mask contour point, finds the strongest nearby edge within tolerance and snaps to it

## Common Use Cases

- **Medical Image Analysis**: Refine organ/tumor segmentation masks to align with anatomical boundaries
- **Industrial Quality Control**: Improve part boundary detection for precise dimension measurement
- **Autonomous Vehicles**: Refine road/lane segmentation boundaries for improved path planning
- **Agricultural Monitoring**: Enhance crop boundary detection for yield estimation
- **Microscopy Analysis**: Refine cell/nuclei segmentation for morphological analysis
- **Document Processing**: Improve text region boundary detection for OCR

## Input Parameters

**image** : Input image (color or grayscale)
- Can be single-channel, 3-channel (BGR), or 4-channel (BGRA)
- Preprocessing (blur, contrast enhancement) should be applied upstream if needed

**segmentation** : Initial instance segmentation predictions
- Source: from object detection or instance segmentation model
- Must contain populated `mask` field; if empty, passed through unchanged

**pixel_tolerance** : Maximum perpendicular distance (pixels) for edge snapping
- Range: 5-50 typically
- 5-15: tight predictions with minimal offset
- 20-50: rough predictions needing more forgiveness

**sigma** : Strictness multiplier for adaptive Sobel threshold
- Range: 0.1-2.0 typically
- 0.1-0.5: permissive, keeps weaker edges, good for low-contrast boundaries
- 1.0-2.0: strict, only strongest edges survive, good for high-contrast images

**min_contour_area** : Minimum enclosed-polygon area for edge components
- Range: 10-1000 typically
- Small (10-50): keeps fragmented edges
- Large (200-1000): aggressive noise rejection

**dilation_iterations** : Number of morphological closing iterations
- Range: 0-10 typically
- 0: no closing, only thresholded edges
- 1-2: bridges hairline gaps
- 3-5: bridges visible dashes
- 10+: aggressive, can merge unrelated edges

**boundary_band_width** : Half-width of search band around mask contour (default: 15)
- Sets maximum distance between predicted and true boundary that can be corrected

**adaptive_window_size** : Side length of local-statistics window (default: 41)
- Should be roughly 5-10% of smaller image dimension
- Smaller (15-25): fine local contrast sensitivity, can pick up noise
- Larger (81-121): smooth threshold field, closer to global thresholding

## Outputs

**refined_segmentation** : Same detections with snapped mask contours
**edges** : Single detection containing union of all surviving edge pixels (debug/visualization)

## Preprocessing

**Preprocessing is usually critical for success.** This block does no preprocessing — what you feed in is what Sobel sees. For challenging imagery, chain Roboflow image-processing blocks upstream:

**Gaussian Blur**
    For grainy or noisy surfaces (welds, machined metal, biological tissue), blur before edge detection to suppress per-pixel noise. A 5x5 kernel with sigma 1.0 is a sensible default; increase to 7x7 or 9x9 for very noisy imagery. Don't over-blur — strong blur rounds off corners and softens real boundaries, leading to boundary positions that are biased inward.

**Bilateral Blur**
    Better than Gaussian when the image has both noise AND important sharp edges (e.g. textured fabric on a clean background). Slower, but preserves edges while denoising flat regions.

**Contrast Enhancement**
    Use when boundary contrast is genuinely too low to threshold reliably. The Contrast Enhancement block normalizes the histogram to use the full range, improving edge detection sensitivity without the noise amplification of aggressive methods. Follow with blur to suppress any remaining noise. Avoid on already-high-contrast images.

**Morphological Opening then Closing**
    Opening (erode then dilate) removes small bright specks and thin protrusions from the input before edge detection — useful when the surface has fine debris or hot pixels that would otherwise generate spurious edges. Closing (dilate then erode) fills small dark holes/gaps in bright regions; less commonly needed as preprocessing, since gap filling on the edge map itself is what the `dilation_iterations` parameter already does. Use the Morphological Transformation v2 block with the "Opening then Closing" operation for this preprocessing.

**Order matters**: Blur first, then contrast adjustment if needed. Reverse causes contrast adjustment to amplify the noise before blur can suppress it.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/mask_edge_snap@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `pixel_tolerance` | `int` | Maximum perpendicular distance (pixels) from each contour point to candidate edges during snapping. Typical: 5-15 for tight predictions, 20-50 for rough ones. Too small: real edges outside range get missed. Too large: snap can wander to unrelated edges.. | ✅ |
| `sigma` | `float` | Strictness multiplier for adaptive Sobel threshold (local_mean + sigma * local_std). Lower (0.1-0.5): permissive, good for low-contrast. Higher (1.0-2.0): strict, only strongest edges. Tune this AFTER other parameters.. | ✅ |
| `min_contour_area` | `float` | Minimum enclosed-polygon area for edge components to keep. Small (10-50): keeps fragmented edges. Large (200-1000): aggressive noise rejection. Scales roughly with dilation_iterations.. | ✅ |
| `dilation_iterations` | `int` | Morphological closing iterations to bridge gaps in thresholded edge map. Each iteration bridges ~2px gaps. 0: no closing. 1-2: hairline gaps. 3-5: visible dashes. 10+: aggressive merging.. | ✅ |
| `boundary_band_width` | `int` | Half-width (pixels) of search band around segmentation contour. Sets maximum distance between predicted boundary and true boundary that can be corrected. Should generally be >= pixel_tolerance.. | ✅ |
| `adaptive_window_size` | `int` | Side length of local-statistics window for adaptive threshold. Small (15-25): fine local sensitivity, can pick noise. Default 41: balanced. Large (81-121): smooth field, closer to global thresholding. Should be ~5-10% of smaller image dimension.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Mask Edge Snap` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Seg Preview`](seg_preview.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Detections Combine`](detections_combine.md), [`Identify Changes`](identify_changes.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Line Counter`](line_counter.md), [`Dynamic Crop`](dynamic_crop.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Camera Focus`](camera_focus.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Detections Stitch`](detections_stitch.md), [`Relative Static Crop`](relative_static_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Velocity`](velocity.md), [`Detection Event Log`](detection_event_log.md), [`Distance Measurement`](distance_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter`](line_counter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Cosine Similarity`](cosine_similarity.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Gaze Detection`](gaze_detection.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md)
    - outputs: [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Detections Combine`](detections_combine.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Overlap Filter`](overlap_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Detection Event Log`](detection_event_log.md), [`Velocity`](velocity.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Line Counter`](line_counter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`Detections Consensus`](detections_consensus.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Mask Edge Snap` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image (color or grayscale) for edge detection and snapping. Can be grayscale, single-channel, BGR, or BGRA. No preprocessing is applied internally; use upstream blocks for blur or contrast enhancement if needed..
        - `segmentation` (*[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)*): Instance segmentation predictions with mask field populated. Each mask contour will be snapped to detected edges. If empty, segmentation is passed through unchanged. Can be a reference string like '$steps.segmentation_model.predictions' or a supervision.Detections object..
        - `pixel_tolerance` (*[`integer`](../kinds/integer.md)*): Maximum perpendicular distance (pixels) from each contour point to candidate edges during snapping. Typical: 5-15 for tight predictions, 20-50 for rough ones. Too small: real edges outside range get missed. Too large: snap can wander to unrelated edges..
        - `sigma` (*[`float`](../kinds/float.md)*): Strictness multiplier for adaptive Sobel threshold (local_mean + sigma * local_std). Lower (0.1-0.5): permissive, good for low-contrast. Higher (1.0-2.0): strict, only strongest edges. Tune this AFTER other parameters..
        - `min_contour_area` (*[`float`](../kinds/float.md)*): Minimum enclosed-polygon area for edge components to keep. Small (10-50): keeps fragmented edges. Large (200-1000): aggressive noise rejection. Scales roughly with dilation_iterations..
        - `dilation_iterations` (*[`integer`](../kinds/integer.md)*): Morphological closing iterations to bridge gaps in thresholded edge map. Each iteration bridges ~2px gaps. 0: no closing. 1-2: hairline gaps. 3-5: visible dashes. 10+: aggressive merging..
        - `boundary_band_width` (*[`integer`](../kinds/integer.md)*): Half-width (pixels) of search band around segmentation contour. Sets maximum distance between predicted boundary and true boundary that can be corrected. Should generally be >= pixel_tolerance..
        - `adaptive_window_size` (*[`integer`](../kinds/integer.md)*): Side length of local-statistics window for adaptive threshold. Small (15-25): fine local sensitivity, can pick noise. Default 41: balanced. Large (81-121): smooth field, closer to global thresholding. Should be ~5-10% of smaller image dimension..

    - output
    
        - `refined_segmentation` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.
        - `edges` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Mask Edge Snap` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/mask_edge_snap@v1",
	    "image": "$inputs.image",
	    "segmentation": "$steps.segmentation_model.predictions",
	    "pixel_tolerance": "<block_does_not_provide_example>",
	    "sigma": "<block_does_not_provide_example>",
	    "min_contour_area": "<block_does_not_provide_example>",
	    "dilation_iterations": "<block_does_not_provide_example>",
	    "boundary_band_width": "<block_does_not_provide_example>",
	    "adaptive_window_size": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

