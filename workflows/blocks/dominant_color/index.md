
# Dominant Color



??? "Class: `DominantColorBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/dominant_color/v1.py">inference.core.workflows.core_steps.classical_cv.dominant_color.v1.DominantColorBlockV1</a>
    



Extract the most prevalent dominant color from an image using K-means clustering on pixel colors, analyzing color distribution, identifying color clusters, and returning the RGB value of the most frequently occurring color cluster for color analysis, theme extraction, color-based filtering, and visual analysis workflows.

## How This Block Works

This block analyzes the color distribution in an image using K-means clustering to identify and extract the dominant (most prevalent) color. The block:

1. Receives an input image (assumes RGB format) to analyze for dominant color
2. Downsamples the image to optimize processing speed:
   - Calculates a scale factor based on the smallest dimension and target_size parameter
   - Reduces image resolution while preserving color characteristics
   - Smaller target_size values speed up processing but may slightly reduce precision
3. Reshapes the image pixels into a 2D array where each row represents one pixel's RGB values
4. Initializes K-means clustering:
   - Selects random pixel colors as initial cluster centroids (number of clusters = color_clusters)
   - Creates initial color clusters to group similar pixel colors
5. Performs iterative K-means clustering (up to max_iterations):
   - Assigns each pixel to the nearest color cluster (based on Euclidean distance in RGB color space)
   - Updates cluster centroids by computing the mean RGB values of pixels in each cluster
   - Handles empty clusters by reinitializing them to random pixel colors
   - Checks for convergence (centroids stop changing significantly) and exits early if converged
6. Counts pixels in each color cluster to determine which cluster contains the most pixels
7. Selects the cluster with the highest pixel count as the dominant color cluster
8. Extracts the RGB values from the dominant cluster's centroid
9. Converts and clips RGB values to valid 0-255 integer range
10. Returns the dominant color as an RGB tuple (R, G, B values)

The block uses K-means clustering to group similar pixel colors together, then identifies the largest color group as the dominant color. Processing time depends on image size, color complexity, and parameter settings (color_clusters, max_iterations, target_size). Most images complete in under half a second with default settings. The downsampling step balances speed and accuracy - reducing resolution speeds up clustering while still capturing the overall color distribution.

## Common Use Cases

- **Color Theme Extraction**: Extract dominant colors from images to identify color themes or palettes (e.g., extract dominant colors from product images, identify color themes in photographs, analyze color palettes in images), enabling color theme analysis workflows
- **Image Color Analysis**: Analyze images to determine their primary color characteristics (e.g., identify dominant colors in images, analyze color distribution, extract color signatures from images), enabling color-based image analysis
- **Color-Based Filtering**: Use dominant colors for filtering or categorizing images (e.g., filter images by dominant color, categorize images by color themes, group images by color characteristics), enabling color-based classification workflows
- **Visual Analysis and Reporting**: Extract color information for visual analysis or reporting (e.g., generate color reports for images, analyze color trends in image collections, extract color metadata for image databases), enabling color reporting workflows
- **Design and Branding Analysis**: Analyze images for design or branding purposes (e.g., extract brand colors from images, analyze design color schemes, identify color usage in branded content), enabling design analysis workflows
- **Quality Control**: Use dominant color analysis for quality control or inspection (e.g., verify expected colors in products, detect color anomalies, validate color characteristics), enabling color-based quality control workflows

## Connecting to Other Blocks

This block receives an image and produces a dominant RGB color:

- **After image input blocks** to extract dominant colors from input images (e.g., analyze colors in camera feeds, extract colors from image inputs, analyze color characteristics of images), enabling color analysis workflows
- **After crop blocks** to analyze dominant colors in specific image regions (e.g., extract dominant colors from cropped regions, analyze colors in specific areas, identify colors in selected regions), enabling region-based color analysis workflows
- **Before filtering or logic blocks** that use color information for decision-making (e.g., filter based on dominant colors, make decisions based on color characteristics, apply logic based on color analysis), enabling color-based conditional workflows
- **Before visualization blocks** to use dominant colors for visualization (e.g., use extracted colors for annotations, apply dominant colors to visualizations, customize visualizations with extracted colors), enabling color-enhanced visualization workflows
- **In color analysis pipelines** where dominant color extraction is part of a larger analysis workflow (e.g., analyze colors in multi-stage workflows, extract colors for comprehensive analysis, process color information in pipelines), enabling color analysis pipeline workflows
- **Before data storage blocks** to store color information along with images (e.g., store dominant colors with image metadata, save color analysis results, record color characteristics), enabling color metadata storage workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/dominant_color@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `color_clusters` | `int` | Number of color clusters (K) to use in K-means clustering. Must be between 1 and 10. The algorithm groups pixel colors into this many clusters, then selects the largest cluster as the dominant color. Higher values (e.g., 6-8) can improve precision for images with complex color distributions but increase processing time. Lower values (e.g., 2-3) are faster but may be less precise for multi-color images. Default is 4, which provides a good balance. Use fewer clusters for images with simple color schemes, more clusters for images with varied colors.. | ✅ |
| `max_iterations` | `int` | Maximum number of K-means clustering iterations to perform. Must be between 1 and 500. The algorithm iteratively refines color clusters and stops early if convergence is reached (centroids stop changing). Higher values allow more refinement and can improve precision but increase processing time. Lower values are faster but may result in less refined color clusters. Default is 100, which is typically sufficient for convergence. Most images converge well before reaching the maximum. Increase if you need more precise clustering, decrease if speed is critical.. | ✅ |
| `target_size` | `int` | Target size in pixels for the smallest dimension of the downsampled image used for clustering. Must be between 1 and 250. The image is downsampled before clustering to speed up processing - the smallest dimension is resized to approximately this size while maintaining aspect ratio. Lower values (e.g., 50-75) speed up processing significantly but may slightly reduce precision for images with fine color details. Higher values (e.g., 150-200) preserve more detail but are slower. Default is 100 pixels, which provides a good balance. Use lower values for speed-critical applications, higher values for maximum precision.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Dominant Color` in version `v1`.

    - inputs: [`Icon Visualization`](icon_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Contours`](image_contours.md), [`Image Slicer`](image_slicer.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Image Preprocessing`](image_preprocessing.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Relative Static Crop`](relative_static_crop.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SIFT`](sift.md), [`Label Visualization`](label_visualization.md), [`Image Threshold`](image_threshold.md), [`Corner Visualization`](corner_visualization.md), [`Template Matching`](template_matching.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Pixel Color Count`](pixel_color_count.md), [`Dynamic Crop`](dynamic_crop.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Dominant Color` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image to analyze for dominant color extraction. The image is assumed to be in RGB format. The block analyzes all pixels in the image to determine the most prevalent color. Processing time depends on image size and complexity. The image is automatically downsampled during processing to optimize speed while preserving color characteristics..
        - `color_clusters` (*[`integer`](../kinds/integer.md)*): Number of color clusters (K) to use in K-means clustering. Must be between 1 and 10. The algorithm groups pixel colors into this many clusters, then selects the largest cluster as the dominant color. Higher values (e.g., 6-8) can improve precision for images with complex color distributions but increase processing time. Lower values (e.g., 2-3) are faster but may be less precise for multi-color images. Default is 4, which provides a good balance. Use fewer clusters for images with simple color schemes, more clusters for images with varied colors..
        - `max_iterations` (*[`integer`](../kinds/integer.md)*): Maximum number of K-means clustering iterations to perform. Must be between 1 and 500. The algorithm iteratively refines color clusters and stops early if convergence is reached (centroids stop changing). Higher values allow more refinement and can improve precision but increase processing time. Lower values are faster but may result in less refined color clusters. Default is 100, which is typically sufficient for convergence. Most images converge well before reaching the maximum. Increase if you need more precise clustering, decrease if speed is critical..
        - `target_size` (*[`integer`](../kinds/integer.md)*): Target size in pixels for the smallest dimension of the downsampled image used for clustering. Must be between 1 and 250. The image is downsampled before clustering to speed up processing - the smallest dimension is resized to approximately this size while maintaining aspect ratio. Lower values (e.g., 50-75) speed up processing significantly but may slightly reduce precision for images with fine color details. Higher values (e.g., 150-200) preserve more detail but are slower. Default is 100 pixels, which provides a good balance. Use lower values for speed-critical applications, higher values for maximum precision..

    - output
    
        - `rgb_color` ([`rgb_color`](../kinds/rgb_color.md)): RGB color.



??? tip "Example JSON definition of step `Dominant Color` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/dominant_color@v1",
	    "image": "$inputs.image",
	    "color_clusters": 4,
	    "max_iterations": 100,
	    "target_size": 100
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

