
# Barcode Detection



??? "Class: `BarcodeDetectorBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/third_party/barcode_detection/v1.py">inference.core.workflows.core_steps.models.third_party.barcode_detection.v1.BarcodeDetectorBlockV1</a>
    



Detect the location of barcodes in an image.

This block is useful for manufacturing and consumer packaged goods projects where you 
need to detect a barcode region in an image. You can then apply Crop block to isolate 
each barcode then apply further processing (i.e. OCR of the characters on a barcode).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/barcode_detector@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Barcode Detection` in version `v1`.

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Calibration`](camera_calibration.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Threshold`](image_threshold.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Triangle Visualization`](triangle_visualization.md), [`Color Visualization`](color_visualization.md), [`Text Display`](text_display.md), [`Image Preprocessing`](image_preprocessing.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Visualization`](mask_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Barcode Detection` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..

    - output
    
        - `predictions` ([`bar_code_detection`](../kinds/bar_code_detection.md)): Prediction with barcode detection.



??? tip "Example JSON definition of step `Barcode Detection` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/barcode_detector@v1",
	    "images": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

