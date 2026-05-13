
# QR Code Detection



??? "Class: `QRCodeDetectorBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/third_party/qr_code_detection/v1.py">inference.core.workflows.core_steps.models.third_party.qr_code_detection.v1.QRCodeDetectorBlockV1</a>
    



Detect the location of a QR code.

This block is useful for manufacturing and consumer packaged goods projects where you 
need to detect a QR code region in an image. You can then apply Crop block to isolate 
each QR code then apply further processing (i.e. read a QR code with a custom block).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/qr_code_detector@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `QR Code Detection` in version `v1`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Color Visualization`](color_visualization.md), [`Label Visualization`](label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT`](sift.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Depth Estimation`](depth_estimation.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`QR Code Detection` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..

    - output
    
        - `predictions` ([`qr_code_detection`](../kinds/qr_code_detection.md)): Prediction with QR code detection.



??? tip "Example JSON definition of step `QR Code Detection` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/qr_code_detector@v1",
	    "images": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

