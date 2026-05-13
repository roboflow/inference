
# Qwen2.5-VL



??? "Class: `Qwen25VLBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/qwen/v1.py">inference.core.workflows.core_steps.models.foundation.qwen.v1.Qwen25VLBlockV1</a>
    


This workflow block runs Qwen2.5-VL—a vision language model that accepts an image and an optional text prompt—and returns a text answer based on a conversation template.

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/qwen25vl@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Optional text prompt to provide additional context to Qwen2.5-VL. Otherwise it will just be a default one, which may affect the desired model behavior.. | ❌ |
| `model_version` | `str` | The Qwen2.5-VL model to be used for inference.. | ✅ |
| `system_prompt` | `str` | Optional system prompt to provide additional context to Qwen2.5-VL.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Qwen2.5-VL` in version `v1`.

    - inputs: [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Color Visualization`](color_visualization.md), [`Label Visualization`](label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Camera Focus`](camera_focus.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Grid Visualization`](grid_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`SIFT`](sift.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Depth Estimation`](depth_estimation.md)
    - outputs: [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Detections Consensus`](detections_consensus.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Qwen2.5-VL` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_version` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): The Qwen2.5-VL model to be used for inference..

    - output
    
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.



??? tip "Example JSON definition of step `Qwen2.5-VL` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/qwen25vl@v1",
	    "images": "$inputs.image",
	    "prompt": "What is in this image?",
	    "model_version": "qwen25-vl-7b",
	    "system_prompt": "You are a helpful assistant."
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

