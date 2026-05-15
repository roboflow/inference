
# SmolVLM2



??? "Class: `SmolVLM2BlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/smolvlm/v1.py">inference.core.workflows.core_steps.models.foundation.smolvlm.v1.SmolVLM2BlockV1</a>
    


This workflow block runs SmolVLM2, a multimodal vision-language model. You can ask questions about images -- including documents and photos -- and get answers in natural language.

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/smolvlm2@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Optional text prompt to provide additional context to SmolVLM2. Otherwise it will just be None. | ❌ |
| `model_version` | `str` | The SmolVLM2 model to be used for inference.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SmolVLM2` in version `v1`.

    - inputs: [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Contours`](image_contours.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Grid Visualization`](grid_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md)
    - outputs: [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Detections Consensus`](detections_consensus.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SmolVLM2` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_version` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): The SmolVLM2 model to be used for inference..

    - output
    
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.



??? tip "Example JSON definition of step `SmolVLM2` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/smolvlm2@v1",
	    "images": "$inputs.image",
	    "prompt": "What is in this image?",
	    "model_version": "smolvlm2/smolvlm-2.2b-instruct"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

