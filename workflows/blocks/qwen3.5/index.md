
# Qwen3.5



??? "Class: `Qwen35VLBlockV2`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/qwen3_5vl/v2.py">inference.core.workflows.core_steps.models.foundation.qwen3_5vl.v2.Qwen35VLBlockV2</a>
    


This workflow block runs Qwen3.5—a vision language model that accepts an image and an optional text prompt—and returns a text answer based on a conversation template.

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/qwen3_5vl@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `prompt` | `str` | Optional text prompt to provide additional context to Qwen3.5. Otherwise it will just be a default one, which may affect the desired model behavior.. | ❌ |
| `model_version` | `str` | The Qwen3.5 model to be used for inference.. | ✅ |
| `system_prompt` | `str` | Optional system prompt to provide additional context to Qwen3.5.. | ❌ |
| `max_new_tokens` | `int` | Maximum number of tokens to generate. If not set, the model's default will be used.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Qwen3.5` in version `v2`.

    - inputs: [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Camera Focus`](camera_focus.md), [`QR Code Generator`](qr_code_generator.md), [`Color Visualization`](color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`PLC EthernetIP`](plc_ethernet_ip.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Qwen3.5` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `model_version` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): The Qwen3.5 model to be used for inference..

    - output
    
        - `parsed_output` ([`dictionary`](../kinds/dictionary.md)): Dictionary.



??? tip "Example JSON definition of step `Qwen3.5` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/qwen3_5vl@v2",
	    "images": "$inputs.image",
	    "prompt": "What is in this image?",
	    "model_version": "qwen3_5-0.8b",
	    "system_prompt": "You are a helpful assistant.",
	    "max_new_tokens": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

