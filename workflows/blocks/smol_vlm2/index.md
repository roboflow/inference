
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SmolVLM2` in version `v1`.

    - inputs: [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stitch Images`](stitch_images.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Image Threshold`](image_threshold.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Depth Estimation`](depth_estimation.md), [`Relative Static Crop`](relative_static_crop.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Text Display`](text_display.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Stability AI Outpainting`](stability_ai_outpainting.md)
    - outputs: [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`SAM 3`](sam3.md)

    
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

