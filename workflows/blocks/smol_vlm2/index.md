
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

    - inputs: [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Stitch Images`](stitch_images.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Contrast Equalization`](contrast_equalization.md), [`Relative Static Crop`](relative_static_crop.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Grid Visualization`](grid_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dot Visualization`](dot_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Threshold`](image_threshold.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md)
    - outputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md)

    
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

