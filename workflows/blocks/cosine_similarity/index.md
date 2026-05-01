
# Cosine Similarity



??? "Class: `CosineSimilarityBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/math/cosine_similarity/v1.py">inference.core.workflows.core_steps.math.cosine_similarity.v1.CosineSimilarityBlockV1</a>
    



Calculate the cosine similarity between two embedding vectors by computing the cosine of the angle between them, measuring directional similarity regardless of magnitude to enable similarity comparison, semantic matching, embedding-based search, and similarity-based filtering workflows.

## How This Block Works

This block computes cosine similarity, a measure of similarity between two vectors based on the cosine of the angle between them. The block:

1. Receives two embedding vectors from workflow steps (e.g., from CLIP, Perception Encoder, or other embedding models)
2. Validates embedding dimensions:
   - Ensures both embeddings have the same dimensionality (same number of elements)
   - Raises an error if dimensions don't match
3. Computes cosine similarity:
   - Calculates the dot product of the two embedding vectors
   - Computes the L2 norm (magnitude) of each embedding vector
   - Divides the dot product by the product of the two norms: similarity = (a · b) / (||a|| × ||b||)
   - This measures the cosine of the angle between the vectors, indicating directional similarity
4. Returns similarity score:
   - Outputs a similarity value ranging from -1 to 1
   - Value of 1: Vectors point in the same direction (identical or proportional) - maximum similarity
   - Value of 0: Vectors are orthogonal (perpendicular) - no similarity
   - Value of -1: Vectors point in opposite directions - maximum dissimilarity
   - Greater values (closer to 1) indicate greater similarity

Cosine similarity is magnitude-invariant, meaning it measures similarity in direction rather than size. Two vectors that point in the same direction will have high cosine similarity even if they have different magnitudes. This makes it ideal for comparing embeddings where magnitude may vary but semantic meaning (direction) is what matters.

## Common Use Cases

- **Semantic Similarity Comparison**: Compare semantic similarity between images, text, or other data types using embeddings (e.g., compare image embeddings, match text to images, find similar content), enabling similarity comparison workflows
- **Embedding-Based Search**: Use similarity scores for embedding-based search and retrieval (e.g., find similar images, search by embedding similarity, retrieve similar content), enabling embedding search workflows
- **Cross-Modal Matching**: Match embeddings across different modalities (e.g., match images to text, find images matching text descriptions, match text to images), enabling cross-modal matching workflows
- **Similarity-Based Filtering**: Filter data based on similarity thresholds (e.g., filter similar items, find duplicates using similarity, identify near-duplicates), enabling similarity filtering workflows
- **Content Recommendation**: Use similarity scores for content recommendation and matching (e.g., recommend similar content, match related items, suggest similar products), enabling recommendation workflows
- **Quality Control and Validation**: Validate embeddings or compare embeddings for quality control (e.g., validate embedding quality, compare embeddings for consistency, check embedding similarity), enabling quality control workflows

## Connecting to Other Blocks

This block receives embeddings from embedding model blocks and produces similarity scores:

- **After embedding model blocks** (CLIP, Perception Encoder, etc.) to compare embeddings (e.g., compare image and text embeddings, compare multiple embeddings, compute similarity scores), enabling embedding-to-similarity workflows
- **Before logic blocks** like Continue If to use similarity scores in conditions (e.g., continue if similarity exceeds threshold, filter based on similarity, make decisions using similarity), enabling similarity-based decision workflows
- **Before filtering blocks** to filter based on similarity (e.g., filter by similarity threshold, remove low-similarity items, keep high-similarity matches), enabling similarity-to-filter workflows
- **Before data storage blocks** to store similarity scores (e.g., store similarity metrics, log similarity comparisons, save similarity results), enabling similarity storage workflows
- **Before notification blocks** to send similarity-based alerts (e.g., notify on high similarity matches, alert on similarity changes, send similarity reports), enabling similarity notification workflows
- **In workflow outputs** to provide similarity scores as final output (e.g., similarity comparison outputs, matching results, similarity metrics), enabling similarity output workflows

## Requirements

This block requires two embedding vectors with the same dimensionality (same number of elements). Embeddings can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. The embeddings are passed as lists of floats. The block computes cosine similarity using the dot product divided by the product of L2 norms, producing a similarity score between -1 and 1. Values closer to 1 indicate greater similarity, values closer to 0 indicate orthogonal vectors, and values closer to -1 indicate opposite directions.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/cosine_similarity@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Unique name of step in workflows. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Cosine Similarity` in version `v1`.

    - inputs: [`CLIP Embedding Model`](clip_embedding_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Detection Event Log`](detection_event_log.md), [`Text Display`](text_display.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Anthropic Claude`](anthropic_claude.md), [`Continue If`](continue_if.md), [`Velocity`](velocity.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Cosine Similarity` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `embedding_1` (*[`embedding`](../kinds/embedding.md)*): First embedding vector to compare. Must have the same dimensionality (same number of elements) as embedding_2. Can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. Embedding vectors are lists of floats representing high-dimensional feature representations..
        - `embedding_2` (*[`embedding`](../kinds/embedding.md)*): Second embedding vector to compare. Must have the same dimensionality (same number of elements) as embedding_1. Can be from any embedding model (CLIP, Perception Encoder, etc.) and can represent images, text, or other data types. Embedding vectors are lists of floats representing high-dimensional feature representations. The cosine similarity measures the similarity between embedding_1 and embedding_2..

    - output
    
        - `similarity` ([`float`](../kinds/float.md)): Float value.



??? tip "Example JSON definition of step `Cosine Similarity` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/cosine_similarity@v1",
	    "embedding_1": "$steps.clip_image.embedding",
	    "embedding_2": "$steps.clip_text.embedding"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    

