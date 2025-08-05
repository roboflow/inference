# SAM3 Refactoring Status

## Summary

We have successfully refactored the SAM3 model to separate the stateless model (`Sam3ImageModel`) from the stateful session management (`Sam3Session`). The refactoring maintains the core functionality while providing a cleaner API.

## Current Status

### ✅ Working Features

1. **Text Prompt Detection** - Successfully detects multiple objects with text prompts
   - Outputs match original implementation (99.99% mask accuracy)
   - Minor differences in probabilities (< 0.5%) are within acceptable tolerance

2. **Basic Architecture** - Core model components are working correctly
   - Image encoding with proper feature extraction
   - Text encoding with correct tensor shapes
   - Geometry encoder for prompts
   - Transformer encoder/decoder pipeline
   - Query initialization for multi-object detection (200 queries)

3. **Testing Framework** - Comprehensive comparison tests implemented
   - Side-by-side comparison with original implementation
   - Detailed output comparison with tolerance checking

### ⚠️ Issues to Fix

1. **Box Prompt (Visual Prompt)** 
   - Getting different number of detected objects (3 vs 2)
   - The visual prompt logic needs to match the original's behavior
   - Issue: How the first box is treated as a visual prompt vs geometric prompt

2. **Combined Prompts (Text + Box Refinement)**
   - Getting different number of objects (5 vs 6)
   - Likely related to the visual prompt issue above

3. **Multi-mask Output Mode**
   - Not implemented correctly yet
   - Original returns multiple masks, refactored only returns one
   - Need to understand how original handles `multimask_output` parameter

## Key Fixes Applied

1. **Text Feature Shape Fix**
   ```python
   # Fixed text encoding to transpose from (batch, seq, dim) to (seq, batch, dim)
   text_outputs["language_features"] = lang_feat.permute(1, 0, 2)
   ```

2. **Query Initialization Fix**
   ```python
   # Use all 200 decoder queries for text/box prompts (not just instance queries)
   query_embed = self.transformer.decoder.query_embed.weight
   ```

3. **Prompt Concatenation Order**
   ```python
   # Correct order: [text, geometric, visual]
   prompt_list = [text_features["language_features"], geo_feats, visual_prompt_embed]
   ```

## Next Steps

1. Debug the visual prompt handling to understand why box prompts produce different results
2. Implement proper multimask output support
3. Add support for point prompts (instance tracking)
4. Consider adding video support in the future (as outlined in the README)

## Usage Example

```python
# Build the model
sam_model = build_sam3_model(
    bpe_path=bpe_path,
    checkpoint_path=checkpoint_path,
    device="cuda"
)

# Create a session
session = Sam3Session(sam_model)

# Set image and add prompts
session.set_image(image_np)
session.set_text_prompt("cars")
predictions = session.predict(output_prob_thresh=0.5)

# Results include multiple detected objects
print(f"Detected {len(predictions['out_probs'])} objects")
```