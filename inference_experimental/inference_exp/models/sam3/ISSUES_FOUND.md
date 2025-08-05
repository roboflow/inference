# SAM3 Refactoring Issues Found

This document outlines the critical issues found in the SAM3 refactoring implementation compared to the original `sam3_demo.py`.

## Critical Issues

### 1. **Text Feature Format Mismatch**

**Problem:**
The `encode_text` method returns raw backbone output, but the model expects specific tensor shapes and keys.

**Current Implementation:**
```python
def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
    return self.backbone.forward_text([text], device=self.device)
```

**Expected Format:**
- Key: `"language_features"` with shape `(seq_len, batch, dim)`
- Key: `"language_mask"` with shape `(batch, seq_len)`
- The backbone might return shape `(batch, seq_len, dim)` which needs transposition

**Fix:**
```python
@torch.inference_mode()
def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
    text_outputs = self.backbone.forward_text([text], device=self.device)
    # Transpose if needed to get (seq_len, batch, dim)
    if text_outputs["language_features"].shape[0] != text_outputs["language_features"].shape[1]:
        text_outputs["language_features"] = text_outputs["language_features"].permute(1, 0, 2)
    return text_outputs
```

### 2. **Missing Visual Prompt Encoding Through Geometry Encoder**

**Problem:**
Visual prompts should be encoded using the geometry encoder (same as geometric prompts), not handled separately.

**Original Flow:**
```python
# In sam3_demo.py's add_prompt method:
visual_prompt_embed, visual_prompt_mask, _backbone_out = self._encode_prompt(
    backbone_out=inference_state["backbone_out"],
    find_input=inference_state["input_batch"].find_inputs[frame_idx],
    geometric_prompt=new_visual_prompt,
    encode_text=False,  # Important: no text encoding for visual prompt
)
```

**Current Implementation Error:**
The refactored code tries to encode visual prompts directly without using the geometry encoder.

### 3. **Incorrect Prompt Concatenation Order**

**Problem:**
Prompts must be concatenated in the specific order: `[text, geometric, visual]`

**Current Implementation:**
```python
prompt_list = [geo_feats, visual_prompt_embed]
if text_features:
    prompt_list.insert(0, text_features["language_features"].permute(1,0,2))
```

**Correct Order:**
```python
if text_features:
    prompt_list = [text_features["language_features"], geo_feats, visual_prompt_embed]
else:
    prompt_list = [geo_feats, visual_prompt_embed]
```

### 4. **Missing Encoder Memory Structure**

**Problem:**
The transformer encoder returns a complex structure that the decoder requires.

**Missing Fields in Encoder Output:**
- `memory["level_start_index"]`
- `memory["spatial_shapes"]`
- `memory["valid_ratios"]`
- `memory["memory_text"]` (optional, for prompt after encoding)

**These are computed by the encoder and needed by the decoder.**

### 5. **Incomplete Instance Query Initialization**

**Problem:**
The `_init_instance_queries` method doesn't handle the `use_instance_query=False` case.

**Missing Code:**
```python
else:
    query_embed = self.transformer.decoder.query_embed.weight
    reference_boxes = self.transformer.decoder.reference_points.weight
```

### 6. **Image Feature Extraction Issues**

**Problem:**
The `_get_img_feats` method expects specific structure from backbone output.

**Missing Logic:**
- Proper handling of multi-scale features
- Extraction of position encodings for each scale
- Proper tensor reshaping from `(batch, channels, height, width)` to `(hw, batch, channels)`

### 7. **Decoder Expected Inputs**

**Problem:**
The decoder's forward method signature expects specific arguments that aren't being provided correctly.

**Missing/Incorrect Arguments:**
- `tgt_mask=None` (should be provided)
- `memory_text` expects prompt tensor
- `text_attention_mask` expects prompt_mask
- `apply_dac` parameter handling

## Root Cause Analysis

The fundamental issue is that the refactoring attempted to simplify the complex SAM3 architecture without preserving all the necessary data flow patterns. The original code has intricate dependencies between components that must be maintained for proper functionality.

## Recommended Approach

1. **Preserve Original Data Flow**: Keep the same data structures and tensor shapes throughout the pipeline
2. **Maintain Encoding Logic**: The `_encode_prompt` pattern from the original should be preserved
3. **Keep Transformer Structure**: The encoder/decoder communication protocol must be maintained exactly
4. **Test Incrementally**: Test each component (text encoding, image encoding, geometry encoding) separately before integration

## Testing Strategy

To debug effectively:
1. Add shape assertions after each encoding step
2. Print tensor shapes and dictionary keys at each stage
3. Compare outputs with the original implementation step-by-step
4. Use dummy inputs to isolate each component