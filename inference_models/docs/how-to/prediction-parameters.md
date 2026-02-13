# Prediction Method Parameters

This guide explains the standard parameter naming conventions used across `inference-models` for prediction methods. Understanding these conventions helps you use models consistently and predictably.

## Overview

The `inference-models` package follows a **gentleman's agreement** for parameter naming across all model implementations. While this is not enforced by the type system, maintainers actively work toward unification to provide a consistent developer experience.

!!! info "Parameter Applicability"
    Not all parameters apply to all models. Each parameter is relevant only for specific groups of models based on their architecture and task type.

## Standard Parameters

### Post-Processing Parameters

These parameters control how model outputs are filtered and refined.

#### `confidence`

**Type:** `float`
**Default:** Varies by model
**Applies to:** Object detection, instance segmentation, keypoint detection

Confidence threshold for filtering predictions. Only predictions with confidence scores above this threshold are returned.

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image, confidence=0.5)  # Only predictions with >50% confidence
```

#### `iou_threshold`

**Type:** `float`
**Default:** Varies by model
**Applies to:** Models requiring Non-Maximum Suppression (NMS)

Intersection-over-Union (IoU) threshold for NMS. This controls how much overlap is allowed between bounding boxes before they are considered duplicates. Lower values are more aggressive at removing overlapping boxes.

**How IoU works:**
- IoU measures the overlap between two bounding boxes as: `Area of Overlap / Area of Union`

- IoU = 0: No overlap

- IoU = 1: Perfect overlap

- During NMS, boxes with IoU > `iou_threshold` are considered duplicates, and only the highest-confidence box is kept

```python
# More aggressive NMS - removes more overlapping boxes
predictions = model(image, iou_threshold=0.3)

# Less aggressive NMS - keeps more overlapping boxes
predictions = model(image, iou_threshold=0.7)
```

#### `class_agnostic_nms`

**Type:** `bool`  
**Default:** `False`  
**Applies to:** Models requiring NMS

Flag to control whether NMS is performed across all classes or separately for each class.

- `False` (default): NMS is applied separately for each class. A box for "person" won't suppress a box for "car" even if they overlap significantly.
- `True`: NMS is applied across all classes. Overlapping boxes are suppressed regardless of their predicted class.

```python
# Class-specific NMS (default)
predictions = model(image, class_agnostic_nms=False)

# Class-agnostic NMS
predictions = model(image, class_agnostic_nms=True)
```

#### `max_detections`

**Type:** `int`

**Default:** Varies by model

**Applies to:** Object detection, instance segmentation, keypoint detection

Maximum number of top-scored detections to return. After NMS, only the top N highest-confidence detections are kept.

```python
# Return at most 50 detections
predictions = model(image, max_detections=50)
```

### Pre-Processing Parameters

These parameters control how input images are processed before inference.

#### `image_size`

**Type:** `Tuple[int, int]` (width, height)  
**Default:** Model-specific  
**Applies to:** Models that allow manual override of input dimensions

Manually override the input image dimensions. Only use this if the model supports dynamic input sizes.

```python
# Override input size to 1280x1280
predictions = model(image, image_size=(1280, 1280))
```

#### `input_color_format`

**Type:** `str` or `ColorFormat`  
**Default:** Model and input type specific  
**Applies to:** All models

Specifies the color format of input images. Use this parameter **only when your input differs from the standard format** for that input type.

**Standard formats:**

- `np.ndarray`: BGR (OpenCV default)

- `torch.Tensor`: RGB

```python
import cv2

# Non-standard: numpy array in RGB instead of BGR
rgb_image = cv2.imread("<your-image>")[:, :, ::-1]  # RGB format
predictions = model(rgb_image, input_color_format="rgb")
```

### Vision-Language Model (VLM) Parameters

These parameters control text generation in vision-language models.

#### `max_new_tokens`

**Type:** `int`

**Default:** Varies by model

**Applies to:** Vision-language models (Florence, PaliGemma, Qwen, SmolVLM, Moondream)

Maximum number of tokens to generate in the model's response. Controls the length of generated text.

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("florence-2-base")

# Short captions
captions = model.caption_image(image, max_new_tokens=100)

# Longer descriptions
captions = model.caption_image(image, max_new_tokens=1000)
```

#### `num_beams`

**Type:** `int`

**Default:** Varies by model

**Applies to:** Vision-language models

Number of beams for beam search during text generation. Beam search explores multiple possible sequences simultaneously to find higher-quality outputs.

- `num_beams=1`: Greedy decoding (fastest, but may miss better sequences)
- `num_beams>1`: Beam search (slower, but typically produces better results)

Higher values generally produce better quality text but are slower.

```python
# Greedy decoding (fastest)
captions = model.caption_image(image, num_beams=1)

# Beam search with 5 beams (better quality, slower)
captions = model.caption_image(image, num_beams=5)
```

#### `do_sample`

**Type:** `bool`  
**Default:** `False`  
**Applies to:** Vision-language models

Whether to use sampling instead of greedy/beam search decoding.

- `False` (default): Deterministic decoding (greedy or beam search)
- `True`: Stochastic sampling (introduces randomness for more diverse outputs)

When `True`, the model samples from the probability distribution over tokens rather than always picking the most likely token.

```python
# Deterministic output
captions = model.caption_image(image, do_sample=False)

# Stochastic sampling for diverse outputs
captions = model.caption_image(image, do_sample=True)
```

#### `skip_special_tokens`

**Type:** `bool`  
**Default:** `False`  
**Applies to:** Vision-language models

Whether to remove special tokens (like `<pad>`, `<eos>`, `<bos>`) from the generated text output.

- `False`: Keep special tokens in the output
- `True`: Remove special tokens for cleaner text

```python
# Keep special tokens
result = model.caption_image(image, skip_special_tokens=False)

# Clean output without special tokens
result = model.caption_image(image, skip_special_tokens=True)
```

## Setting Default Values via Environment Variables

You can set default values for many prediction parameters using environment variables. This is useful for configuring behavior globally without changing code.

The following environment variables are available:

- **General defaults**: `INFERENCE_MODELS_DEFAULT_CONFIDENCE`, `INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD`, `INFERENCE_MODELS_DEFAULT_MAX_DETECTIONS`, `INFERENCE_MODELS_DEFAULT_CLASS_AGNOSTIC_NMS`, `INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS`, `INFERENCE_MODELS_DEFAULT_NUM_BEAMS`, `INFERENCE_MODELS_DEFAULT_DO_SAMPLE`, `INFERENCE_MODELS_DEFAULT_SKIP_SPECIAL_TOKENS`

- **Model-specific overrides**: Each model can have its own defaults (e.g., `INFERENCE_MODELS_YOLOV8_DEFAULT_CONFIDENCE`, `INFERENCE_MODELS_FLORENCE2_DEFAULT_MAX_NEW_TOKENS`)

Example:

```bash
# Set global defaults
export INFERENCE_MODELS_DEFAULT_CONFIDENCE="0.5"
export INFERENCE_MODELS_DEFAULT_IOU_THRESHOLD="0.4"
export INFERENCE_MODELS_DEFAULT_MAX_NEW_TOKENS="1000"

# Override for specific model
export INFERENCE_MODELS_OWLV2_DEFAULT_CONFIDENCE="0.95"
```

See the [Environment Variables Configuration](environment-variables.md#prediction-parameter-defaults) guide for the complete list of available environment variables and their default values.


## Best Practices

### 1. Start with Defaults

Always start with default parameter values and adjust based on your specific use case:

```python
# Start here
predictions = model(image)

# Then tune if needed
predictions = model(image, confidence=0.6, iou_threshold=0.4)
```

### 2. Understand the Trade-offs

**Confidence threshold:**
- Higher → Fewer false positives, more false negatives
- Lower → More false positives, fewer false negatives

**IoU threshold:**
- Higher → More overlapping boxes kept
- Lower → More aggressive duplicate removal

**Max detections:**
- Higher → More results, slower post-processing
- Lower → Fewer results, faster post-processing

### 3. Model-Specific Tuning

Different model architectures may require different parameter values:

```python
# YOLO models typically work well with these
yolo_predictions = yolo_model(image, confidence=0.25, iou_threshold=0.45)

# Open-vocabulary models often need higher confidence
owlv2_predictions = owlv2_model(image, confidence=0.1, max_detections=300)

# VLMs benefit from beam search
captions = vlm_model.caption_image(image, num_beams=5, do_sample=False)
```

### 4. Batch Processing Consistency

When processing multiple images, use consistent parameters across all inputs:

```python
images = [cv2.imread(f"image_{i}.jpg") for i in range(10)]

# Consistent parameters across batch
for image in images:
    predictions = model(
        image,
        confidence=0.5,
        iou_threshold=0.4,
        max_detections=100
    )
```

## Troubleshooting

### Too Many Detections

**Problem:** Model returns too many overlapping boxes

**Solution:**
```python
# Increase confidence threshold
predictions = model(image, confidence=0.6)

# Or use more aggressive NMS
predictions = model(image, iou_threshold=0.3)

# Or limit max detections
predictions = model(image, max_detections=50)
```

### Missing Detections

**Problem:** Model misses objects you expect to detect

**Solution:**
```python
# Lower confidence threshold
predictions = model(image, confidence=0.1)

# Use less aggressive NMS
predictions = model(image, iou_threshold=0.7)

# Increase max detections
predictions = model(image, max_detections=300)
```

### Poor VLM Output Quality

**Problem:** Generated text is low quality or repetitive

**Solution:**
```python
# Increase beam search
captions = model.caption_image(image, num_beams=5)

# Or enable sampling for diversity
captions = model.caption_image(image, do_sample=True, num_beams=1)

# Adjust token limit
captions = model.caption_image(image, max_new_tokens=200)
```


## See Also

- [Environment Variables Configuration](environment-variables.md) - Set default parameter values
- [Work with Predictions](work-with-predictions.md) - Process model outputs
- [Choose Backend](choose-backend.md) - Backend-specific considerations
- [Supported Models](../models/index.md) - Model-specific parameter defaults

