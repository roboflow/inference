# Understanding Model Packages Trained on Roboflow Platform

Models trained on the Roboflow Platform follow a custom RF standard where runtime behavior is determined by `inference_config.json`.

## Overview

When you train a model on Roboflow Platform, the exported model package includes an `inference_config.json` file that controls:

- **Image preprocessing** - Auto-orientation, static cropping, contrast adjustments, grayscale conversion
- **Network input** - Resize modes, padding, color modes, normalization
- **Forward pass** - Batch size configuration
- **Post-processing** - NMS parameters, activation functions

This configuration is automatically generated during training based on your dataset settings and training parameters.

## Package Structure

A typical Roboflow Platform model package contains:

```
model_package/
├── model_config.json       # Auto-generated metadata
├── weights.pt              # Model weights (or .onnx, .engine)
├── class_names.txt         # Class labels
└── inference_config.json   # Runtime behavior configuration
```

## `inference_config.json` Structure

### Complete Example

```json
{
  "image_pre_processing": {
    "auto-orient": {
      "enabled": true
    },
    "static-crop": {
      "enabled": true,
      "x_min": 10,
      "y_min": 10,
      "x_max": 630,
      "y_max": 470
    },
    "contrast": {
      "enabled": false,
      "type": "Adaptive Equalization"
    },
    "grayscale": {
      "enabled": false
    }
  },
  "network_input": {
    "training_input_size": {
      "height": 640,
      "width": 640
    },
    "dynamic_spatial_size_supported": true,
    "dynamic_spatial_size_mode": {
      "type": "divisible-padding",
      "divisor": 32
    },
    "color_mode": "rgb",
    "resize_mode": "letterbox",
    "padding_value": 114,
    "input_channels": 3,
    "scaling_factor": 255.0,
    "normalization": [
      [0.485, 0.456, 0.406],
      [0.229, 0.224, 0.225]
    ]
  },
  "forward_pass": {
    "static_batch_size": null,
    "max_dynamic_batch_size": 8
  },
  "post_processing": {
    "type": "nms",
    "fused": false,
    "nms_parameters": {
      "max_detections": 300,
      "confidence_threshold": 0.25,
      "iou_threshold": 0.45,
      "class_agnostic": 0
    }
  },
  "class_names_operations": [
    {
      "type": "class_name_removal",
      "class_name": "background"
    }
  ]
}
```

### Configuration Sections

#### 1. Image Pre-Processing

Controls transformations applied before network input preparation:

**`auto-orient`** - Automatically correct image orientation based on EXIF data
```json
{
  "enabled": true
}
```

**`static-crop`** - Apply fixed crop to all images
```json
{
  "enabled": true,
  "x_min": 10,
  "y_min": 10,
  "x_max": 630,
  "y_max": 470
}
```

**`contrast`** - Apply contrast enhancement
```json
{
  "enabled": true,
  "type": "Adaptive Equalization"  // or "Contrast Stretching", "Histogram Equalization"
}
```

**`grayscale`** - Convert images to grayscale
```json
{
  "enabled": true
}
```

#### 2. Network Input

Defines how images are prepared for the neural network:

**`training_input_size`** - Size the model was trained on
```json
{
  "height": 640,
  "width": 640
}
```

**`resize_mode`** - How to resize images to match training size
- `"letterbox"` - Maintain aspect ratio with padding
- `"stretch"` - Stretch to fit (may distort)
- `"center-crop"` - Crop center region

**`dynamic_spatial_size_supported`** - Whether model supports variable input sizes

**`dynamic_spatial_size_mode`** - How to handle dynamic sizes
```json
{
  "type": "divisible-padding",
  "divisor": 32
}
```

#### 3. Forward Pass

Controls batch processing:

```json
{
  "static_batch_size": null,
  "max_dynamic_batch_size": 8
}
```

#### 4. Post-Processing

**NMS (Object Detection)**
```json
{
  "type": "nms",
  "fused": false,
  "nms_parameters": {
    "max_detections": 300,
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "class_agnostic": 0
  }
}
```

## How It Works

When you load a Roboflow Platform model:

1. Model package is downloaded and cached
2. `inference_config.json` is parsed and validated
3. Preprocessing pipeline is configured automatically
4. Model is initialized with appropriate backend
5. Post-processing is configured based on settings

## Next Steps

- [Load Models Locally](local-packages.md)
- [Cache Management](../getting-started/cache.md)
- [Supported Models](../models/index.md)

