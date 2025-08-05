# SAM3 Model and Session Guide

This document describes the refactored architecture for running interactive segmentation with SAM3. The logic is split into two main classes: `Sam3ImageModel` and `Sam3Session`.

---

## 1. Overview

The core principle is the **separation of concerns**:

-   **`Sam3ImageModel`**: A stateless PyTorch module that contains the neural network architecture.
-   **`Sam3Session`**: A stateful controller that manages the interactive workflow.

This design makes the model itself more reusable and the interactive logic easier to manage.

---

## 2. `Sam3ImageModel`

A stateless PyTorch module created by a dedicated `build_sam3_model` function.

### Key Responsibilities
- Preprocessing images (NumPy arrays or PyTorch tensors).
- Encoding an image, text, and geometric prompts into feature embeddings.
- Executing the main prediction pass.
- Post-processing raw model outputs into a user-friendly format.

---

## 3. `Sam3Session`

A stateful controller that provides a high-level API for an interactive segmentation session.

### Key Responsibilities
- Managing the state of a session, including the currently loaded image and its features.
- **Caching features** for efficiency. The image is encoded only once when `set_image` is called.
- **Handling Visual Prompt Logic**: It automatically treats the first box prompt on a new image as a special "visual prompt," mimicking the behavior of the original demo.
- Providing a simple interface that accepts either NumPy arrays or PyTorch tensors for images.

---

## 4. Usage Example

```python
import torch
import numpy as np
from PIL import Image

from .sam3_image_model import build_sam3_model, Sam3ImageModel
from .sam3_session import Sam3Session

# 1. Instantiate the core model
bpe_path = "path/to/bpe/vocab.bpe"
checkpoint_path = "path/to/sam3_model.pth"
sam_model: Sam3ImageModel = build_sam3_model(bpe_path=bpe_path, checkpoint_path=checkpoint_path)

# 2. Create an interactive session
session = Sam3Session(sam_model)

# 3. Set an image
image_np = np.array(Image.open("path/to/image.jpg").convert("RGB"))
session.set_image(image_np)

# 4. Add prompts and predict
# The first box prompt is automatically handled as a visual prompt.
session.add_box_prompt(boxes=[[100, 150, 250, 300]])
predictions = session.predict(output_prob_thresh=0.8)

# Subsequent prompts act as refinements
session.add_point_prompt(points=[[180, 220]], labels=[1])
refined_predictions = session.predict()

# 5. Requesting Multi-Mask Output
# Use the `multimask_output` flag to get multiple mask candidates.
multi_mask_predictions = session.predict(multimask_output=True)

# The output will contain additional keys like 'multi_pred_masks'.
print(multi_mask_predictions.keys())

```

---

## 5. Notes on Future Enhancements

The current refactoring is focused on creating a clean architecture for single-image interactive segmentation. The following capabilities from the original codebase can be added back in a similarly modular fashion in the future.

### a) Model Compilation

-   **Plan**: A `compile_model()` method could be added to the `Sam3ImageModel` class. This method would apply `torch.compile` to the most computationally intensive parts of the model.

### b) Video Support

-   **How it Works in the Original Code**: The original `Sam3Image` class has video tracking deeply integrated. It works via two main mechanisms:
    1.  **Temporal Query Propagation**: The key to tracking is that the decoder's input queries for the current frame `t` are the *output queries* from the previous frame `t-1`.
    2.  **Efficient Feature Caching**: The original code is designed to encode all frames of a video in a single batch and cache the features, avoiding redundant computation.

-   **Future Plan**: Video support would be implemented by creating a new **`Sam3VideoSession`** class, which would use the *same* `Sam3ImageModel`.
    -   **Architecture**:
        -   The `Sam3VideoSession` would manage the state for a sequence of frames.
        -   Its `set_video()` method would stack all video frames into a single batch tensor and make **one call** to `model.encode_image()` to efficiently compute all features at once. These features would be cached.
        -   It would implement a `propagate()` method to handle the frame-by-frame prediction loop, passing the output queries from frame `t` as the input queries to frame `t+1`. This would require a minor, non-breaking addition to `Sam3ImageModel.predict()` to optionally accept tracking queries as input.
    -   **Benefit**: This compositional approach avoids polluting the core model with video-specific logic and keeps both the single-image and video use cases clean and separate.
