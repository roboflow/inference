# SAM3 Refactoring: Implementation Plan

This document outlines the plan to refactor the SAM3 interactive segmentation logic into a stateless `Sam3ImageModel` and a stateful `Sam3Session` controller.

## 1. Objective

The goal is to decouple the core model's computational graph from the application-specific logic of managing an interactive session. This will result in a cleaner, more modular, testable, and reusable codebase.

-   **`Sam3ImageModel`**: A pure `torch.nn.Module` responsible only for tensor-in, tensor-out computations.
-   **`Sam3Session`**: A controller class that manages the state of an interactive session.

## 2. File Structure

-   `inference-private/inference_experimental/inference_exp/models/sam3/sam3_image_model.py`: Will contain the `Sam3ImageModel` class and a new model builder function.
-   `inference-private/inference_experimental/inference_exp/models/sam3/sam3_session.py`: Will contain the `Sam3Session` class.
-   `inference-private/inference_experimental/inference_exp/models/sam3/README.md`: Documentation for the new classes.

## 3. Implementation Steps

### Step 1: Implement `Sam3ImageModel` and its Builder

**File**: `sam3_image_model.py`

-   **Create a `build_sam3_model(...)` function**:
    -   This will be a self-contained builder that imports components from the `sam3` library and constructs the `Sam3ImageModel`.
-   **Create the `Sam3ImageModel(torch.nn.Module)` class**:
    -   `__init__(...)`: Standard initialization.
    -   `preprocess_image(self, image: Union[np.ndarray, torch.Tensor]) -> (torch.Tensor, tuple)`: Handles conversion, normalization, and batching of input images.
    -   `encode_image(...)`: Runs the vision backbone and returns image features.
    -   `encode_text(...)`: Encodes a text string.
    -   `encode_geometric_prompts(...)`: Encodes point and box coordinates into a `Prompt` object.
    -   `predict(self, ..., multimask_output: bool = False)`: The core forward pass. It will be stateless.
    -   `postprocess_outputs(self, ..., multimask_output: bool = False)`:
        -   **Updated**: Will contain logic from `_postprocess_out` in `sam3_image.py` to handle the `multimask_output` flag, returning either the single best mask or the full set of masks.

### Step 2: Implement `Sam3Session`

**File**: `sam3_session.py`

-   Create the `Sam3Session` class.

**Attributes and Methods to Implement:**

-   `__init__(self, model: Sam3ImageModel)`:
    -   Stores a reference to the model.
    -   Initializes state attributes: `self.image_features = None`, `self.has_predicted = False`, `self.prompts = {}`.
-   `set_image(self, image: Union[np.ndarray, torch.Tensor])`:
    -   Caches image features and original size.
    -   **Updated**: Resets `self.has_predicted = False`.
-   `set_text_prompt(...)`, `add_point_prompt(...)`, `add_box_prompt(...)`: Methods to manage raw prompt data in `self.prompts`.
-   `predict(self, output_prob_thresh: float = 0.5, multimask_output: bool = False) -> Dict[str, np.ndarray]`:
    -   **Updated**: This method will contain the "visual prompt" logic. It will check if `self.has_predicted` is `False` and if a box prompt exists. If so, it will treat the first box as a special visual prompt when calling the model.
    -   It will orchestrate calls to the model's `encode_*` and `predict` methods.
    -   It will pass the `multimask_output` flag to the model.
    -   After a successful prediction, it will set `self.has_predicted = True`.
-   `reset_prompts(self)`: Clears prompt-related state but not the image features.
