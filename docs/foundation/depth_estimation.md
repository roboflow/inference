<a href="https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf" target="_blank">Depth-Anything-V2-Small</a> is a depth estimation model developed by Hugging Face.

You can use Depth-Anything-V2-Small to estimate the depth of objects in images, creating a depth map where:
- Each pixel's value represents its relative distance from the camera
- Lower values (darker colors) indicate closer objects
- Higher values (lighter colors) indicate further objects

You can deploy Depth-Anything-V2-Small with Inference.

### Execution Modes

Depth Estimation supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server (GPU recommended for faster inference)
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server using the `depth_estimation()` client method

### Installation

To install inference with the extra dependencies necessary to run Depth-Anything-V2-Small, run

```pip install inference[transformers]```

or

```pip install inference-gpu[transformers]```

### How to Use Depth-Anything-V2-Small

Create a new Python file called `app.py` and add the following code:

```python
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from inference.models.depth_estimation.depthestimation import DepthEstimator

# Initialize the model
model = DepthEstimator()

# Load an image
image = Image.open("your_image.jpg")

# Run inference
results = model.predict(image)

# Get the depth map and visualization
depth_map = results[0]['normalized_depth']
visualization = results[0]['image']

# Convert visualization to numpy array for display
visualization_array = visualization.numpy()

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(visualization_array)
plt.title('Depth Map')
plt.axis('off')

plt.show()
```

In this code, we:
1. Load the Depth-Anything-V2-Small model
2. Load an image for depth estimation
3. Run inference to get the depth map
4. Display both the original image and the depth map visualization

The depth map visualization uses a viridis colormap where:
- Darker colors (purple/blue) represent objects closer to the camera
- Lighter colors (yellow/green) represent objects further from the camera

To use Depth-Anything-V2-Small with Inference, you will need a Hugging Face token. If you don't already have a Hugging Face account, <a href="https://huggingface.co/join" target="_blank">sign up for a free Hugging Face account</a>.

Then, set your Hugging Face token as an environment variable:

```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

Or you can log in using the Hugging Face CLI:

```bash
huggingface-cli login
```

Then, run the Python script you have created:

```bash
python app.py
```

The script will display both the original image and the depth map visualization.
