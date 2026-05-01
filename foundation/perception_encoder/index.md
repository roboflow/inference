<a href="https://github.com/facebookresearch/perception_models?tab=readme-ov-file#perception-encoder-pe" target="_blank">Perception Encoder</a> is a computer vision model that can measure the similarity between text and images, as well as compute useful text and image embeddings.

Perception Encoder embeddings can be used for, among other things:

- Image classification
- Image clustering
- Gathering images for model training that are sufficiently dissimilar from existing samples
- Content moderation

With Inference, you can calculate PE embeddings for images and text in real-time.

In this guide, we will show:

1. How to classify video frames with PE in real time, and;
2. How to calculate PE image and text embeddings for use in clustering and comparison.

## How can I use PE in `inference`?

- directly from `inference[transformers]` package, integrating the model directly into your code
- using `inference` HTTP API (hosted locally, or on the Roboflow platform), integrating via HTTP protocol
  - using `inference-sdk` package (`pip install inference-sdk`) and [`InferenceHTTPClient`](../inference_helpers/inference_sdk.md)
  - creating custom code to make HTTP requests (see [API Reference](/api.md))

## Supported PE versions

- `perception_encoder/PE-Core-B16-224`
- `perception_encoder/PE-Core-L14-336`
- `perception_encoder/PE-Core-G14-448`

We currently only support the CLIP interface for PE models. We don't support the language or spatial aligned models yet.

## Classify Video Frames

With PE, you can classify images and video frames without training a model. This is because PE has been pre-trained to recognize many different objects.

To use PE to classify video frames, you need a prompt. In the example below, we will use the prompt "an image of a guy with a beard holding a can of sparkling water".

We can compare the similarity of the prompt to each video frame and use that to classify the video frame.

Below is a demo of PE classifying video frames in real time. The code for the example is below the image.

<img src="https://storage.googleapis.com/com-roboflow-marketing/inference/pe2.png" alt="Perception Encoder demo" width="800"/>


First, install the Inference transformers extension:

```
pip install "inference[transformers]"
```

Then, create a new Python file and add the following code:

```python
import cv2
import inference
from inference.core.utils.postprocess import cosine_similarity

from inference.models import PerceptionEncoder
pe = PerceptionEncoder(model_id="perception_encoder/PE-Core-B16-224", device="mps")  # `model_id` has default, but here is how to test other versions

prompt = "an image of a guy with a beard holding a can of sparkling water"
text_embedding = pe.embed_text(prompt)

def render(result, image):
    # get the cosine similarity between the prompt & the image
    similarity = cosine_similarity(result["embeddings"][0], text_embedding[0])

    # scale the result to 0-100 based on heuristic (~the best & worst values I've observed)
    range = (0.15, 0.40)
    similarity = (similarity-range[0])/(range[1]-range[0])
    similarity = max(min(similarity, 1), 0)*100

    # print the similarity
    text = f"{similarity:.1f}%"
    cv2.putText(image, text, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 12, (255, 255, 255), 30)
    cv2.putText(image, text, (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 12, (206, 6, 103), 16)

    # print the prompt
    cv2.putText(image, prompt, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10)
    cv2.putText(image, prompt, (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 2, (206, 6, 103), 5)

    # display the image
    cv2.imshow("PE", image)
    cv2.waitKey(1)

# start the stream
inference.Stream(
    source="webcam",
    model=pe,
    output_channel_order="BGR",
    use_main_thread=True,
    on_prediction=render
)
```

Run the code to use Perception Encoder on your webcam.

**Note:** The model will take a minute or two to load. You will not see output while the model is loading.

## Using PE in Workflows

Perception Encoder can be used in Roboflow Workflows via the
**Perception Encoder Embedding Model** block. This block lets you generate
embeddings for images or text without writing code.

## API Compatibility

The Perception Encoder model uses the **same API as CLIP**. This means you can use all the same methods and request/response formats as you would with CLIP, including `embed_text`, `embed_image`, and `compare`.

For more details and advanced usage, see the [CLIP documentation](./clip.md).

