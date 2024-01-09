[CLIP](https://github.com/openai/CLIP) is a computer vision model that can measure the similarlity between text and images.

CLIP can be used for, among other things:

- Image classification
- Automated labeling for classification models
- Image clustering
- Gathering images for model training that are sufficiently dissimilar from existing samples
- Content moderation

With Inference, you can calculate CLIP embeddings for images and text in real-time.

In this guide, we will show:

1. How to classify video frames with CLIP in real time, and;
2. How to calculate CLIP image and text embeddings for use in clustering and comparison.

## How can I use CLIP model in `inference`?
* directly from `inference[clip]` package, integrating the model directly into your code
* using `inference` HTTP API (hosted locally, or at Roboflow platform), integrating via HTTP protocol
  * using `inference-sdk` package (`pip install inference-sdk`) and [`InferenceHTTPClient`](/docs/inference_sdk/http_client.md)
  * creating custom code to make HTTP requests (see [API Reference](https://inference.roboflow.com/api/))

## Classify Video Frames

With CLIP, you can classify images and video frames without training a model. This is because CLIP has been pre-trained to recognize many different objects.

To use CLIP to classify video frames, you need a prompt. In the example below, we will use the prompt "cell phone".

We can compare the similarity of "cell phone" to each video frame and use that to classify the video frame.

Below is a demo of CLIP classifying video frames in real time. The code for the example is below the video.

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/clip-coffee.mp4" type="video/mp4">
</video>

First, install the Inference CLIP extension:

```
pip install inference[clip]
```

Then, create a new Python file and add the following code:

```python
import cv2
import inference
from inference.core.utils.postprocess import cosine_similarity

from inference.models import Clip
clip = Clip()

prompt = "an ace of spades playing card"
text_embedding = clip.embed_text(prompt)

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
    cv2.imshow("CLIP", image)
    cv2.waitKey(1)

# start the stream
inference.Stream(
    source="webcam",
    model=clip,

    output_channel_order="BGR",
    use_main_thread=True,

    on_prediction=render
)
```

Run the code to use CLIP on your webcam.

**Note:** The model will take a minute or two to load. You will not see output while the model is loading.

## Calculate a CLIP Embedding

CLIP enables you to calculate embeddings. Embeddings are numeric, semantic representations of images and text. They are useful for clustering and comparison.

You can use CLIP embeddings to compare the similarity of text and images.

There are two types of CLIP embeddings: image and text.

Below we show how to calculate, then compare, both types of embeddings.

### Image Embedding

!!! tip
    
    In this example, we assume `inference-sdk` package installed
    ```
    pip install inference-sdk
    ```

In the code below, we calculate an image embedding.

Create a new Python file and add this code:

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)
embeddings = CLIENT.get_clip_image_embeddings(inference_input="https://i.imgur.com/Q6lDy8B.jpg")
print(embeddings)
```

### Text Embedding

In the code below, we calculate a text embedding.

!!! tip
    
    In this example, we assume `inference-sdk` package installed
    ```
    pip install inference-sdk
    ```

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

embeddings = CLIENT.get_clip_text_embeddings(text="the quick brown fox jumped over the lazy dog")
print(embeddings)
```

### Compare Embeddings

To compare embeddings for similarity, you can use cosine similarity.

The code you need to compare image and text embeddings is the same.

!!! tip
    
    In this example, we assume `inference-sdk` package installed
    ```
    pip install inference-sdk
    ```

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://infer.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
)

result = CLIENT.clip_compare(
  subject="./image.jpg",
  prompt=["dog", "cat"] 
)
print(result)
```

The resulting number will be between 0 and 1. The higher the number, the more similar the image and text are.

## See Also

- [What is CLIP?](https://blog.roboflow.com/openai-clip/)
- [Build an Image Search Engine with CLIP and Faiss](https://blog.roboflow.com/clip-image-search-faiss/)
- [Build a Photo Memories App with CLIP](https://blog.roboflow.com/build-a-photo-memories-app-with-clip/)
- [Analyze and Classify Video with CLIP](https://blog.roboflow.com/how-to-analyze-and-classify-video-with-clip/)
