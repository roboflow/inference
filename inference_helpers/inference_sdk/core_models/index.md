# Core Models

`InferenceHTTPClient` supports core models hosted via `inference`. Part of the models can be used on the Roboflow
hosted inference platform (use `https://infer.roboflow.com` as url), other are possible to be deployed locally (usually
local server will be available under `http://localhost:9001`).

## Clip

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.get_clip_image_embeddings(inference_input="./my_image.jpg")  # single image request
CLIENT.get_clip_image_embeddings(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
CLIENT.get_clip_text_embeddings(text="some")  # single text request
CLIENT.get_clip_text_embeddings(text=["some", "other"])  # other text request
CLIENT.clip_compare(
    subject="./my_image.jpg",
    prompt=["fox", "dog"],
)
```

`CLIENT.clip_compare(...)` method allows to compare different combination of `subject_type` and `prompt_type`:

- `(image, image)` (default)
- `(image, text)`
- `(text, image)`
- `(text, text)`

Async methods are also available:

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
    api_key="ROBOFLOW_API_KEY"
)

async def see_async_method():
  await CLIENT.get_clip_image_embeddings_async(inference_input="./my_image.jpg")  # single image request
  await CLIENT.get_clip_image_embeddings_async(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
  await CLIENT.get_clip_text_embeddings_async(text="some")  # single text request
  await CLIENT.get_clip_text_embeddings_async(text=["some", "other"])  # other text request
  await CLIENT.clip_compare_async(
      subject="./my_image.jpg",
      prompt=["fox", "dog"],
  )
```

## DocTR

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or "https://infer.roboflow.com" to use hosted serving
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.ocr_image(inference_input="./my_image.jpg")  # single image request
CLIENT.ocr_image(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
```

Async equivalent: `CLIENT.ocr_image_async(...)`

## Gaze

```python
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # only local hosting supported
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.detect_gazes(inference_input="./my_image.jpg")  # single image request
CLIENT.detect_gazes(inference_input=["./my_image.jpg", "./other_image.jpg"])  # batch image request
```

Async equivalent: `CLIENT.detect_gazes_async(...)`
