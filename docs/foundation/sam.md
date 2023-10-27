[Segment Anything](https://github.com/facebookresearch/segment-anything) is an open source image segmentation model.

You can use Segment Anything to identify the precise location of objects in an image.

To use Segment Anything, you need to:

1. Create an embedding for an image, and;
2. Specify the coordinates of the object you want to segment.

## How to Use Segment Anything

To use Segment Anything with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

### Embed an Image

An embedding is a numeric representation of an image. SAM uses embeddings as input to calcualte the location of objects in an image.

Create a new Python file and add the following code:

```python
import requests

infer_payload = {
    "image": {
        "type": "base64",
        "value": "https://i.imgur.com/Q6lDy8B.jpg",
    },
    "image_id": "example_image_id",
}

base_url = "https://infer.roboflow.com"

# Define your Roboflow API Key
api_key = ""

res = requests.post(
    f"{base_url}/sam/embed_image?api_key={api_key}",
    json=infer_clip_payload,
)

embeddings = request.json()['embeddings']
```

This code makes a request to Inference to embed an image using SAM.

The `example_image_id` is used to cache the embeddings for later use so you don't have to send them back in future segmentation requests.

### Segment an Object

To segment an object, you need to know at least one point in the image that represents the object that you want to use.

!!! tip
    For testing with a single image, you can upload an image to the [Polygon Zone web interface](https://roboflow.github.io/polygonzone/) and hover over a point in the image to see the coordinates of that point.

You may also opt to use an object detection model to identify an object, then use the center point of the bounding box as a prompt for segmentation.

Create a new Python file and add the following code:

```python
#Define request payload
infer_payload = {
    "image": {
        "type": "base64",
        "value": "https://i.imgur.com/Q6lDy8B.jpg",
    },
    "point_coords": [[380, 350]],
    "point_labels": [1],
    "image_id": "example_image_id",
}

res = requests.post(
    f"{base_url}/sam/embed_image?api_key={api_key}",
    json=infer_clip_payload,
)

masks = request.json()['masks']
```

This request returns segmentation masks that represent the object of interest.

## See Also

- [What is Segment Anything Model (SAM)?](https://blog.roboflow.com/segment-anything-breakdown/)