<a href="https://github.com/facebookresearch/segment-anything-2" target="_blank">Segment Anything 2</a> is an open source image segmentation model.

You can use Segment Anything 2 to identify the precise location of objects in an image. This process can generate masks for objects in an image iteratively, by specifying points to be included or discluded from the segmentation mask.

## How To Use SAM2 Locally With Inference

We will follow along with the example located at `examples/sam2/sam2.py`.

We start with the following image,

![Input image](https://media.roboflow.com/inference/sam2/hand.png)

compute the most prominent mask,

![Most prominent mask](https://media.roboflow.com/inference/sam2/sam.png)

and negative prompt the wrist to obtain only the fist.

![Negative prompt](https://media.roboflow.com/inference/sam2/sam_negative_prompted.png)

### Running within docker
Build the dockerfile (make sure your cwd is at the root of inference) with
```
docker build -f docker/dockerfiles/Dockerfile.sam2 -t sam2 .
```

Start up an interactive terminal with
```
docker run -it --rm --entrypoint bash -v $(pwd)/scratch/:/app/scratch/ -v /tmp/cache/:/tmp/cache/ -v $(pwd)/inference/:/app/inference/ --gpus=all --net=host sam2
```
You can save files to `/app/scratch/` to use them on the host device.

Or, start a sam2 server with
```
docker run -it --rm -v /tmp/cache/:/tmp/cache/ -v $(pwd)/inference/:/app/inference/ --gpus=all --net=host sam2
```

and interact over http.

### Imports
Set up your api key, and install <a href="https://github.com/facebookresearch/segment-anything-2" target="_blank">Segment Anything 2</a>

!!! note

    There's <a href="https://github.com/facebookresearch/segment-anything-2/issues/48" target="_blank">currently a problem</a> with sam2 + flash attention on certain gpus, like the L4 or A100. Use the fix in the posted thread, or use the docker image we provide for sam2. 

```
import os

os.environ["API_KEY"] = "<YOUR-API-KEY>"
from inference.models.sam2 import SegmentAnything2
from inference.core.utils.postprocess import masks2poly
import supervision as sv
from PIL import Image
import numpy as np

image_path = "./examples/sam2/hand.png"
```
### Model Loading
Load the model with 
```
m = SegmentAnything2(model_id="sam2/hiera_large")
```

Other values for `model_id` are `"hiera_small", "hiera_large", "hiera_tiny", "hiera_b_plus"`.

### Compute the Most Prominent Mask

```
# call embed_image before segment_image to precompute embeddings
embedding, img_shape, id_ = m.embed_image(image_path)

# segments image using cached embedding if it exists, else computes it on the fly
raw_masks, raw_low_res_masks = m.segment_image(image_path)

# convert binary masks to polygons
raw_masks = raw_masks >= m.predictor.mask_threshold
poly_masks = masks2poly(raw_masks)
```
Note that you can embed the image as soon as you know you want to process it, and the embeddings are cached automatically for faster downstream processing.

The resulting mask will look like this:

![Most prominent mask](https://media.roboflow.com/inference/sam2/sam.png)

### Negative Prompt the Model
```
point = [250, 800]
# give a negative point (point_label 0) or a positive example (point_label 1)
# uses cached masks from prior call
raw_masks2, raw_low_res_masks2 = m.segment_image(
    image_path,
    point_coords=[point],
    point_labels=[0],
)

raw_masks2 = raw_masks2 >= m.predictor.mask_threshold
```
Here we tell the model that the cached mask should not include the wrist.

The resulting mask will look like this:

![Negative prompt](https://media.roboflow.com/inference/sam2/sam_negative_prompted.png)

### Annotate
Use <a href="https://github.com/roboflow/supervision" target="_blank">Supervision</a> to draw the results of the model.

```
image = np.array(Image.open(image_path).convert("RGB"))

mask_annotator = sv.MaskAnnotator()
dot_annotator = sv.DotAnnotator()

detections = sv.Detections(
    xyxy=np.array([[0, 0, 100, 100]]), mask=np.array([raw_masks])
)
detections.class_id = [i for i in range(len(detections))]
annotated_image = mask_annotator.annotate(image.copy(), detections)
im = Image.fromarray(annotated_image)
im.save("sam.png")

detections = sv.Detections(
    xyxy=np.array([[0, 0, 100, 100]]), mask=np.array([raw_masks2])
)
detections.class_id = [i for i in range(len(detections))]
annotated_image = mask_annotator.annotate(image.copy(), detections)

dot_detections = sv.Detections(
    xyxy=np.array([[point[0] - 1, point[1] - 1, point[0] + 1, point[1] + 1]]),
    class_id=np.array([1]),
)
annotated_image = dot_annotator.annotate(annotated_image, dot_detections)
im = Image.fromarray(annotated_image)
im.save("sam_negative_prompted.png")
```