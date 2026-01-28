This model uses [SAM 3 (Segment Anything 3)](https://github.com/facebookresearch/sam3) from Meta.

# Instance Segmentation with Box Prompts

```python
import cv2 as cv
import numpy as np
import supervision as sv

from inference_models import AutoModel

model = AutoModel.from_pretrained("sam3")

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(color=sv.Color.BLACK)

img = cv.imread("image.png")
predictions = model.segment_images(
    images=img,
    boxes=[[(100, 200, 300, 400)]],  # xyxy format
)

masks = predictions[0].masks.cpu().numpy()
detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks,
)

annotated_frame = mask_annotator.annotate(scene=img, detections=detections)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

cv.imshow("", annotated_frame)
cv.waitKey(0)
```

# Instance Segmentation with Point Prompts

```python
import cv2 as cv
import numpy as np
import supervision as sv

from inference_models import AutoModel

model = AutoModel.from_pretrained("sam3")

mask_annotator = sv.MaskAnnotator()

img = cv.imread("image.png")
predictions = model.segment_images(
    images=img,
    point_coordinates=[[[250, 300]]],  # xy format
    point_labels=[[[1]]],  # 1 = foreground, 0 = background
)

masks = predictions[0].masks.cpu().numpy()
detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks,
)

annotated_frame = mask_annotator.annotate(scene=img, detections=detections)

cv.imshow("", annotated_frame)
cv.waitKey(0)
```

# Text-Prompted Segmentation

SAM3 supports text-based prompting to segment objects described in natural language.

```python
import cv2 as cv
import numpy as np
import supervision as sv

from inference_models import AutoModel

model = AutoModel.from_pretrained("sam3")

mask_annotator = sv.MaskAnnotator()

img = cv.imread("image.png")
results = model.segment_with_text(
    images=img,
    prompts=[
        {"text": "person"},
        {"text": "dog"},
    ],
)

# Process results for each prompt
for prompt_result in results[0]:
    masks = prompt_result["masks"]
    scores = prompt_result["scores"]

    if len(masks) > 0:
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks=masks),
            mask=masks,
        )
        img = mask_annotator.annotate(scene=img, detections=detections)

cv.imshow("", img)
cv.waitKey(0)
```

# Visual Prompting with Text

Combine bounding box prompts with text descriptions for more precise segmentation.

```python
import cv2 as cv
import numpy as np
import supervision as sv

from inference_models import AutoModel

model = AutoModel.from_pretrained("sam3")

mask_annotator = sv.MaskAnnotator()

img = cv.imread("image.png")
results = model.segment_with_text(
    images=img,
    prompts=[
        {
            "text": "shirt",
            "boxes": [[100, 150, 300, 400]],  # xyxy format
            "box_labels": [1],  # 1 = positive, 0 = negative
        },
    ],
)

masks = results[0][0]["masks"]
if len(masks) > 0:
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks,
    )
    img = mask_annotator.annotate(scene=img, detections=detections)

cv.imshow("", img)
cv.waitKey(0)
```

# Embeddings Caching

For interactive applications, you can cache image embeddings to speed up subsequent predictions.

```python
import cv2 as cv
from inference_models import AutoModel
from inference_models.models.sam3.cache import Sam3ImageEmbeddingsInMemoryCache

# Initialize cache
embeddings_cache = Sam3ImageEmbeddingsInMemoryCache.init(size_limit=10)

model = AutoModel.from_pretrained(
    "sam3",
    sam3_image_embeddings_cache=embeddings_cache,
)

img = cv.imread("image.png")

# First call computes and caches embeddings
predictions = model.segment_images(
    images=img,
    boxes=[[(100, 200, 300, 400)]],
)

# Subsequent calls with same image use cached embeddings
predictions = model.segment_images(
    images=img,
    boxes=[[(150, 250, 350, 450)]],  # Different prompt, same image
)
```
