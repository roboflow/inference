import os
os.environ["API_KEY"] = "API_KEY"

import timeit
import numpy as np
import supervision as sv
from PIL import Image

from inference.core.entities.requests.sam2 import Sam2PromptSet
from inference.models.sam2 import SegmentAnything2

image_path = "./examples/sam2/hand.png"
m = SegmentAnything2(model_id="sam2/hiera_large")

# call embed_image before segment_image to precompute embeddings

start = timeit.timeit()
embedding, img_shape, id_ = m.embed_image(image_path)
end = timeit.timeit()
print("embedd time: ", end - start)

# segments image using cached embedding if it exists, else computes it on the fly
start = timeit.timeit()
raw_masks, scores, raw_low_res_masks = m.segment_image(image_path, use_mask_input_cache=True)
end = timeit.timeit()
print("segment time: ", end - start)

# convert binary masks to polygons
raw_masks = raw_masks >= m.predictor.mask_threshold

point = [250, 800]
label = False
# give a negative point (point_label 0) or a positive example (point_label 1)
prompt = Sam2PromptSet(
    prompts=[{"points": [{"x": point[0], "y": point[1], "positive": label}]}]
)

# uses cached masks from prior call

raw_masks2, scores2, raw_low_res_masks2 = m.segment_image(
    image_path,
    prompts=prompt,
    use_mask_input_cache=True,
)

raw_masks2 = raw_masks2 >= m.predictor.mask_threshold

image = np.array(Image.open(image_path).convert("RGB"))

mask_annotator = sv.MaskAnnotator()
dot_annotator = sv.DotAnnotator()

detections = sv.Detections(
    xyxy=np.array([[0, 0, 100, 100]]),
    mask=np.array(raw_masks)
)
detections.class_id = [i for i in range(len(detections))]
annotated_image = mask_annotator.annotate(image.copy(), detections)
im = Image.fromarray(annotated_image)
im.save("sam.png")

detections = sv.Detections(
    xyxy=np.array([[0, 0, 100, 100]]), mask=np.array(raw_masks2)
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