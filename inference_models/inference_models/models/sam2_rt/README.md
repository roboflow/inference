This model is taking advantage of [Segment Anything 2 Real Time fork](https://github.com/Gy920/segment-anything-2-real-time)

# Tracking

```python
import cv2 as cv
import supervision as sv

from inference_models import AutoModel

# Available checkpoints: Gy920/sam2-1-hiera-tiny, Gy920/sam2-1-hiera-small, Gy920/sam2-1-hiera-base-plus, Gy920/sam2-1-hiera-large
model = AutoModel.from_pretrained("Gy920/sam2-1-hiera-tiny")

mask_annotator = sv.MaskAnnotator(opacity=0.7, color_lookup=sv.ColorLookup.TRACK)
# Download from https://drive.google.com/uc?id=1EsxiyaYGj3FeXSXoK51pre5OjVGWTCSE
vid = cv.VideoCapture("G/boston-celtics-new-york-knicks-game-4-q1-05.06-05.01.mp4")
f_num = 0
while True:
    is_ok, frame = vid.read()
    if not is_ok:
        break
    if f_num == 0:
        ids, masks, *_ = model.prompt(frame, bboxes=[(477, 337, 560, 529), (633, 570, 843, 804)])
        f_num += 1
    else:
        ids, masks, *_ = model.track(frame)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks,
        tracker_id=ids
    )

    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)

    cv.imshow("", annotated_frame)
    cv.waitKey(1)
```

# Instance segmentation

```python
import cv2 as cv
import supervision as sv

from inference_models import AutoModel

model = AutoModel.from_pretrained("sam2-rt-hiera-tiny")

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator(color=sv.Color.BLACK)

# Download from https://media.roboflow.com/inference/sam2/hand.png
img = cv.imread("G/hand.png")
masks, *_ = model.prompt(img, [(117, 303, 670, 650)])
classes = np.array([0 for _ in masks])

detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=masks),
    mask=masks,
    class_id=classes,
)

annotated_frame = mask_annotator.annotate(scene=img, detections=detections)
annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

cv.imshow("", annotated_frame)
cv.waitKey(0)
```
