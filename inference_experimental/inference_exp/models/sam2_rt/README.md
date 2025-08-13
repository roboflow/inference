This model is taking advantage of [Segment Anything 2 Real Time fork](https://github.com/Gy920/segment-anything-2-real-time)

```python
import cv2 as cv
import supervision as sv

from inference_exp import AutoModel


model = AutoModel.from_pretrained("sam2-rt-hiera-tiny")
mask_annotator = sv.MaskAnnotator(opacity=0.7, color_lookup=sv.ColorLookup.TRACK)
# Download from https://drive.google.com/uc?id=1EsxiyaYGj3FeXSXoK51pre5OjVGWTCSE
vid = cv.VideoCapture("G/boston-celtics-new-york-knicks-game-4-q1-05.06-05.01.mp4")
f_num = 0
while True:
    is_ok, frame = vid.read()
    if not is_ok:
        break
    if f_num == 0:
        ids, masks = model(frame, [(477, 337, 560, 529)])
        f_num += 1
    else:
        ids, masks = model(frame)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks,
        tracker_id=ids
    )

    annotated_frame = mask_annotator.annotate(scene=frame, detections=detections)

    cv.imshow("", annotated_frame)
    cv.waitKey(1)
```
