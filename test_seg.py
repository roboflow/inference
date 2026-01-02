from inference import get_model
import cv2
import supervision as sv

# given
model = get_model("rfdetr-seg-preview")
image_url = "https://media.roboflow.com/dog.jpeg"

# load image for visualization
import urllib.request
import numpy as np
resp = urllib.request.urlopen(image_url)
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# when
result_raw = model.infer(image)[0]
result = sv.Detections.from_inference(result_raw)

print(result)

# visualize and save
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = mask_annotator.annotate(scene=image.copy(), detections=result)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=result)

cv2.imwrite("rfdetr_seg_result.png", annotated_image)
print("Saved visualization to rfdetr_seg_result.png")