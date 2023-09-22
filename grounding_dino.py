# from inference.models.grounding_dino.grounding_dino_model import GroundingDINO
# import os

# model = GroundingDINO(api_key=os.environ["ROBOFLOW_API_KEY"])

# results = model.infer(image="https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg", confidence=0.5, iou_threshold=0.5)

# print(results)

from inference.models.doctr.doctr_model import DocTR
import os

model = DocTR(api_key=os.environ["ROBOFLOW_API_KEY"])

results = model.infer(image="https://source.roboflow.com/pwYAXv9BTpqLyFfgQoPZ/u48G0UpWfk8giSw7wrU8/original.jpg", confidence=0.5, iou_threshold=0.5)

print(results)