from inference import get_roboflow_model
model = get_roboflow_model("microsoft-coco-obj-det/2")
print(model.infer("https://source.roboflow.one/AWklLvEzMUUAhCA8Jl82sBDxIwd2/acAl5kzbUmPv3Qj3y7Uh/original.jpg"))
