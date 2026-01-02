from inference import get_model

#API_KEY = "Em3CODkJT1vbZ52wlWwk"
IMAGE = "blazenek-scaled.jpg"

for model_name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
    model = get_model(model_name) #, api_key=API_KEY)
    result = model.infer(IMAGE, confidence=0.7)
    r = result[0]
    preds = r.predictions
    top_class = max(preds.items(), key=lambda x: x[1].confidence)
    print(f"{model_name}: {top_class[0]} ({top_class[1].confidence:.4f})")