VLM_TASKS_METADATA = {
    "unconstrained": {
        "name": "Open Prompt",
        "description": "Use any prompt to generate a raw response",
    },
    "ocr": {
        "name": "Text Recognition (OCR)",
        "description": "Model recognizes text in the image",
    },
    "ocr-with-text-detection": {
        "name": "Text Detection & Recognition (OCR)",
        "description": "Model detects text regions in the image, and then performs OCR on each detected region",
    },
    "visual-question-answering": {
        "name": "Visual Question Answering",
        "description": "Model answers the question you submit in the prompt",
    },
    "caption": {
        "name": "Captioning (short)",
        "description": "Model provides a short description of the image",
    },
    "detailed-caption": {
        "name": "Captioning",
        "description": "Model provides a long description of the image",
    },
    "more-detailed-caption": {
        "name": "Captioning (long)",
        "description": "Model provides a very long description of the image",
    },
    "classification": {
        "name": "Single-Label Classification",
        "description": "Model classifies the image content as one of the provided classes",
    },
    "multi-label-classification": {
        "name": "Multi-Label Classification",
        "description": "Model classifies the image content as one or more of the provided classes",
    },
    "object-detection": {
        "name": "Unprompted Object Detection",
        "description": "Model detects and returns the bounding boxes for prominent objects in the image",
    },
    "open-vocabulary-object-detection": {
        "name": "Object Detection",
        "description": "Model detects and returns the bounding boxes for the provided classes",
    },
    "object-detection-and-caption": {
        "name": "Detection & Captioning",
        "description": "Model detects prominent objects and captions them",
    },
    "phrase-grounded-object-detection": {
        "name": "Prompted Object Detection",
        "description": "Based on the textual prompt, model detects objects matching the descriptions",
    },
    "phrase-grounded-instance-segmentation": {
        "name": "Prompted Instance Segmentation",
        "description": "Based on the textual prompt, model segments objects matching the descriptions",
    },
    "detection-grounded-instance-segmentation": {
        "name": "Segment Bounding Box",
        "description": "Model segments the object in the provided bounding box into a polygon",
    },
    "detection-grounded-classification": {
        "name": "Classification of Bounding Box",
        "description": "Model classifies the object inside the provided bounding box",
    },
    "detection-grounded-caption": {
        "name": "Captioning of Bounding Box",
        "description": "Model captions the object in the provided bounding box",
    },
    "detection-grounded-ocr": {
        "name": "Text Recognition (OCR) for Bounding Box",
        "description": "Model performs OCR on the text inside the provided bounding box",
    },
    "region-proposal": {
        "name": "Regions of Interest proposal",
        "description": "Model proposes Regions of Interest (Bounding Boxes) in the image",
    },
    "structured-answering": {
        "name": "Structured Output Generation",
        "description": "Model returns a JSON response with the specified fields",
    },
}
