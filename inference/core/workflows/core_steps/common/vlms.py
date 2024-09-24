VLM_TASKS_METADATA = {
    "unconstrained": {
        "name": "Open Prompt",
        "description": "Let you use arbitrary prompt",
    },
    "ocr": {
        "name": "Text Recognition (OCR)",
        "description": "Model recognises text in the image",
    },
    "ocr-with-text-detection": {
        "name": "Text Detection & Recognition (OCR)",
        "description": "Model recognises text in the image",
    },
    "visual-question-answering": {
        "name": "Visual Question Answering",
        "description": "Model answers the question you submit in the prompt",
    },
    "caption": {
        "name": "Captioning (short)",
        "description": "Model describes the image",
    },
    "detailed-caption": {
        "name": "Captioning",
        "description": "Model provides long description of the image",
    },
    "more-detailed-caption": {
        "name": "Captioning (long)",
        "description": "Model provides very long description of the image",
    },
    "classification": {
        "name": "Multi-Class Classification",
        "description": "Model classifies the image content selecting one of many classes",
    },
    "multi-label-classification": {
        "name": "Multi-Label Classification",
        "description": "Model classifies the image content selecting potentially multiple classes",
    },
    "object-detection": {
        "name": "Detection",
        "description": "Model detect bounding boxes over for set of classes",
    },
    "open-vocabulary-object-detection": {
        "name": "Open Vocabulary Detection",
        "description": "Model detect bounding boxes for arbitrary classes",
    },
    "object-detection-and-caption": {
        "name": "Detection & Captioning",
        "description": "Model detects Regions of Interest and caption them",
    },
    "phrase-grounded-object-detection": {
        "name": "Phase Grounded Detection",
        "description": "Based on textual prompt model detect objects that are suggested",
    },
    "phrase-grounded-instance-segmentation": {
        "name": "Phase Grounded Segmentation",
        "description": "Based on textual prompt model performs instance segmentation of objects that are suggested",
    },
    "detection-grounded-instance-segmentation": {
        "name": "Segmentation of RoI",
        "description": "Model performs instance segmentation within provided Region of Interest (Bounding Box)",
    },
    "detection-grounded-classification": {
        "name": "Classification of RoI",
        "description": "Model performs instance classification of provided Region of Interest (Bounding Box)",
    },
    "detection-grounded-caption": {
        "name": "Captioning of RoI",
        "description": "Model performs captioning of provided Region of Interest (Bounding Box)",
    },
    "detection-grounded-ocr": {
        "name": "Text Recognition (OCR) of RoI",
        "description": "Model performs OCR of provided Region of Interest (Bounding Box) to "
        "recognise text within region",
    },
    "region-proposal": {
        "name": "Regions of Interest proposal",
        "description": "Model proposes Regions of Interest (Bounding Boxes) in the image",
    },
    "structured-answering": {
        "name": "Structured Output Generation",
        "description": "Model produces JSON structure that you specify",
    },
}
