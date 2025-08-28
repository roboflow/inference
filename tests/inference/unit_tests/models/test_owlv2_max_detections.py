from unittest.mock import MagicMock

import torch

from inference.models.owlv2 import owlv2


def test_infer_from_embed_respects_max_detections(monkeypatch):
    model = owlv2.OwlV2.__new__(owlv2.OwlV2)
    image_boxes = torch.tensor(
        [[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3], [0, 0, 4, 4]],
        dtype=torch.float32,
    )
    image_class_embeds = torch.zeros((4, 2))
    model.get_image_embeds = MagicMock(
        return_value=(None, image_boxes, image_class_embeds, None, None)
    )

    def fake_get_class_preds_from_embeds(*args, **kwargs):
        boxes = image_boxes
        classes = torch.zeros(4, dtype=torch.int64)
        scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
        return boxes, classes, scores

    monkeypatch.setattr(
        owlv2, "get_class_preds_from_embeds", fake_get_class_preds_from_embeds
    )
    monkeypatch.setattr(
        owlv2.torchvision.ops,
        "nms",
        lambda boxes, scores, iou: torch.arange(boxes.shape[0]),
    )

    query_embeddings = {"a": {"positive": torch.zeros((1, 2)), "negative": None}}
    predictions = model.infer_from_embed(
        "hash", query_embeddings, confidence=0.5, iou_threshold=0.5, max_detections=2
    )
    assert len(predictions) == 2
