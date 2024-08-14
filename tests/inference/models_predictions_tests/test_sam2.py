import numpy as np
import pytest
import torch

from inference.models.sam2 import SegmentAnything2
from inference.core.entities.requests.sam2 import Sam2PromptSet




@pytest.mark.slow
def test_sam2_single_image_embedding(
    sam2_small_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    # when
    embedding, img_shape, id_ = model.embed_image(example_image)

    # then
    drift = np.abs(torch.sum(embedding['image_embed']).cpu().detach().numpy() - 25155.9336)
    assert  drift < 1, "embedding sum doesnt drift/change"
    assert img_shape == (427,640) , "Image shape must match the expected shape" 
    


@pytest.mark.slow
def test_sam2_single_prompted_image_segmentation(
    sam2_small_model: str,
    truck_image: np.ndarray,
    sam2_small_truck_mask: np.ndarray,
) -> None:
    # given
    model = SegmentAnything2(model_id=sam2_small_model)

    prompt = Sam2PromptSet(
        prompts=[{"points": [{"x": 500, "y": 375, "positive": True}]}]
    )

    # when
    masks, scores, low_res_logits = model.segment_image(truck_image, prompts=prompt)    
    
    
    #vislualization fo result for debugging
    # expected result for small model is the part of the rear window on the truck
    # where the prompt point is provided
    #
    # import supervision as sv
    # import matplotlib.pyplot as plt
    # from PIL import Image
    
    # raw_masks = raw_masks >= model.predictor.mask_threshold
    # mask_annotator = sv.MaskAnnotator()
    # detections = sv.Detections(
    #     xyxy=np.array([[0, 0, 100, 100]]),
    #     mask=np.array(raw_masks)
    # )
    # detections.class_id = [i for i in range(len(detections))]
    # annotated_image = mask_annotator.annotate(truck_image.copy(), detections)
    # im = Image.fromarray(annotated_image)
    # im.save("sam-test.png")
    
    # print("mask", np.sum(masks))
    # print("scores", scores)
    # print("low_res_logits", np.sum(low_res_logits))

    # then
    score_drift = np.abs(scores[0] - 0.9426716566085815)
    assert np.allclose(masks, sam2_small_truck_mask, atol=0.01), "segmentation mask is as expected"
    assert score_drift < 0.01, "score doesnt drift/change"
    # assert np.sum(low_res_logits) == -1772890.2, "logits is as expected"
    
    


