import torch
from typing import Optional, Dict, Any, List
import warnings
from loguru import logger
import numpy as np

from lidra.model.backbone.sam2.build_sam import build_sam2_hf
import lidra.model.backbone.sam2.sam2_image_predictor as sam2_image_predictor


class SAM2ImagePredictor(torch.nn.Module):
    def __init__(
        self,
        hf_hub_model: str = "facebook/sam2.1-hiera-large",
        sam2_config: str = "etc/lidra/model/backbone/sam2/sam2.1_hiera_l.yaml",
        kwargs: Dict[str, Any] = {},
    ):
        super().__init__()

        self.hf_hub_model = hf_hub_model
        self.embed_dim = 256  # Always 256 for SAM2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Build base model
            self.backbone = build_sam2_hf(
                hf_hub_model,
                sam2_config,
                **kwargs,
            )

            # Wrap with SAM2ImagePredictor for image embedding.
            self._sam2_image_predictor = sam2_image_predictor.SAM2ImagePredictor(
                self.backbone,
                **kwargs,
            )

        # freeze
        self.requires_grad_(False)
        self.eval()

    def get_image_embedding(self, rgb_image: List[np.ndarray]):
        """
        Args:
            rgb_image: (np.ndarray or PIL Image): The input image to embed in RGB format.
                The image should be in HWC format if np.ndarray, or WHC format if PIL Image
                with pixel values in [0, 255].
        Returns:
            image_embedding: (B, T, D)
        """
        # TODO(sasha) remove conversion to numpy required for set_image_batch
        rgb_image = rgb_image.float()
        rgb_numpy = rgb_image.permute(0, 2, 3, 1).cpu().numpy()
        rgb_numpy_list = list(rgb_numpy)

        # print(
        #     f"rgb_numpy shape: {rgb_numpy.shape=} {rgb_numpy_list[0].shape=}",
        #     flush=True,
        # )

        self._sam2_image_predictor.set_image_batch(rgb_numpy_list)
        image_embedding = self._sam2_image_predictor.get_image_embedding()
        assert (
            image_embedding.shape[1] == self.embed_dim
        ), f"{image_embedding.shape[1]=} != {self.embed_dim=}"
        return image_embedding

    def get_prompt_embedding(
        self,
        mask_input: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ):
        # called: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L406
        # returns: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/prompt_encoder.py#L202
        low_res_masks = torch.nn.functional.interpolate(
            mask_input,
            size=self.backbone.sam_prompt_encoder.mask_input_size,
            mode="bilinear",
        )

        # We get the embeddings directly from the prompt encoder
        #   but instead if we wanted to get direct high-res masks from SAM2...
        #   >>> masks, _, _ = predictor.predict_batch(
        #   >>>  mask_input_batch=low_res_masks, multimask_output=False
        #   >>> )
        point_and_box_embeddings, mask_embeddings = self.backbone.sam_prompt_encoder(
            points=points, boxes=boxes, masks=low_res_masks
        )
        return point_and_box_embeddings, mask_embeddings

    def get_image_pe(self):
        return self.backbone.sam_prompt_encoder.get_dense_pe()

    def forward(self, image: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            image: (B, C, H, W)
            masks: (B, M, H, W)
        Returns:
            sam_embeddings: (B, M, T, D) # D=256 in SAM2
        """
        # with torch.use_deterministic_algorithms(False):
        image_embedding = self.get_image_embedding(image)
        _, mask_embedding = self.get_prompt_embedding(masks)
        image_pe = self.get_image_pe()

        # Combine the image and prompt embeddings as done in SAM2
        # Sum image and prompt embeddings: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/mask_decoder.py#L205
        sam_embeddings = image_embedding + mask_embedding

        # Should we add image_pe?
        # PE added here: https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/transformer.py#L99
        # However, the first embedding before the transformer gets skipped https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/sam/transformer.py#L56
        sam_embeddings = sam_embeddings + image_pe
        sam_embeddings = sam_embeddings.permute(0, 2, 3, 1)
        sam_embeddings = sam_embeddings.reshape(
            sam_embeddings.shape[0], -1, self.embed_dim
        )
        return sam_embeddings
