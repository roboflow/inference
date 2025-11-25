from collections import namedtuple
from typing import List, Optional, Dict, Any, Tuple, Callable, Sequence, Iterable
import os
from tqdm import tqdm
from functools import partial

from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn
from loguru import logger
import optree
import skimage.io as io
import torch
import torch.nn.functional as F

from lidra.model.module.base import Base, TrainableBackbone, BackboneWithBenefits
from lidra.model.backbone.autoencoder.vae import DiagonalGaussianDistribution

from lidra.data.utils import (
    build_batch_extractor,
    empty_mapping,
    tree_reduce_unique,
    get_child,
)
from lidra.data.dataset.return_type import extract_data, extract_sample_uuid

import lidra.model.module.tdfy.batch_transform_utils.trellis as trellis_batch_transform_utils

from lidra.data.dataset.tdfy.pose_target import InstancePose, PoseTargetConverter
from lidra.data.dataset.tdfy.transforms_3d import DecomposedTransform

warning_cache = WarningCache()


class BaseVAE(Base):
    DUMP_TYPES = {"image", "tensor", None}

    def __init__(
        self,
        encoder: Optional[TrainableBackbone] = None,
        decoder: Optional[TrainableBackbone] = None,
        batch_encoder_input_mapping: Dict[str, Any] = "input",
        batch_decoder_input_mapping: Dict[str, Any] = empty_mapping,
        extract_encoder_out: Optional[Callable] = None,
        validation_dump_type: Optional[str] = None,
        batch_preprocessing_fn: Optional[Callable] = None,
        prior_loss_lambda: float = 1e-6,
        **kwargs,
    ):
        """
        Base class for VAE, default would assume a gaussian distribution
        Default assumes a Gaussian distribution
        Encoder/Decoder: trainable model to encode to latents/ decode from latents
            Optional since we may only use encoder/ decoder sometime
            The Encoder should output sampled latent during training: e.g. the reparametrization
                trick needs to happen in the encoder
        prior: a callable that takes input of a shape of the latent and generate the latent
        """
        models = {}
        if encoder is not None:
            models["encoder"] = encoder
        if decoder is not None:
            models["decoder"] = decoder
        self._batch_preprocessing_fn = (
            (lambda x: x) if batch_preprocessing_fn is None else batch_preprocessing_fn
        )

        super().__init__(
            models,
            **kwargs,
        )

        self.encoder_extractor_fn = build_batch_extractor(batch_encoder_input_mapping)
        self.decoder_extractor_fn = build_batch_extractor(batch_decoder_input_mapping)
        self.extract_encoder_out = extract_encoder_out
        assert (
            validation_dump_type in BaseVAE.DUMP_TYPES
        ), f'invalid validation_dump_type "{validation_dump_type}", should be one of {BaseVAE.DUMP_TYPES}'
        self.validation_dump_type = validation_dump_type
        self.prior_loss_lambda = prior_loss_lambda

    def _encode(self, x, *args, **kwargs):
        assert "encoder" in self.base_models, "Encoder not available"
        return x, self.base_models["encoder"](x, *args, **kwargs)

    def _decode(self, x, *args, **kwargs):
        assert "decoder" in self.base_models, "Decoder not available"
        return self.base_models["decoder"](x, *args, **kwargs)

    def _prior(self, x_shape, x_device):
        pass

    def _sample_prior(self, x_shape, x_device):
        def is_shape(maybe_shape):
            return isinstance(maybe_shape, Sequence) and all(
                (isinstance(s, int) and s >= 0) for s in maybe_shape
            )

        return optree.tree_map(
            partial(self._prior, x_device=x_device),
            x_shape,
            is_leaf=is_shape,
            none_is_leaf=False,
        )

    @staticmethod
    def _get_batch_size(x):
        first_tensor = optree.tree_flatten(x)[0][0]
        return first_tensor.shape[0]

    def _prior_loss(self, encoded_pred):
        pass

    def _recon_loss(self, decoded_pred, gt):
        pass

    def _encode_decode_step(self, batch, batch_idx):
        batch = self._batch_preprocessing_fn(batch)

        # extract inputs from batch
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        gt, encoded_pred = self._encode(*encoder_args, **encoder_kwargs)
        decoder_args, decoder_kwargs = self.decoder_extractor_fn(batch)
        if self.extract_encoder_out is not None:
            x = self.extract_encoder_out(encoded_pred)
        else:
            x = encoded_pred
        decoded_pred = self._decode(x, decoder_args, decoder_kwargs)
        return gt, encoded_pred, decoded_pred

    def _make_effective_loss(self, loss, batch_size):
        n_items = self.all_gather(batch_size).sum()
        return loss * n_items / 512

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None

        gt, encoded_pred, decoded_pred = self._encode_decode_step(batch, batch_idx)

        batch_size = BaseVAE._get_batch_size(decoded_pred)
        loss = self._prior_loss(
            encoded_pred
        ) * self.prior_loss_lambda + self._recon_loss(decoded_pred, gt)
        loss = self._make_effective_loss(loss, batch_size=batch_size)
        return loss

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        show_progress: bool = False,
    ):
        if batch is None:
            return None
        batch = self._batch_preprocessing_fn(batch)
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        decoder_args, decoder_kwargs = self.decoder_extractor_fn(batch)

        # encode (compute x only to get its device and shape, is there a better way ?)
        _, encoded_pred = self._encode(*encoder_args, **encoder_kwargs)
        if self.extract_encoder_out is not None:
            x = self.extract_encoder_out(encoded_pred)
        else:
            x = encoded_pred
        shape = optree.tree_map(lambda tensor: tensor.shape, x)
        device = tree_reduce_unique(lambda tensor: tensor.device, x)
        prior_x = self._sample_prior(shape, device)
        decoded_pred = self._decode(prior_x, decoder_args, decoder_kwargs)
        return decoded_pred

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None

        gt, encoded_pred, decoded_pred = self._encode_decode_step(batch, batch_idx)

        metrics_dict = self._compute_and_log_val_metrics(
            (gt, encoded_pred, decoded_pred), batch
        )
        return metrics_dict


class SparseStructureVAE(BaseVAE):
    @staticmethod
    def _batch_processing(batch):
        batch = batch[1]
        return batch

    def __init__(
        self,
        loss_type="dice",
        extract_encoder_out=lambda x: x["z"],
        batch_encoder_input_mapping: Dict[str, Any] = "voxels",
        **kwargs,
    ):
        super().__init__(
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=self._batch_processing,
            extract_encoder_out=extract_encoder_out,
            **kwargs,
        )
        self.loss_type = loss_type

    def _prior_loss(self, encoded_pred):
        _, mean, logvar = encoded_pred.values()
        return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    def _recon_loss(self, decoded_pred, gt):
        logits = decoded_pred
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                logits, gt.float(), reduction="sum"
            )
        elif self.loss_type == "l1":
            return F.l1_loss(F.sigmoid(logits), gt.float(), reduction="sum")
        elif self.loss_type == "dice":
            batch_size = logits.shape[0]
            logits = F.sigmoid(logits)
            return batch_size * (
                1
                - (2 * (logits * gt.float()).sum() + 1)
                / (logits.sum() + gt.float().sum() + 1)
            )
        else:
            raise ValueError(f"Invalid loss type {self.loss_type}")

    def _decode(self, x, *args, **kwargs):
        assert "decoder" in self.base_models, "Decoder not available"
        return self.base_models["decoder"](x)
