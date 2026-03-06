from collections import namedtuple
from typing import List, Optional, Dict, Any, Tuple, Callable, Sequence
import os
from tqdm import tqdm

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn
from lidra.model.utils import _get_total_norm
from loguru import logger
import optree
import skimage.io as io
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

from lidra.model.module.base import Base, TrainableBackbone, BackboneWithBenefits
from lidra.model.backbone.autoencoder.vae import DiagonalGaussianDistribution

from lidra.data.utils import (
    build_batch_extractor,
    empty_mapping,
    tree_reduce_unique,
)
from lidra.data.dataset.return_type import extract_data, extract_sample_uuid

import lidra.model.module.tdfy.batch_transform_utils.trellis as trellis_batch_transform_utils

from lidra.data.dataset.tdfy.pose_target import InstancePose, PoseTargetConverter
from lidra.data.dataset.tdfy.transforms_3d import DecomposedTransform
from lidra.data.dataset.tdfy.pose_target import LogScaleShiftNormalizer

warning_cache = WarningCache()


class Generative(Base):
    DUMP_TYPES = {"image", "tensor", None}

    def __init__(
        self,
        generator: TrainableBackbone,
        encoder: Optional[BackboneWithBenefits] = None,
        decoder: Optional[BackboneWithBenefits] = None,
        batch_encoder_input_mapping: Dict[str, Any] = "input",
        batch_decoder_input_mapping: Dict[str, Any] = empty_mapping,
        batch_conditions_mapping: Dict[str, Any] = empty_mapping,
        condition_embedder: Optional[BackboneWithBenefits] = None,
        validation_dump_type: Optional[str] = None,
        batch_preprocessing_fn: Optional[Callable] = None,
        max_sequential_steps: Optional[int] = None,
        **kwargs,
    ):
        models = {}
        if generator is not None:
            models["generator"] = generator
        if encoder is not None:
            models["encoder"] = encoder
        if decoder is not None:
            models["decoder"] = decoder
        if condition_embedder is not None:
            models["condition_embedder"] = condition_embedder

        self._batch_preprocessing_fn = (
            (lambda x: x) if batch_preprocessing_fn is None else batch_preprocessing_fn
        )

        super().__init__(
            models,
            **kwargs,
        )

        self.max_sequential_steps = max_sequential_steps

        self.encoder_extractor_fn = build_batch_extractor(batch_encoder_input_mapping)
        self.decoder_extractor_fn = build_batch_extractor(batch_decoder_input_mapping)
        self.condition_extractor_fn = build_batch_extractor(batch_conditions_mapping)
        assert (
            validation_dump_type in Generative.DUMP_TYPES
        ), f'invalid validation_dump_type "{validation_dump_type}", should be one of {Generative.DUMP_TYPES}'
        self.validation_dump_type = validation_dump_type

    def _encode(self, x, *args, **kwargs):
        if "encoder" in self.base_models:
            with torch.no_grad():
                return self.base_models["encoder"](x, *args, **kwargs)
        return x

    def _decode(self, x, *args, **kwargs):
        if "decoder" in self.base_models:
            with torch.no_grad():
                return self.base_models["decoder"](x, *args, **kwargs)
        return x

    def _embed_condition(self, *args, **kwargs):
        if "condition_embedder" in self.base_models:
            return self.base_models["condition_embedder"](*args, **kwargs), None, None
        return None, args, kwargs

    def _generate(self, x_shape, x_device, *args, **kwargs):
        if "generator" in self.base_models:
            return self.base_models["generator"](x_shape, x_device, *args, **kwargs)
        return x

    def _log_likelihood(
        self,
        x1,
        *args,
        **kwargs,
    ):
        if "generator" in self.base_models:
            return self.base_models["generator"].log_likelihood(
                x1,
                *args,
                **kwargs,
            )
        raise NotImplementedError("Log likelihood requires a generator.")

    def generate_iter(self, batch):
        # skip if batch happens to be empty
        if batch is None:
            return None

        # preprocess
        batch = self._batch_preprocessing_fn(batch)

        # extract inputs from batch
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        condition_args, condition_kwargs = self.condition_extractor_fn(batch)

        # encode
        x = self._encode(*encoder_args, **encoder_kwargs)
        # assert isinstance(
        #     x, torch.Tensor
        # ), "x should be a tensor -- dict version not supported"
        yield from self.base_models["generator"]._generate_iter(
            x.shape,
            x.device,
            *condition_args,
            **condition_kwargs,
        )

    @staticmethod
    def _get_batch_size(x):
        first_tensor = optree.tree_flatten(x)[0][0]
        return first_tensor.shape[0]

    def _step(self, label, batch, batch_idx):
        # skip if batch happens to be empty
        if batch is None:
            return None

        # preprocess
        batch = self._batch_preprocessing_fn(batch)

        # extract inputs from batch
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        condition_args, condition_kwargs = self.condition_extractor_fn(batch)

        # encode
        x = self._encode(*encoder_args, **encoder_kwargs)
        batch_size = Generative._get_batch_size(x)

        embedded_cond, condition_args, condition_kwargs = self._embed_condition(
            *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        # compute predictions
        loss, detail_losses = self.base_models["generator"].loss(
            x,
            *condition_args,
            **condition_kwargs,
        )

        self.log(
            name=f"{label}/loss",
            value=loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log detailed losses if available
        if detail_losses is not None:
            for loss_name, loss_value in detail_losses.items():
                self.log(
                    name=f"{label}/{loss_name}",
                    value=loss_value,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        return loss, batch_size

    def _make_effective_loss(self, loss, batch_size):
        n_items = self.all_gather(batch_size).sum()
        return loss * n_items / 512

    def training_step(self, batch, batch_idx):
        loss, batch_size = self._step("train", batch, batch_idx)
        loss = self._make_effective_loss(loss, batch_size=batch_size)
        # log learning rate
        self.log(
            "trainer/lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        generated_x = self.predict_step(batch, batch_idx)
        metrics_dict = self._compute_and_log_val_metrics(generated_x, batch)

        # Use a prediction writer callback instead
        self._save_to_disk(generated_x, batch_idx)
        return metrics_dict

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        show_progress: bool = False,
        n_trials: Optional[int] = None,
    ):
        """
        Generate samples from the model based on input batch.

        Args:
            batch: Input batch of data
            batch_idx: Index of the current batch
            dataloader_idx: Index of the dataloader (default: 0)
            show_progress: Whether to display a progress bar during generation (default: False)
            max_trials: Number of sequential samples to generate; if None, uses self.max_sequential_steps;
                        if not None, returns only the first sample (default: None)

        Returns:
            Either a single generated sample (if max_trials is not None) or a list of generated samples

        Note:
            This is where guidance policies could be implemented (e.g., interactive editing,
            error estimation with repeated denoising).
        """
        if batch is None:
            return None

        if n_trials is None:
            n_trials = self.max_sequential_steps

        return_first_element = n_trials is None
        if return_first_element:
            n_trials = 1
            warning_cache.warn(
                "n_trials will be required to be 'int' in a future version, and predict_step will return a list. For now, returning first element only."
            )

        # preprocess once at the beginning
        batch = self._batch_preprocessing_fn(batch)
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        decoder_args, decoder_kwargs = self.decoder_extractor_fn(batch)
        condition_args, condition_kwargs = self.condition_extractor_fn(batch)

        # encode (compute x only to get its device and shape, is there a better way ?)
        x = self._encode(*encoder_args, **encoder_kwargs)

        generated_xs = []

        # Add tqdm progress bar
        iterator = range(n_trials)
        if show_progress:
            iterator = tqdm(iterator, desc="Sequential prediction", total=n_trials)

        shape = optree.tree_map(lambda tensor: tensor.shape, x)
        device = tree_reduce_unique(lambda tensor: tensor.device, x)

        embedded_cond, condition_args, condition_kwargs = self._embed_condition(
            *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        for _ in iterator:
            # generate sample
            # Note: here is where we would put a search policy
            #  e.g.: x = generated_x
            generated_x = self._generate(
                shape,
                device,
                *condition_args,
                **condition_kwargs,
            )

            # decode
            generated_x = self._decode(generated_x, *decoder_args, **decoder_kwargs)
            generated_xs.append(generated_x)

        if return_first_element:
            warning_cache.warn(
                "Returning first element only is deprecated and will be removed in a future version."
            )
            return generated_xs[0]
        else:
            return generated_xs

    @torch.autograd.grad_mode.inference_mode(mode=False)
    def log_likelihood(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        show_progress: bool = False,
        steps: Optional[int] = None,
        z_samples: int = 1,
        solver: Optional[str] = None,
    ):
        # skip if batch happens to be empty
        if batch is None:
            return None

        # preprocess
        batch = self._batch_preprocessing_fn(batch)

        # extract inputs from batch
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        condition_args, condition_kwargs = self.condition_extractor_fn(batch)
        if len(condition_args) > 0:
            raise ValueError(
                "condition_args in log_likelihood must be empty (because of implementation)"
            )

        # encode
        x1 = self._encode(*encoder_args, **encoder_kwargs)
        log_p1 = self._log_likelihood(
            x1,
            solver=solver,
            steps=steps,
            z_samples=z_samples,
            **condition_kwargs,
        )
        return log_p1

    def _save_to_disk(self, generated_x, batch_idx):
        # Skip saving if validation_dump_type is None
        if self.validation_dump_type is None:
            return

        # Deprecation warning for validation_dump_type
        warning_cache.warn(
            "The validation_dump_type parameter is deprecated and will be removed in a future version. "
            "Use a prediction writer callback instead."
        )
        ## TODO(Pierre) : Clean dump extension
        if not isinstance(generated_x, torch.Tensor):
            assert isinstance(generated_x, dict)
            assert len(generated_x) == 1
            generated_x = list(generated_x.values())[0]

        if self.validation_dump_type == "image":
            save_images_to_disk(
                f"generated_images/{self.current_epoch}/{self.trainer.global_rank}",
                generated_x,
                batch_idx,
            )
        elif self.validation_dump_type == "tensor":
            save_tensors_to_disk(
                f"generated_images/{self.current_epoch}/{self.trainer.global_rank}",
                generated_x,
                batch_idx,
            )

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # the wandb logger lives in self.loggers
        # find the wandb logger and watch the model and gradients
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                # log gradients, parameter histogram and model topology
                logger.watch(
                    self._base_models.generator.reverse_fn,
                    log="all",
                    log_freq=500,
                    log_graph=False,
                )

    def on_before_optimizer_step(self, optimizer):
        """
        Log total gradient norm according to PyTorch Lightning's recommendation.
        This is called before the optimizer step, after gradients are computed.
        For mixed precision training, gradients are already unscaled here.
        """
        # Only log gradient norm every 1 steps
        if self.global_step % 1 != 0:
            return

        # Compute the 2-norm for each module in base_models
        # If using mixed precision, the gradients are already unscaled here
        grads = []
        for opt in self.trainer.optimizers:
            for group in opt.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        grads.append(param.grad)

        # Time the gradient norm calculation
        total_norm = _get_total_norm(grads, norm_type=2.0)

        # must use .full_tensor() to trigger a all-reduce across ranks
        # see: https://www.linkedin.com/posts/kai-zhang-53910214a_dtensor-with-normpartial-stores-local-partial-activity-7360490130744266753-2EhX
        # otherwise, the norm will have different values on different ranks in FSDP2, making logging wrong
        if isinstance(total_norm, torch.distributed.tensor.DTensor):
            total_norm_val = total_norm.full_tensor().item()
        else:
            total_norm_val = total_norm.item()

        self.log("train/grad_norm", total_norm_val, on_step=True, prog_bar=True)


# Saving -- recommended to now use a prediction writer callback
def save_images_to_disk(path, x, batch_idx):
    os.makedirs(path, exist_ok=True)
    for i, img in enumerate(x):
        fname = os.path.join(path, f"{batch_idx:06d}-{i:02d}.png")
        img = torch.clamp((img + 1) / 2, min=0, max=1)
        img = (img * 255.0).to(torch.uint8)
        img = torch.permute(img, (1, 2, 0))
        img = img.detach().cpu().numpy()
        io.imsave(fname, img)


def save_tensors_to_disk(path, x, batch_idx):
    bs = Generative._get_batch_size(x)
    x = optree.tree_map(lambda tensor: tensor.detach().cpu(), x)

    os.makedirs(path, exist_ok=True)

    for i in range(bs):
        x_i = optree.tree_map(lambda tensor: tensor[i : i + 1], x)

        fname = os.path.join(path, f"{batch_idx:06d}-{i:02d}.pt")
        torch.save(x_i, fname)


# Legacy
class FeaturesOnly(Generative):
    def __init__(
        self,
        generator: TrainableBackbone,
        batch_encoder_input_mapping: Dict[str, Any] = "x",
        batch_preprocessing_fn: Callable = trellis_batch_transform_utils.FeaturesOnly.batch_preprocessing,
        validation_dump_type: str = "tensor",
        **kwargs,
    ):

        if "deterministic_sampling" in kwargs:
            warning_cache.warn(
                f"deterministic_sampling should be set in the batch_preprocessing_fn. Setting to the default in that function. {batch_preprocessing_fn}"
            )
            kwargs.pop("deterministic_sampling")

        super().__init__(
            generator=generator,
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=batch_preprocessing_fn,
            validation_dump_type=validation_dump_type,
            **kwargs,
        )

    def _decode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return x
        # return_vals = {"x_shape_latent": x}
        # if "decoder" in self.base_models:
        #     # Assume decoder is for shape
        #     return_vals["x_shape"] = self.base_models["decoder"](x)

        # return return_vals


# Batch transform encodes pose token. No decoder paired for this
class FeaturesAndPose(Generative):
    def __init__(
        self,
        generator: TrainableBackbone,
        batch_encoder_input_mapping: Dict[str, Any] = "x",
        batch_preprocessing_fn: Callable = trellis_batch_transform_utils.FeaturesAndPose.batch_preprocessing,
        validation_dump_type: str = "tensor",
        **kwargs,
    ):
        warning_cache.warn(
            "FeaturesAndPose is deprecated. Use FeaturesAndPoseToken instead."
        )
        if "deterministic_sampling" in kwargs:
            warning_cache.warn(
                f"deterministic_sampling should be set in the batch_preprocessing_fn. Setting to the default in that function. {batch_preprocessing_fn}"
            )
            kwargs.pop("deterministic_sampling")

        super().__init__(
            generator=generator,
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=batch_preprocessing_fn,
            validation_dump_type=validation_dump_type,
            **kwargs,
        )


# Stage 2
class TrellisDINOFeaturesOnly(Generative):
    def __init__(
        self,
        generator: TrainableBackbone,
        batch_encoder_input_mapping: Dict[str, Any] = "x",
        batch_preprocessing_fn: Callable = trellis_batch_transform_utils.FeaturesOnly.batch_preprocessing,
        validation_dump_type: str = "tensor",
        **kwargs,
    ):
        if "deterministic_sampling" in kwargs:
            warning_cache.warn(
                f"deterministic_sampling should be set in the batch_preprocessing_fn. Setting to the default in that function. {batch_preprocessing_fn}"
            )
            kwargs.pop("deterministic_sampling")

        super().__init__(
            generator=generator,
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=batch_preprocessing_fn,
            validation_dump_type=validation_dump_type,
            **kwargs,
        )


class FeaturesAndPoseToken(Generative):
    def __init__(
        self,
        generator: TrainableBackbone,
        batch_encoder_input_mapping: Dict[str, Any] = {
            "shape_latent": "x_shape_latent",
            "instance_quaternion": "instance_quaternion_l2c",
            "instance_translation": "instance_position_l2c",
            "instance_scale": "instance_scale_l2c",
            "scene_scale": "pointmap_scale",
            "scene_shift": "pointmap_shift",
        },
        batch_preprocessing_fn: Optional[Callable] = None,
        validation_dump_type: str = "tensor",
        pose_target_convention: str = "ApparentSize",
        keep_preds: Sequence[str] = (
            "shape",
            "quaternion",
            "translation",
            "scale",
            "translation_scale",
        ),
        **kwargs,
    ):
        if batch_preprocessing_fn is None:
            batch_preprocessing_fn = FeaturesAndPoseToken._prepare_training_targets

        if "deterministic_sampling" in kwargs:
            warning_cache.warn(
                f"deterministic_sampling should be set in the batch_preprocessing_fn. Setting to the default in that function. {batch_preprocessing_fn}"
            )
            kwargs.pop("deterministic_sampling")

        self.pose_target_convention = pose_target_convention
        self.keep_preds = keep_preds
        super().__init__(
            generator=generator,
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=batch_preprocessing_fn,
            validation_dump_type=validation_dump_type,
            **kwargs,
        )

    @staticmethod
    def _prepare_training_targets(
        batch,
        deterministic_sampling=False,
        latent_scale_factor=1.0,
    ):
        batch = extract_data(batch)

        # Shape prediction
        mean = batch["mean"]
        logvar = batch["logvar"]
        shape_posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic_sampling
        )
        x_shape_latent = shape_posterior.sample() * latent_scale_factor
        batch["x_shape_latent"] = x_shape_latent
        return batch

    def _encode(
        self,
        shape_latent: torch.Tensor,
        instance_quaternion: torch.Tensor,
        instance_translation: torch.Tensor,
        instance_scale: torch.Tensor,
        scene_scale: torch.Tensor,
        scene_shift: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if "encoder" in self.base_models:
            # Assume encoder is for shape
            raise NotImplementedError(
                "Using learned encoder is not implemented in FeaturesAndPose"
            )

        instance_pose = InstancePose(
            instance_scale_l2c=instance_scale,
            instance_position_l2c=instance_translation,
            instance_quaternion_l2c=instance_quaternion,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )
        pose_target = PoseTargetConverter.instance_pose_to_pose_target(
            instance_pose,
            pose_target_convention=self.pose_target_convention,
        )
        return_dict = {
            "shape": shape_latent,
            "quaternion": pose_target.x_instance_rotation,
            "translation": pose_target.x_instance_translation,
            "scale": torch.log(pose_target.x_instance_scale),
            "translation_scale": torch.log(pose_target.x_translation_scale),
        }

        return {k: v for k, v in return_dict.items() if k in self.keep_preds}

    def _decode(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decodes pose token and shape latent into instance pose components
        Only used in predict_step and validation_step.
        """

        key_mapping = {
            "shape": "x_shape_latent",
            "quaternion": "x_instance_rotation",
            "translation": "x_instance_translation",
            "scale": "x_instance_scale",
            "translation_scale": "x_translation_scale",
        }

        # Decodes for metrics
        return_vals = {}
        for k, v in x.items():
            return_vals[key_mapping.get(k, k)] = v

        if "x_instance_scale" in return_vals:
            return_vals["x_instance_scale"] = torch.exp(return_vals["x_instance_scale"])

        if "x_translation_scale" in return_vals:
            return_vals["x_translation_scale"] = torch.exp(
                return_vals["x_translation_scale"]
            )

        if "decoder" in self.base_models:
            x_shape_latent = x["shape"]
            # Assume decoder is for shape
            return_vals["x_shape"] = self.base_models["decoder"](x_shape_latent)

        self._add_pose_target_convention(return_vals)
        return return_vals

    def _add_pose_target_convention(self, return_vals):
        bs = Generative._get_batch_size(return_vals)
        return_vals["pose_target_convention"] = [self.pose_target_convention] * bs


# TODO Hao & Bowen: this class needs significant work!
# We allow merging now to unblock MoT
class FeaturesAndPoseTokenWithNormalization(FeaturesAndPoseToken):
    def __init__(
        self,
        generator: TrainableBackbone,
        batch_encoder_input_mapping: Dict[str, Any] = {
            "shape_latent": "x_shape_latent",
            "instance_quaternion": "instance_quaternion_l2c",
            "instance_translation": "instance_position_l2c",
            "instance_scale": "instance_scale_l2c",
            "scene_scale": "pointmap_scale",
            "scene_shift": "pointmap_shift",
        },
        batch_preprocessing_fn: Optional[Callable] = None,
        validation_dump_type: str = "tensor",
        pose_target_convention: str = "ApparentSize",
        keep_preds: Sequence[str] = (
            "shape",
            "quaternion",
            "translation",
            "scale",
            "translation_scale",
            "6drotation",
            "6drotation_normalized",
        ),
        normalize: bool = False,
        pose_target_normalizers: Dict[str, LogScaleShiftNormalizer] = {
            "scale": LogScaleShiftNormalizer(),
            "translation_scale": LogScaleShiftNormalizer(),
        },
        **kwargs,
    ):
        super().__init__(
            generator,
            batch_encoder_input_mapping=batch_encoder_input_mapping,
            batch_preprocessing_fn=batch_preprocessing_fn,
            validation_dump_type=validation_dump_type,
            pose_target_convention=pose_target_convention,
            keep_preds=keep_preds,
            **kwargs,
        )
        self.normalize = normalize
        self.rotation_6d_mean = torch.tensor(
            [
                -0.06366084883674913,
                0.008438224692279752,
                0.00017084786438302483,
                0.0007126610473540038,
                -0.0030916726538816417,
                0.5166093753457688,
            ],
            device=self.device,
        ).to(torch.float32)
        self.rotation_6d_std = torch.tensor(
            [
                0.6656971967514863,
                0.6787012271867754,
                0.30345010594844524,
                0.4394504420678794,
                0.39817973931717104,
                0.6176286868761914,
            ],
            device=self.device,
        ).to(torch.float32)

        self.pose_target_normalizers = pose_target_normalizers

    def _convert_to_rotation6d(self, quaternion: torch.Tensor, normalize: bool = False):
        """
        Convert a quaternion to a 6D rotation representation.

        This converts quaternions to rotation matrices and then takes the first two columns
        of the rotation matrix as the 6D representation. This representation is continuous
        and avoids discontinuities common in other rotation representations.

        Args:
            quaternion: Tensor of shape (..., 4) containing quaternions in (w, x, y, z) format
            normalize: Whether to normalize the 6D rotation representation
        Returns:
            Tensor of shape (..., 6) containing the 6D rotation representation
        """
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_matrix(quaternion)  # Shape: (..., 3, 3)

        # Extract first two columns and flatten
        col1 = rotation_matrix[..., :, 0]  # Shape: (..., 3)
        col2 = rotation_matrix[..., :, 1]  # Shape: (..., 3)

        # Concatenate to get 6D representation
        rotation_6d = torch.cat([col1, col2], dim=-1)  # Shape: (..., 6)

        if normalize:
            if self.rotation_6d_mean.device != rotation_6d.device:
                self.rotation_6d_mean = self.rotation_6d_mean.to(rotation_6d.device)
                self.rotation_6d_std = self.rotation_6d_std.to(rotation_6d.device)
            rotation_6d = (rotation_6d - self.rotation_6d_mean) / self.rotation_6d_std

        return rotation_6d

    def _encode(
        self,
        shape_latent: torch.Tensor,
        instance_quaternion: torch.Tensor,
        instance_translation: torch.Tensor,
        instance_scale: torch.Tensor,
        scene_scale: torch.Tensor,
        scene_shift: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if "encoder" in self.base_models:
            # Assume encoder is for shape
            raise NotImplementedError(
                "Using learned encoder is not implemented in FeaturesAndPose"
            )

        instance_pose = InstancePose(
            instance_scale_l2c=instance_scale,
            instance_position_l2c=instance_translation,
            instance_quaternion_l2c=instance_quaternion,
            scene_scale=scene_scale,
            scene_shift=scene_shift,
        )
        pose_target = PoseTargetConverter.instance_pose_to_pose_target(
            instance_pose,
            pose_target_convention=self.pose_target_convention,
            normalize=self.normalize,
        )
        return_dict = {
            "shape": shape_latent,
            "quaternion": pose_target.x_instance_rotation,
            "translation": pose_target.x_instance_translation,
            "scale": (
                self.pose_target_normalizers["scale"].normalize(
                    pose_target.x_instance_scale
                )
                if not self.normalize
                else pose_target.x_instance_scale
            ),
            "translation_scale": (
                self.pose_target_normalizers["translation_scale"].normalize(
                    pose_target.x_translation_scale
                )
                if not self.normalize
                else pose_target.x_translation_scale
            ),
            "6drotation": self._convert_to_rotation6d(pose_target.x_instance_rotation),
        }

        if "6drotation_normalized" in self.keep_preds:
            return_dict["6drotation_normalized"] = self._convert_to_rotation6d(
                pose_target.x_instance_rotation, normalize=True
            )

        return {k: v for k, v in return_dict.items() if k in self.keep_preds}

    def _decode(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decodes pose token and shape latent into instance pose components
        Only used in predict_step and validation_step.
        """

        key_mapping = {
            "shape": "x_shape_latent",
            "quaternion": "x_instance_rotation",
            "6drotation": "x_instance_rotation_6d",
            "6drotation_normalized": "x_instance_rotation_6d_normalized",
            "translation": "x_instance_translation",
            "scale": "x_instance_scale",
            "translation_scale": "x_translation_scale",
        }

        # Decodes for metrics
        return_vals = {}
        for k, v in x.items():
            return_vals[key_mapping.get(k, k)] = v

        # Convert 6D rotation to quaternion if needed
        if ("x_instance_rotation_6d" in return_vals) or (
            "x_instance_rotation_6d_normalized" in return_vals
        ):
            # Extract the two 3D vectors
            if "x_instance_rotation_6d_normalized" in return_vals:
                if (
                    self.rotation_6d_mean.device
                    != return_vals["x_instance_rotation_6d_normalized"].device
                ):
                    self.rotation_6d_mean = self.rotation_6d_mean.to(
                        return_vals["x_instance_rotation_6d_normalized"].device
                    )
                    self.rotation_6d_std = self.rotation_6d_std.to(
                        return_vals["x_instance_rotation_6d_normalized"].device
                    )
                rot_6d = (
                    return_vals["x_instance_rotation_6d_normalized"]
                    * self.rotation_6d_std
                    + self.rotation_6d_mean
                )
            else:
                rot_6d = return_vals["x_instance_rotation_6d"]

            return_vals["x_instance_rotation"] = self._convert_to_quaternion(rot_6d)

        if "x_instance_scale" in return_vals:
            return_vals["x_instance_scale"] = (
                self.pose_target_normalizers["scale"].denormalize(
                    return_vals["x_instance_scale"]
                )
                if not self.normalize
                else return_vals["x_instance_scale"]
            )
            orig = return_vals["x_instance_scale"]
            return_vals["x_instance_scale"] = (
                return_vals["x_instance_scale"]
                .mean(dim=-1, keepdim=True)
                .expand_as(orig)
            )

        if "x_translation_scale" in return_vals:
            return_vals["x_translation_scale"] = (
                self.pose_target_normalizers["translation_scale"].denormalize(
                    return_vals["x_translation_scale"]
                )
                if not self.normalize
                else return_vals["x_translation_scale"]
            )

        if "decoder" in self.base_models:
            x_shape_latent = x["shape"]
            # Assume decoder is for shape
            return_vals["x_shape"] = self.base_models["decoder"](x_shape_latent)

        bs = Generative._get_batch_size(return_vals)
        self._add_pose_target_convention(return_vals)
        return_vals["pose_normalize"] = [self.normalize] * bs
        return return_vals

    def _convert_to_quaternion(self, rot_6d: torch.Tensor):
        a1 = rot_6d[..., 0:3]
        a2 = rot_6d[..., 3:6]

        # Normalize first vector
        b1 = torch.nn.functional.normalize(a1, dim=-1)

        # Make second vector orthogonal to first
        b2 = a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
        b2 = torch.nn.functional.normalize(b2, dim=-1)

        # Compute third vector as cross product
        b3 = torch.cross(b1, b2, dim=-1)

        # Stack to create rotation matrix
        rotation_matrix = torch.stack([b1, b2, b3], dim=-1)

        # Convert to quaternion
        quaternion = matrix_to_quaternion(rotation_matrix)

        return quaternion

    def _step(self, label, batch, batch_idx):
        # skip if batch happens to be empty
        if batch is None:
            return None

        # preprocess
        batch = self._batch_preprocessing_fn(batch)

        # extract inputs from batch
        encoder_args, encoder_kwargs = self.encoder_extractor_fn(batch)
        condition_args, condition_kwargs = self.condition_extractor_fn(batch)

        # encode
        x = self._encode(*encoder_args, **encoder_kwargs)
        batch_size = Generative._get_batch_size(x)

        embedded_cond, condition_args, condition_kwargs = self._embed_condition(
            *condition_args, **condition_kwargs
        )
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        # TODO Bowen & Hao: need to make sure how we pass in the decoder
        decoder = self.base_models["decoder"] if "decoder" in self.base_models else None
        condition_args = (decoder,) + condition_args

        # compute predictions
        loss, detail_losses = self.base_models["generator"].loss(
            x,
            *condition_args,
            **condition_kwargs,
        )

        self.log(
            name=f"{label}/loss",
            value=loss,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        # Log detailed losses if available
        if detail_losses is not None:
            for loss_name, loss_value in detail_losses.items():
                self.log(
                    name=f"{label}/{loss_name}",
                    value=loss_value,
                    prog_bar=False,
                    batch_size=batch_size,
                    sync_dist=True,
                )

        return loss, batch_size
