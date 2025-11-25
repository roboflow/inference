from .base import GenerativePreprocessing

import torch
from pytorch3d.renderer.cameras import FoVPerspectiveCameras

from lidra.model.backbone.autoencoder.vae import DiagonalGaussianDistribution
from lidra.model.module.tdfy.util import camera_encoding

from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn

warning_cache = WarningCache()


class FeaturesOnly(GenerativePreprocessing):
    batch_encoder_input_mapping = "x"

    @staticmethod
    def batch_preprocessing(
        batch,
        deterministic_sampling=False,
        latent_scale_factor=1.0,
        latent_prior_shape=(4096, 8),
    ):
        batch = batch[1]
        if "mean" not in batch or "logvar" not in batch:
            warning_cache.warn(
                f"No 'mean' or 'logvar' found in batch, using standard normal with shape {latent_prior_shape=}"
            )
            batch["x"] = torch.randn(
                batch["image"].shape[0],
                *latent_prior_shape,
                device=batch["image"].device,
            )
            return batch

        posterior = DiagonalGaussianDistribution(
            mean=batch["mean"],
            logvar=batch["logvar"],
            deterministic=deterministic_sampling,
        )
        batch["x"] = posterior.sample() * latent_scale_factor
        return batch


class FeaturesAndPose(GenerativePreprocessing):
    batch_encoder_input_mapping = "x"

    @staticmethod
    def batch_preprocessing(
        batch,
        deterministic_sampling=False,
        latent_scale_factor=1.0,
        pose_std=None,
    ):
        batch = batch[1]

        mean = batch["mean"]
        logvar = batch["logvar"]
        posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic_sampling
        )
        camera_T = batch["camera_T"]
        camera_R = batch["camera_R"]
        cameras = FoVPerspectiveCameras(R=camera_R, T=camera_T, device=camera_R.device)
        camera_embedding = camera_encoding(cameras)
        x = posterior.sample() * latent_scale_factor
        camera_embedding = torch.nn.functional.pad(
            camera_embedding, [0, x.shape[-1] - camera_embedding.shape[-1]]
        )
        if pose_std is not None:
            noise = (
                torch.randn(camera_embedding.shape, device=camera_embedding.device)
                * pose_std
            )
            camera_embedding += noise
        x = torch.cat([x, camera_embedding.unsqueeze(1)], dim=1)

        batch["x"] = x
        return batch


class TrellisDINOFeaturesOnly(GenerativePreprocessing):
    batch_encoder_input_mapping = "x"
    batch_decoder_input_mapping = "coords"

    @staticmethod
    def batch_preprocessing(
        batch,
        deterministic_sampling=False,
    ):
        batch = batch[1]
        sparse_t = batch["sparse_t"]
        device = batch["image"].device

        mean = torch.concat([st.mean for st in sparse_t], dim=0).to(device)
        logvar = torch.concat([st.logvar for st in sparse_t], dim=0).to(device)
        posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic_sampling
        )
        # add one dimension for psuedo batch; used to handle t
        # one can also do t for the real batch size, but is tricky, require another argument there
        batch["x"] = posterior.sample()[None]

        coords = []
        for i, st in enumerate(sparse_t):
            coord = st.coords
            batch_ind = torch.ones([coord.shape[0], 1], dtype=coord.dtype) * i
            # add the batch index so the sparse tensor is aware of the slice when batched
            coord = torch.concat([batch_ind, coord], dim=1)
            coords.append(coord.type(torch.int32))
        # convert to numpy to avoid checks on whether it's a tensor
        coords = torch.concat(coords, dim=0).cpu().numpy()
        batch["coords"] = coords
        return batch
