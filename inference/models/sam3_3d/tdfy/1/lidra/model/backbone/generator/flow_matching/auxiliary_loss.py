from functools import partial
import os
import torch
from lidra.utils.visualization.object_pointcloud import save_points_to_ply
from lidra.metrics.tdfy.metric_collection_per_sample import TdfyPerSample
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.loss import chamfer_distance
from lidra.model.backbone.extensions.sinkhorn.sinkhorn import sinkhorn
from lidra.data.utils import right_broadcasting
from lidra.data.utils import tree_tensor_map


class PointCloudLoss(torch.nn.Module):
    def __init__(self, cd_weight, emd_weight):
        self.cd_weight = cd_weight
        self.emd_weight = emd_weight
        self.step = 0
        self.pointclouds_dir = "/home/bowenzhang/output/pointclouds"

    def _predict_x1_from_xt_and_v_tensor(
        self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor
    ):
        tb = right_broadcasting(t.to(x_t.device), x_t)
        sigma_term = 1 - self.sigma_min

        # x1 = v * (1 - (1 - σ) * t) + (1 - σ) * x_t
        predicted_x1 = v * (1 - sigma_term * tb) + sigma_term * x_t

        return predicted_x1

    def _predict_x1_from_xt_and_v(self, x_t, v, t):
        return tree_tensor_map(
            partial(self._predict_x1_from_xt_and_v_tensor, t=t),
            x_t,
            v,
        )

    def forward(self, prediction, target, x_t, t, decoder):
        self.step += 1
        # additional point cloud matching loss
        # TODO: perharps need padding since the point cloud is not the same size
        if self.cd_weight > 0.0 or self.emd_weight > 0.0:
            assert decoder is not None, "decoder is required for emd loss"

            prediction_x1 = self._predict_x1_from_xt_and_v(x_t, prediction, t)
            prediction_x1_decoded = decoder(prediction_x1["shape"])
            target_decoded = decoder(target["shape"])

            R_gt = quaternion_to_matrix(target["quaternion"]).squeeze()
            R_pred = quaternion_to_matrix(prediction["quaternion"]).squeeze()

            # Obtain point clouds from predicted and target volumes
            pred_points_list = []
            gt_points_list = []
            for i in range(prediction_x1_decoded.shape[0]):
                predicted_volume = {"occupancy_volume": prediction_x1_decoded[i]}
                target_volume = {"occupancy_volume": target_decoded[i]}
                pred_points = TdfyPerSample._get_points(
                    predicted_volume, logit_threshold=0.0
                )
                gt_points = TdfyPerSample._get_points(
                    target_volume, logit_threshold=0.0
                )
                pred_points = pred_points @ R_pred[i]
                gt_points = gt_points @ R_gt[i]
                if pred_points.shape[0] == 0 or gt_points.shape[0] == 0:
                    continue
                pred_points_list.append(pred_points.clip(-1, 1))
                gt_points_list.append(gt_points.clip(-1, 1))

            if len(pred_points_list) == 0 or len(gt_points_list) == 0:
                return loss

            cd_loss = 0.0
            emd_loss = 0.0
            if self.emd_weight > 0:
                for i in range(len(pred_points_list)):
                    pred_points_i = pred_points_list[i].contiguous()
                    gt_points_i = gt_points_list[i].contiguous()

                    emd_loss_batch, _, _ = sinkhorn(
                        pred_points_i,
                        gt_points_i,
                        p=2,
                        eps=1e-3,
                        max_iters=50,
                        stop_thresh=0.05,
                    )
                    emd_loss_batch = torch.sqrt(emd_loss_batch).mean()
                    emd_loss += emd_loss_batch * self.emd_weight

                loss += emd_loss / len(pred_points_list)

            if self.cd_weight > 0:
                torch.use_deterministic_algorithms(True, warn_only=True)
                device = pred_points_list[0].device
                pred_lengths = torch.tensor(
                    [p.shape[0] for p in pred_points_list], device=device
                )
                gt_lengths = torch.tensor(
                    [p.shape[0] for p in gt_points_list], device=device
                )

                pred_points_padded = torch.nn.utils.rnn.pad_sequence(
                    pred_points_list, batch_first=True, padding_value=0.0
                )
                gt_points_padded = torch.nn.utils.rnn.pad_sequence(
                    gt_points_list, batch_first=True, padding_value=0.0
                )

                cd_loss_batch, _ = chamfer_distance(
                    pred_points_padded,
                    gt_points_padded,
                    x_lengths=pred_lengths,
                    y_lengths=gt_lengths,
                    batch_reduction="mean",
                    point_reduction="mean",
                )
                cd_loss = cd_loss_batch * self.cd_weight
                loss += cd_loss

            print(
                "diffusion loss: {:.4f}, cd loss: {:.4f}, emd loss: {:.4f}".format(
                    loss, cd_loss, emd_loss / len(pred_points_list)
                )
            )

            # Save point clouds to PLY files if enabled
            if self.step % 1000 == 0:
                print("saving point clouds")
                batch_idx = i
                iter_count = self.step

                # Save predicted points
                pred_filename = os.path.join(
                    self.pointclouds_dir,
                    f"pred_points_iter{iter_count}_batch{batch_idx}.ply",
                )
                save_points_to_ply(pred_points.squeeze().detach().cpu(), pred_filename)

                # Save ground truth points
                gt_filename = os.path.join(
                    self.pointclouds_dir,
                    f"gt_points_iter{iter_count}_batch{batch_idx}.ply",
                )
                save_points_to_ply(gt_points.squeeze().detach().cpu(), gt_filename)

                print(
                    f"Saved point clouds for batch {batch_idx} to {pred_filename} and {gt_filename}"
                )
