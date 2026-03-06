import torch
import utils3d


def load_trellis_voxels(voxel_path, resolution=64):
    position = utils3d.io.read_ply(voxel_path)[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return {"voxels": ss}
