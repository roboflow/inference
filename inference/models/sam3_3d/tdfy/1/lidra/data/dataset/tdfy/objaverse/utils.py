import numpy as np
import Imath
import OpenEXR
import torch
import pytorch3d


def blender2pytorch3d(R, T):
    R = R.clone().float()
    T = T.clone().float()

    T_world = -R.T @ T
    R = R.T
    R = R @ pytorch3d.transforms.axis_angle_to_matrix(
        torch.FloatTensor([0, torch.pi, 0])
    )
    T = -R.T @ T_world

    return R, T[None]


# revised to ensure cycle consistency now
def pytorch3d2blender(R, T):
    R = R.clone().float()
    T = T.clone().float()

    T_world = -R @ T
    R = R @ pytorch3d.transforms.axis_angle_to_matrix(
        torch.FloatTensor([0, torch.pi, 0])
    ).transpose(-1, -2)
    R = R.T
    T = -R @ T_world

    return R, T[None]


def read_depth_channel(exr_path):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(exr_path)
    # Define the pixel type
    header = exr_file.header()
    # print(header['channels'])
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    # Get the data window
    dw = exr_file.header()["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Read the depth channel
    channels = header["channels"]
    depth_channel_name = "R" if "R" in channels else "V"
    depth_str = exr_file.channel(depth_channel_name, pt)

    # Convert the string to a NumPy array
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth.shape = (size[1], size[0])  # Reshape the array to the correct image size
    # Close the file
    exr_file.close()
    return depth


def get_cam_transform(R, T):
    B = R.shape[0]
    assert R.shape[0] == T.shape[0]

    res = torch.cat([R, T.unsqueeze(1)], dim=1)
    c = (
        torch.FloatTensor([0, 0, 0, 1])
        .to(R.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(B, 1, 1)
    )

    return torch.cat([res, c], dim=2)


def get_relative_cam_transform(cam1, cam2):
    """
    Find relative pose from cam1 -> cam2
    camera transform in the shape of
    [R 0
     T 1]
    cam1: [1, 4, 4]
    cam2: [B, 4, 4]
    """
    # [B, 3, 3]
    assert len(cam1.shape) == len(cam2.shape) == 3

    B = cam2.shape[0]
    if cam1.shape[0] != B:
        assert cam1.shape[0] == 1
        cam1 = cam1.repeat(B, 1, 1)

    return torch.inverse(cam1) @ cam2
