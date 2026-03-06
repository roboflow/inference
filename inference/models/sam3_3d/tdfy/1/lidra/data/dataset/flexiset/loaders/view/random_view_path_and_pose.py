import os
import random

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.dataset.tdfy.trellis.pose_loader import load_trellis_pose

# COMMENT(Pierre) : Super ugly but keep for backwards compatibility. Will have to change.


class RandomViewPathAndPose(Base):
    def __init__(self, pose_loader=load_trellis_pose):
        super().__init__()
        self._pose_loader = pose_loader

    def _get_available_views(self, image_dir):
        available_views = [
            f
            for f in os.listdir(image_dir)
            if f.endswith(".png") and os.path.isfile(os.path.join(image_dir, f))
        ]

        # TODO: weiyaowang clean this up once we standardize our data format
        if len(available_views) == 0:
            available_views = [
                f
                for f in os.listdir(image_dir)
                if f.endswith("_rgb0001.jpg")
                and os.path.isfile(os.path.join(image_dir, f))
            ]
        assert len(available_views) > 0
        return available_views

    def _get_available_poses(self, image_dir):
        pose_json_path = os.path.join(image_dir, "transforms.json")
        return self._pose_loader(pose_json_path)

    def _load(self, path, uid):
        image_dir = os.path.join(path, uid)

        available_views = self._get_available_views(image_dir)
        available_poses = self._get_available_poses(image_dir)

        selected_view = random.choice(available_views)
        selected_pose = available_poses[selected_view]
        return (
            os.path.join(image_dir, selected_view),
            selected_pose["instance_quaternion_l2c"],
            selected_pose["instance_position_l2c"],
            selected_pose["instance_scale_l2c"],
            selected_pose["camera_K"],
        )
