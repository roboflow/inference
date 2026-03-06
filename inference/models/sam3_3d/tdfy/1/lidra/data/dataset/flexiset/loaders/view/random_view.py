import numpy as np

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.model.backbone.trellis.utils.random_utils import sphere_hammersley_sequence


# TODO(Pierre) expose relevant parameters when needed
class RandomViews(Base):
    def __init__(self):
        super().__init__()

    def _get_random_view(self):
        # generate random view in same format as Trellis
        # https://github.com/microsoft/TRELLIS/blob/eeacb0bf6a7d25058232d746bef4e5e880b130ff/dataset_toolkits/render_cond.py#L31
        num_views = 24  # TODO expose ?
        fov_min, fov_max = 10, 70

        # get random yaw and pitch
        offset = (np.random.rand(), np.random.rand())
        i = np.random.randint(0, num_views)
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)

        # get radius
        radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
        radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
        k_min = 1 / radius_max**2
        k_max = 1 / radius_min**2

        k = np.random.uniform(k_min, k_max)

        radius = 1 / np.sqrt(k)
        fov = 2 * np.arcsin(np.sqrt(3) / 2 / radius)

        # TODO(Pierre) : will need to convert that Trellis format into a unified format for views / poses
        view = {"yaw": yaw, "pitch": pitch, "radius": radius, "fov": fov}
        return view

    def _load(self, n=1):
        return tuple(self._get_random_view() for _ in range(n))
