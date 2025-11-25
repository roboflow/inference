import uuid
import contextlib
from copy import deepcopy

from lidra.data.dataset.flexiset.loaders.base import Base
from lidra.data.rendering.blender import (
    Blender as BlenderRenderer,
    atomic_sequential_changes,
    default_eevee_settings,
    default_cycles_settings,
    default_rendering_settings,
    add_camera,
    add_light,
    normalize_scene,
    load_model,
    enable_nodes,
    output_rgba,
    update_camera,
)


# from Trellis
def get_transform_matrix(obj):
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix


class Blender(Base):
    VALID_ENGINES = {"BLENDER_EEVEE_NEXT", "CYCLES"}

    def __init__(self, engine="BLENDER_EEVEE_NEXT"):
        super().__init__()
        self._engine = engine
        self._blender = None

        if not self._engine in self.VALID_ENGINES:
            raise ValueError(
                f"Invalid rendering engine: {self._engine}. "
                f"Valid engines are: {self.VALID_ENGINES}"
            )

    def _set_settings(self):
        self._blender.apply(
            atomic_sequential_changes(
                default_rendering_settings(
                    self._engine,
                    resolution=1024,  # for parity with disk-saved images
                ),
                default_eevee_settings(),
                default_cycles_settings(geo_mode=False),
            )
        )

    def _set_outputs(self):
        self._blender.apply(
            atomic_sequential_changes(
                enable_nodes(),
                output_rgba(),
            )
        )

    def _set_lights(self):
        self._blender.apply(
            atomic_sequential_changes(
                add_light(
                    "default_light",
                    (4, 1, 6),
                    energy=1000,
                    type="POINT",
                ),
                add_light(
                    "top_light",
                    (0, 0, 10),
                    energy=10000,
                    scale=(100, 100, 100),
                    type="AREA",
                ),
                add_light(
                    "bottom_light",
                    (0, 0, -10),
                    energy=1000,
                    type="AREA",
                ),
            )
        )

    def _maybe_init_blender(self):
        if self._blender is not None:
            return
        self._blender = BlenderRenderer()
        self._set_settings()
        self._set_outputs()

    @contextlib.contextmanager
    def _normalized_model(self, filepath):
        # load model file and normalize
        model_name = f"model_{uuid.uuid4()}"  # model name needs to be unique, otherwise it will be renamed by blender
        self._blender.apply(
            atomic_sequential_changes(
                load_model(filepath, group_as=model_name),
                normalize_scene(model_name),
            )
        )
        self._set_lights()
        self._n_changes = 2
        obj = self._blender.get_object(model_name)
        try:
            # COMMENT(Pierre) : deepcopy required since the buffer will be de-allocated by blender
            # leads to silent crash otherwise
            yield deepcopy(obj.scale), deepcopy(obj.matrix_world.translation)

        finally:
            self._blender.remove_last_change(self._n_changes)  # remove model loading

    def _render_view(self, view):
        # add camera
        if not len(self._blender.data.cameras) > 0:
            self._blender.apply(add_camera(view, name="cam"))
            self._n_changes += 1
        else:
            camera = self._blender.objects["cam"]
            update_camera(camera, view)

        # render
        result = self._blender.render()
        image = result["out/rgba0001.png"]

        # reverts last two changes
        cam = self._blender.objects["cam"]
        cam_matrix = get_transform_matrix(cam)

        return image, cam_matrix

    def _load(self, path, views):
        self._maybe_init_blender()
        with self._normalized_model(path) as (scale, offset):
            images_and_cam_matrices = tuple(self._render_view(view) for view in views)
            images = tuple(iacm[0] for iacm in images_and_cam_matrices)
            cam_matrices = tuple(iacm[1] for iacm in images_and_cam_matrices)
        return images, cam_matrices, scale, offset
