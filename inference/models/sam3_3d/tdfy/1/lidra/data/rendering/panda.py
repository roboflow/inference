import gc

import numpy as np
from skimage import io
from panda3d.core import (
    Texture,
    GraphicsOutput,
    GraphicsPipe,
    FrameBufferProperties,
    WindowProperties,
    LoaderOptions,
    ModelPool,
    GeomVertexReader,
    GeomVertexData,
    LVecBase3f,
    PerspectiveLens,
    PointLight,
)

import lidra.utils.singleton as singleton
from lidra.profiler.timeit import timeit

ENABLE_PBR = True
DEBUG_MODE = False


def init_panda3d():
    from panda3d.core import load_prc_file_data

    load_prc_file_data("", "audio-library-name null")
    load_prc_file_data("", "load-display p3headlessgl")

    if DEBUG_MODE:
        load_prc_file_data("", "notify-level-glgsg debug")
        load_prc_file_data("", "notify-level-shader spam")
        load_prc_file_data("", "gl-debug #t")
        load_prc_file_data("", "track-memory-usage #t")

    load_prc_file_data("", "gltf-skip-animations true")
    load_prc_file_data("", "gltf-skip-axis-conversion true")

    from direct.showbase.ShowBase import ShowBase

    if ENABLE_PBR:
        import simplepbr
        import gltf

        gltf.GltfSettings.skip_animations = True

    show_base = ShowBase(fStartDirect=True, windowType="offscreen")
    if ENABLE_PBR:
        show_base.pipeline = simplepbr.init(
            use_normal_maps=True,
            use_emission_maps=True,
            use_occlusion_maps=True,
        )

    return show_base


singleton.register(
    "panda3d.show_base",
    init_fn=init_panda3d,
)


class Panda3D:

    @staticmethod
    def _convert_xyz_from_blender(x, y, z):
        return (x, z, -y)

    def __init__(self):
        self._panda3d = singleton.get("panda3d.show_base")

        self.model = None
        self.pbr = ENABLE_PBR

        fb_prop = FrameBufferProperties()
        fb_prop.set_rgb_color(True)
        fb_prop.set_rgba_bits(8, 8, 8, 8)
        fb_prop.set_depth_bits(24)
        fb_prop.set_multisamples(4)

        win_prop = WindowProperties.size(1024, 1024)
        self._panda3d.win = self._panda3d.graphics_engine.make_output(
            self._panda3d.pipe,
            "cameraview",
            0,
            fb_prop,
            win_prop,
            GraphicsPipe.BFRefuseWindow,
        )

        disp_region = self._panda3d.win.make_display_region()
        disp_region.set_camera(self._panda3d.cam)

        # texture where the view will be rendered
        self.texture = Texture()
        self._panda3d.win.add_render_texture(
            self.texture,
            GraphicsOutput.RTMCopyRam,
            GraphicsOutput.RTPColor,
        )

    def load_model(self, model_path):
        if self.model is not None:
            self._panda3d.loader.unload_model(self.model)
            self.model.remove_node()

            # attempt at controlling the VRAM leak
            self.model = None
            ModelPool.release_all_models()
            ModelPool.garbage_collect()
            gc.collect()

        loaded_options = LoaderOptions()
        self.model = self._panda3d.loader.load_model(
            model_path,
            loaderOptions=loaded_options,
        )
        self.model.reparent_to(self._panda3d.render)

        if self.pbr:
            self.model.set_shader_input(
                "camera_world_position",
                self._panda3d.cam.get_pos(),
            )

        # normalize
        self.normalize()

    def get_bounds(self):

        OLD_VERSION = False
        if OLD_VERSION:
            # BUG(Pierre) get_tight_bounds() returns garbage when node has no matrix
            bottom_left, top_right = self.model.get_tight_bounds()
        else:
            # custom made bbox calculation (slow because in python)
            inf = float("inf")
            bottom_left = LVecBase3f(inf, inf, inf)
            top_right = LVecBase3f(-inf, -inf, -inf)

            geom_node_collection = self.model.find_all_matches("**/+GeomNode")
            for geom_node in geom_node_collection:
                world_mat = geom_node.get_transform(self._panda3d.render).get_mat()
                geom_node = geom_node.node()

                for i in range(geom_node.get_num_geoms()):
                    geom = geom_node.get_geom(i)
                    vdata = geom.get_vertex_data()
                    # make mutable copy to prepare for in-place transform
                    vdata = GeomVertexData(vdata)
                    vdata.transform_vertices(world_mat)
                    vertex_reader = GeomVertexReader(vdata, "vertex")
                    while not vertex_reader.isAtEnd():
                        vertex = vertex_reader.get_data3()
                        bottom_left = bottom_left.fmin(vertex)
                        top_right = top_right.fmax(vertex)
        return bottom_left, top_right

    def normalize(self):
        bottom_left, top_right = self.get_bounds()
        scale = max(top_right - bottom_left)
        center = (bottom_left + top_right) / (2 * scale)
        self.model.set_scale(1 / scale)
        self.model.set_pos(-center)

    def set_camera(self, view):
        cx = view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"])
        cy = view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"])
        cz = view["radius"] * np.sin(view["pitch"])

        self._panda3d.cam.setPos(cx, cy, cz)

        self._panda3d.cam.lookAt((0, 0, 0))

        self._panda3d.cam.node().setLens(PerspectiveLens())
        self._panda3d.cam.node().getLens().setFov(view["fov"] / (3.14) * 180)
        self._panda3d.cam.node().getLens().setNearFar(0.01, 200)

    def add_light(self, name, position, color=(1, 1, 1, 1)):
        plight = PointLight(name)
        plight.setColor(color)
        plnp = self._panda3d.render.attachNewNode(plight)
        position = Panda3D._convert_xyz_from_blender(*position)
        plnp.setPos(*position)
        self._panda3d.render.setLight(plnp)

    def set_lights(self):
        # REMARK(Pierre) hardcoded, but will do for now
        if self.pbr:
            color = (2.0,) * 4
        else:
            color = (0.5,) * 4
        self.add_light("light", (4, 1, 6), color=color)
        for x in range(-1, 4):
            for y in range(-1, 4):
                self.add_light(
                    f"light_top_{x}_{y}",
                    (x, y, 10),
                    color=tuple(c * 10 for c in color),
                )
                self.add_light(f"light_bottom_{x}_{y}", (x, y, -10), color=color)

    def render(self):
        self._panda3d.graphicsEngine.renderFrame()
        return self._get_pixels()

    def _get_pixels(self):
        raw_data = self.texture.getRamImage()
        image = np.frombuffer(raw_data, dtype=np.uint8)
        shape = (
            self.texture.getYSize(),
            self.texture.getXSize(),
            self.texture.getNumComponents(),
        )
        image = image.reshape(shape)
        # flip vertically (Panda3D's coordinate origin is bottom-left)
        image = np.flipud(image)
        bgr = image[..., :-1]
        rgb = bgr[..., ::-1]
        alpha = image[..., -1:]
        image = np.concatenate((rgb, alpha), axis=-1)
        return image


def local_debug_test():
    view = {
        "yaw": -1.57,  # 8.329076232502526,
        "pitch": 1.57,  # 0.0,  # -0.7618072284012608,
        "radius": 2,
        "fov": 0.6981317007977318,
    }

    MODEL_PATH = "/checkpoint/3dfy/shared/datasets/trellis500k/ABO/raw/3dmodels/original/0/B00XBC3BF0.glb"

    with timeit(level="WARNING", message_template="rendering time {duration}"):
        renderer = Panda3D()
        renderer.set_lights()
        renderer.set_camera(view)
        renderer.load_model(MODEL_PATH)
        img = renderer.render()

    with timeit(level="WARNING", message_template="rendering time {duration}"):
        renderer.load_model(MODEL_PATH)
        img = renderer.render()

    io.imsave("output-panda3d.png", img)

    print("done")


if __name__ == "__main__":
    local_debug_test()
