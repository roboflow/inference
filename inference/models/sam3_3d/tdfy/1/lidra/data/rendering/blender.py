import math
import warnings
from loguru import logger
import numpy as np
import os
import skimage


from lidra.profiler.timeit import timeit
from lidra.test.util import temporary_directory
from lidra.utils.io import stdout_redirected
import lidra.utils.singleton as singleton


def init_blender():
    import bpy

    # from mathutils import Vector as MUVector, Matrix as MUMatrix
    # the following imports are ONLY accessible after first importing "bpy"
    import mathutils

    return {"bpy": bpy, "mathutils": mathutils}


singleton.register(
    "blender.modules",
    init_fn=init_blender,
)


class Blender:
    LAST_INSTANCE = None
    IMPORT_FUNCTIONS = {
        "obj": lambda blender: blender.ops.wm.obj_import,
        "glb": lambda blender: blender.ops.import_scene.gltf,
        "gltf": lambda blender: blender.ops.import_scene.gltf,
        "usd": lambda blender: blender.ops.wm.usd_import,
        "usda": lambda blender: blender.ops.wm.usd_import,
        "fbx": lambda blender: blender.ops.import_scene.fbx,
        "stl": lambda blender: blender.ops.wm.stl_import,
        "dae": lambda blender: blender.ops.wm.collada_import,
        "ply": lambda blender: blender.ops.wm.ply_import,
        "abc": lambda blender: blender.ops.wm.alembic_import,
        "blend": lambda blender: blender.ops.wm.append,
    }

    def _tag_as_last_instance(self):
        Blender.LAST_INSTANCE = id(self)

    def _was_last_instance(self):
        return Blender.LAST_INSTANCE == id(self)

    def _reapply(self, change_stack):
        self.reset(remove_data=True)
        for change_fn in change_stack:
            self.apply(change_fn)

    def _check_state_is_current_instance(self):
        if not self._was_last_instance():
            if Blender.LAST_INSTANCE is not None:
                warnings.warn(
                    "multiple instances of blender are being used, this can lead to major slowdowns",
                    RuntimeWarning,
                )
            self._reapply(self._change_stack)

    def reset(self, remove_data=True):
        self.ops.wm.read_factory_settings(use_empty=remove_data)
        self._change_stack = []
        self._tag_as_last_instance()

    def group_objects_as(self, name, object_names, orphan_only):
        if name in self.data.objects:
            raise RuntimeError(f'object with named "{name}" already exist')

        # create an empty object to be used as a parent for all specified objects
        group_node = self.data.objects.new(name, None)
        self.context.scene.collection.objects.link(group_node)
        for obj in object_names:
            if (not orphan_only) or (not self.data.objects[obj].parent):
                self.data.objects[obj].parent = group_node
        return group_node

    def get_object(self, object_name):
        if isinstance(object_name, str):
            return self.data.objects[object_name]
        return object_name

    def get_children(self, object, recursive=True):
        if recursive:
            return self.get_object(object).children_recursive
        return self.get_object(object).children

    def bbox(self, object_name, mesh_only=True):
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False

        objects = [
            self.data.objects[object_name],
        ]
        objects += self.get_children(object_name, recursive=True)
        if mesh_only:
            objects = tuple(
                obj for obj in objects if isinstance(obj.data, self.bpy.types.Mesh)
            )

        for obj in objects:
            found = True
            for coord in obj.bound_box:
                coord = self.mathutils.Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return self.mathutils.Vector(bbox_min), self.mathutils.Vector(bbox_max)

    def __init__(self):
        self.reset(remove_data=True)
        self._is_applying = False

    @property
    def bpy(self):
        return singleton.get("blender.modules")["bpy"]

    @property
    def mathutils(self):
        return singleton.get("blender.modules")["mathutils"]

    @property
    def context(self):
        return self.bpy.context

    @property
    def ops(self):
        return self.bpy.ops

    @property
    def data(self):
        return self.bpy.data

    @property
    def objects(self):
        return self.bpy.data.objects

    @property
    def scenes(self):
        return self.bpy.data.scenes

    @property
    def materials(self):
        return self.bpy.data.materials

    def apply(self, change_fn):
        self._check_state_is_current_instance()
        if self._is_applying:
            raise RuntimeError("cannot apply a change within a change")
        self._is_applying = True
        self._change_stack.append(change_fn)
        try:
            change_fn(self)
        finally:
            self._is_applying = False

    def remove_last_change(self, number_of_changes=1):
        self._reapply(self._change_stack[:-number_of_changes])

    @staticmethod
    def _load_output(filepath):
        _, ext = os.path.splitext(filepath)
        match ext:
            case ".png":
                output = skimage.io.imread(filepath)
            case _:
                raise NotImplementedError("invalid extension found for output file")
        return output

    def render(self):
        output_nodes = [
            node
            for node in self.context.scene.node_tree.nodes
            if isinstance(node, self.bpy.types.CompositorNodeOutputFile)
        ]
        outputs = {}

        # render
        with temporary_directory() as render_path:
            for node in output_nodes:
                node.base_path = render_path + "/" + node.base_path

            with stdout_redirected(file_descriptor_mode=True):
                self.ops.render.render(write_still=False)

            for dirpath, dirnames, filenames in os.walk(render_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    key = os.path.relpath(filepath, render_path)
                    outputs[key] = self._load_output(filepath)

            for node in output_nodes:
                node.base_path = node.base_path[len(render_path) + 1 :]

        return outputs


def atomic_sequential_changes(*changes):
    def change_fn(blender):
        for change in changes:
            change(blender)

    return change_fn


def default_rendering_settings(engine, resolution):
    def change_fn(blender):
        render = blender.context.scene.render
        render.engine = engine
        render.resolution_x = resolution
        render.resolution_y = resolution
        render.resolution_percentage = 100
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGBA"
        render.film_transparent = True

    return change_fn


def default_eevee_settings():
    def change_fn(blender):
        eevee = blender.context.scene.eevee
        eevee.taa_render_samples = 1

        eevee.bokeh_neighbor_max = 0
        eevee.bokeh_max_size = 0

        eevee.use_gtao = False
        eevee.use_raytracing = False
        eevee.use_shadows = False
        eevee.use_taa_reprojection = False
        eevee.use_volume_custom_range = False

        eevee.shadow_pool_size = "16"
        eevee.gi_visibility_resolution = "8"
        eevee.gi_irradiance_pool_size = "16"
        eevee.gi_diffuse_bounces = 1
        eevee.gi_cubemap_resolution = "128"

        eevee.fast_gi_step_count = 1
        eevee.fast_gi_resolution = "16"
        eevee.fast_gi_ray_count = 1

    return change_fn


def default_cycles_settings(geo_mode=False):
    def change_fn(blender):
        blender.context.scene.cycles.device = "GPU"
        blender.context.scene.cycles.samples = 1
        blender.context.scene.cycles.filter_type = "BOX"
        blender.context.scene.cycles.filter_width = 1
        blender.context.scene.cycles.diffuse_bounces = 1
        blender.context.scene.cycles.glossy_bounces = 1
        blender.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
        blender.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
        blender.context.scene.cycles.use_denoising = True

        blender.context.preferences.addons["cycles"].preferences.get_devices()
        blender.context.preferences.addons["cycles"].preferences.compute_device_type = (
            "CUDA"
        )

    return change_fn


def add_light(
    name,
    location=(0.0, 0.0, 0.0),
    energy=10.0,
    rotation_euler=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    type="POINT",
):
    def change_fn(blender):
        light = blender.data.lights.new(name, type=type)
        light = blender.data.objects.new(name, light)
        blender.context.collection.objects.link(light)

        light.data.energy = energy
        light.location = location
        light.rotation_euler = rotation_euler
        light.scale = scale

    return change_fn


def add_camera(
    view,
    name="Camera",
    sensor_width=32,
):
    def change_fn(blender):
        camera = blender.data.cameras.new(name)
        camera = blender.data.objects.new(name, camera)

        blender.context.collection.objects.link(camera)
        blender.context.scene.camera = camera

        camera.data.sensor_height = camera.data.sensor_width = sensor_width

        cam_constraint = camera.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"

        cam_empty = blender.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        blender.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty

        update_camera(camera, view)

        return camera

    return change_fn


def update_camera(camera, view):
    camera.location = (
        view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
        view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
        view["radius"] * np.sin(view["pitch"]),
    )
    camera.data.lens = 16 / np.tan(view["fov"] / 2)


def load_model(path, group_as=None):
    def change_fn(blender):
        file_extension = path.split(".")[-1].lower()
        if file_extension is None:
            raise ValueError(f"Unsupported file type: {path}")

        # load from existing import functions
        import_function = Blender.IMPORT_FUNCTIONS[file_extension](blender)

        logger.info(f"loading object from {path}")
        if group_as is not None:
            object_names = set(blender.data.objects.keys())

        with stdout_redirected(file_descriptor_mode=False):
            if file_extension == "blend":
                import_function(directory=path, link=False)
            elif file_extension in {"glb", "gltf"}:
                import_function(
                    filepath=path,
                    merge_vertices=True,
                    import_shading="NORMALS",
                )
            else:
                import_function(filepath=path)
        if group_as is not None:
            model_objects = set(blender.data.objects.keys()) - object_names
            blender.group_objects_as(group_as, model_objects, orphan_only=True)

    return change_fn


def enable_nodes(discard_existing=True):
    def change_fn(blender):
        blender.context.scene.render.use_compositing = True
        blender.context.scene.use_nodes = True
        tree = blender.context.scene.node_tree

        if discard_existing:
            for n in tree.nodes:
                tree.nodes.remove(n)

        tree.nodes.new("CompositorNodeRLayers")

    return change_fn


def output_rgba(
    render_layer_name="Render Layers",
    base_path="out",
    file_format="PNG",
    file_name="rgba",
):
    def change_fn(blender, antialiasing=False):
        tree = blender.context.scene.node_tree
        links = tree.links
        render_layer = tree.nodes[render_layer_name]

        # create anti-aliasing
        if antialiasing:
            anti_aliasing_node = tree.nodes.new("CompositorNodeAntiAliasing")
            anti_aliasing_node.contrast_limit = 0.5
            anti_aliasing_node.corner_rounding = 0.5
            anti_aliasing_node.threshold = 0.5
            links.new(render_layer.outputs["Image"], anti_aliasing_node.inputs["Image"])
            render_layer = anti_aliasing_node

        # create and link output node
        normal_file_output = tree.nodes.new("CompositorNodeOutputFile")
        normal_file_output.base_path = base_path
        normal_file_output.file_slots["Image"].use_node_format = True
        normal_file_output.file_slots["Image"].path = file_name
        normal_file_output.format.file_format = file_format
        normal_file_output.format.color_mode = "RGBA"
        normal_file_output.format.color_depth = "8"
        links.new(render_layer.outputs["Image"], normal_file_output.inputs["Image"])

    return change_fn


def normalize_scene(object_name):
    def change_fn(blender):
        obj = blender.get_object(object_name)

        assert obj.scale == blender.mathutils.Vector((1.0, 1.0, 1.0))
        assert obj.matrix_world.translation == blender.mathutils.Vector((0.0, 0.0, 0.0))

        # Re-scale
        bbox_min, bbox_max = blender.bbox(object_name)
        scale = 1 / max(bbox_max - bbox_min)
        obj.scale = obj.scale * scale

        # Apply scale to matrix_world.
        blender.context.view_layer.update()

        # Re-center
        bbox_min, bbox_max = blender.bbox(object_name)
        offset = -(bbox_min + bbox_max) / 2
        obj.matrix_world.translation += offset

        blender.ops.object.select_all(action="DESELECT")  # ?

    return change_fn


def local_debug_test():
    b = Blender()
    engine = "BLENDER_EEVEE_NEXT"  # in {"BLENDER_EEVEE_NEXT", "CYCLES"}

    # apply rendering settings
    b.apply(
        atomic_sequential_changes(
            default_rendering_settings(engine, resolution=518),
            default_eevee_settings(),
            default_cycles_settings(geo_mode=False),
        )
    )

    b.apply(
        atomic_sequential_changes(
            enable_nodes(),
            output_rgba(),
        )
    )

    # add lights
    b.apply(
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

    # add camera
    b.apply(
        add_camera(
            view={
                "yaw": 8.329076232502526,
                "pitch": -0.7618072284012608,
                "radius": 2,
                "fov": 0.6981317007977318,
            }
        )
    )

    b.apply(
        load_model(
            # "/large_experiments/3dfy/datasets/objaverse-xl/hf-objaverse-v1/glbs/000-000/f86ba7e95b664edaac48a7da6abd548e.glb",
            "/large_experiments/3dfy/datasets/objaverse-xl/hf-objaverse-v1/glbs/000-000/006d1922549f4f83a87158c46c8f8ea8.glb",
            group_as="model",
        )
    )
    b.apply(normalize_scene("model"))

    with timeit(level="WARNING", message_template="rendering time {duration}"):
        outputs = b.render()

    skimage.io.imsave("output.png", outputs["out/rgba0001.png"])

    with timeit(level="WARNING", message_template="rendering time {duration}"):
        outputs = b.render()

    print("done")


if __name__ == "__main__":
    local_debug_test()
