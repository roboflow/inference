import os
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"
os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
os.environ["ATTN_BACKEND"] = "flash_attn"

from inference.models.sam3_3d.segment_anything_3d import SegmentAnything3_3D_Objects
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import Sam3_3D_Objects_Response
from inference import get_model
import json

def load_coco_masks(annotations_path, image_id):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] == image_id
    ]

    segmentations = [ann['segmentation'] for ann in annotations]

    return segmentations

image_path = "inference/models/sam3_3d/demo-data/train/IMG_6860_jpeg.rf.9fceef9265cef07f0cd4b339527a6756.jpg"
annotations_path = "inference/models/sam3_3d/demo-data/train/_annotations.coco.json"
image_id = 0

mask_polygon = load_coco_masks(annotations_path, image_id)

image = {
    "type": "file",
    "value": image_path
}

model = get_model("sam3-3d-objects")

request = Sam3_3D_Objects_InferenceRequest(
    image=image,
    mask_input=mask_polygon,
)

print("Running SAM3_3D inference...")
response: Sam3_3D_Objects_Response = model.infer_from_request(request)

print(f"\nSAM3_3D inference completed in {response.time:.2f} seconds!")
print("=" * 80)

# 1. Scene Mesh (GLB)
if response.mesh_glb is not None:
    print(f"\n[Output 1/3] Scene Mesh (GLB format)")
    with open("out_mesh.glb", "wb") as f:
        f.write(response.mesh_glb)
    print(f"  Saved mesh to out_mesh.glb ({len(response.mesh_glb):,} bytes)")
else:
    print(f"\n[Output 1/3] No mesh output")

# 2. Combined Gaussian splatting
if response.gaussian_ply is not None:
    print(f"\n[Output 2/3] Combined Gaussian splatting (PLY format)")
    with open("out_gaussian.ply", "wb") as f:
        f.write(response.gaussian_ply)
    print(f"  Saved gaussian to out_gaussian.ply ({len(response.gaussian_ply):,} bytes)")
else:
    print(f"\n[Output 2/3] No combined gaussian output")

# 3. Individual objects
print(f"\n[Output 3/3] Individual objects ({len(response.objects)} objects)")
objects_metadata = []
for i, obj in enumerate(response.objects):
    print(f"\n  Object {i}:")

    # Save individual mesh
    if obj.mesh_glb is not None:
        filename = f"out_object_{i}_mesh.glb"
        with open(filename, "wb") as f:
            f.write(obj.mesh_glb)
        print(f"    Saved mesh to {filename} ({len(obj.mesh_glb):,} bytes)")

    # Save individual gaussian
    if obj.gaussian_ply is not None:
        filename = f"out_object_{i}_gaussian.ply"
        with open(filename, "wb") as f:
            f.write(obj.gaussian_ply)
        print(f"    Saved gaussian to {filename} ({len(obj.gaussian_ply):,} bytes)")

    # Collect metadata
    obj_metadata = {
        "object_index": i,
        "rotation": obj.metadata.rotation,
        "translation": obj.metadata.translation,
        "scale": obj.metadata.scale,
    }
    objects_metadata.append(obj_metadata)
    print(f"    Metadata: rotation={obj.metadata.rotation is not None}, translation={obj.metadata.translation is not None}, scale={obj.metadata.scale is not None}")

# Save all metadata to file
with open("out_metadata.json", "w") as f:
    json.dump({"objects": objects_metadata}, f, indent=2)
print(f"\n  Saved all metadata to out_metadata.json")

print("\n" + "=" * 80)
print("All outputs saved successfully!")
print("=" * 80)
