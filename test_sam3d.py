import os
os.environ["PROJECT"] = "roboflow-staging"
os.environ["ROBOFLOW_API_KEY"] = "4vZML1nYbFsGjgSLccm6"
os.environ["API_BASE_URL"] = "https://api.roboflow.one"
os.environ["SAM3_3D_OBJECTS_ENABLED"] = "true"
from inference.models.sam3_3d.segment_anything_3d import SegmentAnything3_3D_Objects #, SegmentAnything3_3D_Body
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest #, Sam3_3D_Body_InferenceRequest
from inference.core.entities.responses.sam3_3d import Sam3_3D_Objects_Response #, Sam3_3D_Body_Response
from inference import get_model
import json

model = get_model("sam3-3d-objects", api_key = "4vZML1nYbFsGjgSLccm6") #SegmentAnything3_3D_Objects(api_key = "Em3CODkJT1vbZ52wlWwk")

def load_coco_mask(annotations_path, image_id, category_id):
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)

    annotations = [
        ann for ann in coco_data['annotations']
        if ann['image_id'] == image_id and ann['category_id'] == category_id
    ]

    if not annotations:
        raise ValueError(f"No annotation found for image_id={image_id}, category_id={category_id}")

    annotation = annotations[0]
    segmentation = annotation['segmentation'][0]

    return segmentation

# Paths
image_path = "/home/matveipopov/tdfy/test/00082_jpg.rf.ebbad5a13dd1489cc121247fa14f1967.jpg"
annotations_path = "/home/matveipopov/tdfy/test/_annotations.coco.json"
image_id = 43
category_id = 8

# Load mask polygon from COCO annotations
mask_polygon = load_coco_mask(annotations_path, image_id, category_id)

# Prepare image for inference
image = {
    "type": "file",
    "value": image_path
}

# Create the SAM3_3D inference request
request = Sam3_3D_Objects_InferenceRequest(
    image=image,
    mask_input=mask_polygon,
)

# Run inference
print("Running SAM3_3D inference...")
response: Sam3_3D_Objects_Response = model.infer_from_request(request)

print(f"\nSAM3_3D inference completed in {response.time:.2f} seconds!")
print("=" * 80)

# 1. Mesh (GLB)
if response.mesh_glb is not None:
    print(f"\n[Output 1/5] Mesh (GLB format)")
    with open("out_mesh.glb", "wb") as f:
        f.write(response.mesh_glb)
    print(f"  ✓ Saved mesh to out_mesh.glb ({len(response.mesh_glb):,} bytes)")
else:
    print(f"\n[Output 1/5] No mesh output")

# 2. Gaussian splatting
if response.gaussian_ply is not None:
    print(f"\n[Output 2/5] Gaussian splatting (PLY format)")
    with open("out_gaussian.ply", "wb") as f:
        f.write(response.gaussian_ply)
    print(f"  ✓ Saved gaussian to out_gaussian.ply ({len(response.gaussian_ply):,} bytes)")
else:
    print(f"\n[Output 2/5] No gaussian output")

# 3. Metadata (rotation, translation, scale)
print(f"\n[Output 5/5] Metadata:")
metadata_dict = {
    "rotation": response.metadata.rotation,
    "translation": response.metadata.translation,
    "scale": response.metadata.scale,
}
print(json.dumps(metadata_dict, indent=2))

# Save metadata to file
with open("out_metadata.json", "w") as f:
    json.dump(metadata_dict, f, indent=2)
print(f"  ✓ Saved metadata to out_metadata.json")

print("\n" + "=" * 80)
print("All outputs saved successfully!")
print("=" * 80)
""" 

body_model = SegmentAnything3_3D_Body(api_key = "Em3CODkJT1vbZ52wlWwk")

# Paths
image_path = "ds2_women-school-secretariat-model-159687_png.rf.c44754ecbf000ad9f0de28df2bc08f95.jpg"

# Prepare image for inference
image = {
    "type": "file",
    "value": image_path
}

# Create the SAM3_3D inference request
request = Sam3_3D_Body_InferenceRequest(
    image=image
)

# Run inference
print("Running SAM3_3D inference...")
response: Sam3_3D_Body_Response = body_model.infer_from_request(request)

print(f"\nSAM3_3D inference completed in {response.time:.2f} seconds!")
print("=" * 80)

# Save the GLB mesh
if response.mesh_glb is not None:
    print(f"\n[Output] Mesh (GLB format)")
    with open("out_mesh.glb", "wb") as f:
        f.write(response.mesh_glb)
    print(f"  ✓ Saved mesh to out_mesh.glb ({len(response.mesh_glb):,} bytes)")
else:
    print(f"\n[Output] No mesh output")

# Save gaussian splatting if available
if hasattr(response, 'gaussian_ply') and response.gaussian_ply is not None:
    print(f"\n[Output] Gaussian splatting (PLY format)")
    with open("out_gaussian.ply", "wb") as f:
        f.write(response.gaussian_ply)
    print(f"  ✓ Saved gaussian to out_gaussian.ply ({len(response.gaussian_ply):,} bytes)")

# Save metadata if available
if hasattr(response, 'metadata') and response.metadata is not None:
    print(f"\n[Output] Metadata:")
    metadata_dict = {
        "rotation": response.metadata.rotation,
        "translation": response.metadata.translation,
        "scale": response.metadata.scale,
    }
    print(json.dumps(metadata_dict, indent=2))

    with open("out_metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=2)
    print(f"  ✓ Saved metadata to out_metadata.json")

print("\n" + "=" * 80)
print("All outputs saved successfully!")
print("=" * 80)
"""