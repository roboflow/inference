from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class Sam3_3D_Objects_Metadata(BaseModel):
    rotation: Optional[List[float]] = Field(
        default=None, description="Rotation transformation parameters (quaternion, 4 floats)"
    )
    translation: Optional[List[float]] = Field(
        default=None, description="Translation transformation parameters (x, y, z)"
    )
    scale: Optional[List[float]] = Field(
        default=None, description="Scale transformation parameters (x, y, z)"
    )


class Sam3_3D_Object_Item(BaseModel):
    """Individual 3D object output with mesh, gaussian, and transformation metadata."""
    mesh_glb: Optional[bytes] = Field(
        default=None, description="The 3D mesh in GLB format (binary)"
    )
    gaussian_ply: Optional[bytes] = Field(
        default=None, description="The Gaussian splatting in PLY format (binary)"
    )
    metadata: Sam3_3D_Objects_Metadata = Field(
        default_factory=Sam3_3D_Objects_Metadata,
        description="3D transformation metadata (rotation, translation, scale)"
    )

    class Config:
        arbitrary_types_allowed = True


class Sam3_3D_Objects_Response(BaseModel):
    mesh_glb: Optional[bytes] = Field(
        default=None, description="The 3D scene mesh in GLB format (binary)"
    )
    gaussian_ply: Optional[bytes] = Field(
        default=None, description="The combined Gaussian splatting in PLY format (binary)"
    )
    objects: List[Sam3_3D_Object_Item] = Field(
        default=[], description="List of individual 3D objects with their meshes, gaussians, and metadata"
    )
    time: float = Field(
        description="The time in seconds it took to produce the 3D outputs including preprocessing"
    )

    class Config:
        arbitrary_types_allowed = True
