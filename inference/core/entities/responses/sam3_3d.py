from typing import List, Optional

from pydantic import BaseModel, Field


class Sam3_3D_Objects_Metadata(BaseModel):
    """3D transformation metadata.

    Attributes:
        rotation (List[float]): Rotation parameters.
        translation (List[float]): Translation parameters.
        scale (List[float]): Scale parameters.
    """

    rotation: Optional[List[float]] = Field(
        default=None, description="Rotation transformation parameters"
    )
    translation: Optional[List[float]] = Field(
        default=None, description="Translation transformation parameters"
    )
    scale: Optional[List[float]] = Field(
        default=None, description="Scale transformation parameters"
    )


class Sam3_3D_Objects_Response(BaseModel):
    """SAM3_3D inference response for 3D object generation.

    Attributes:
        mesh_glb (Optional[bytes]): The 3D mesh in GLB format (binary).
        gaussian_ply (Optional[bytes]): The Gaussian splatting in PLY format (binary).
        gaussian_4_ply (Optional[bytes]): The Gaussian 4GS in PLY format (binary).
        voxel_ply (Optional[bytes]): The voxel point cloud in PLY format (binary).
        metadata (Sam3_3DMetadata): Transformation metadata (rotation, translation, scale).
        time (float): The time in seconds it took to produce the 3D outputs including preprocessing.
    """

    mesh_glb: Optional[bytes] = Field(
        default=None, description="The 3D mesh in GLB format (binary)"
    )
    gaussian_ply: Optional[bytes] = Field(
        default=None, description="The Gaussian splatting in PLY format (binary)"
    )
    metadata: Sam3_3D_Objects_Metadata = Field(
        description="3D transformation metadata (rotation, translation, scale)"
    )
    time: float = Field(
        description="The time in seconds it took to produce the 3D outputs including preprocessing"
    )

    class Config:
        # Allow bytes type
        arbitrary_types_allowed = True
