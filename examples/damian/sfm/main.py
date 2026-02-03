"""
SfM example: SIFT → Feature Comparison → Essential Matrix → Triangulation → 3D visualization.
Run from repo root. Requires data in examples/damian/sfm/data/ (dresden_1.png, dresden_2.png).
"""

from pathlib import Path

from inference.core.workflows.core_steps.classical_cv.sift.v1 import SIFTBlockV1
from inference.core.workflows.core_steps.classical_cv.feature_comparison.v1 import (
    FeatureComparisonBlockV1,
)
from inference.core.workflows.core_steps.transformations.essential_matrix.v1 import (
    EssentialMatrixBlockV1,
)
from inference.core.workflows.core_steps.transformations.triangulation.v1 import (
    TriangulationBlockV1,
)
from inference.core.workflows.core_steps.visualizations.sfm_3d.v1 import (
    SfMVisualization3DBlockV1,
    create_sfm_3d_figure,
)
from inference.core.workflows.execution_engine.entities.base import (
    CameraIntrinsics,
    ImageParentMetadata,
    WorkflowImageData,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


if __name__ == "__main__":
    camera_intrinsics = CameraIntrinsics(
        fx=2780.1700000000000728,
        fy=2773.5399999999999636,
        cx=1539.25,
        cy=1001.2699999999999818,
        k1=0,
        k2=0,
        k3=0,
        p1=0,
        p2=0,
    )

    image_1 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="dresden_1",
            camera_intrinsics=camera_intrinsics,
        ),
        image_reference=str(DATA_DIR / "dresden_1.png"),
    )

    image_2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="dresden_2",
            camera_intrinsics=camera_intrinsics,
        ),
        image_reference=str(DATA_DIR / "dresden_2.png"),
    )

    sift_block = SIFTBlockV1()
    feature_comparison_block = FeatureComparisonBlockV1()
    essential_matrix_block = EssentialMatrixBlockV1()
    triangulation_block = TriangulationBlockV1()
    sfm_3d_viz_block = SfMVisualization3DBlockV1()

    results_1 = sift_block.run(image=image_1)
    results_2 = sift_block.run(image=image_2)

    fc_results = feature_comparison_block.run(
        keypoints_1=results_1["keypoints"],
        descriptors_1=results_1["descriptors"],
        keypoints_2=results_2["keypoints"],
        descriptors_2=results_2["descriptors"],
    )

    em_results = essential_matrix_block.run(
        good_matches=fc_results["good_matches"],
        camera_intrinsics_1=camera_intrinsics,
        camera_intrinsics_2=camera_intrinsics,
    )

    tri_results = triangulation_block.run(
        good_matches=fc_results["good_matches"],
        camera_intrinsics_1=camera_intrinsics,
        camera_intrinsics_2=camera_intrinsics,
        rotation=em_results["rotation"],
        translation=em_results["translation"],
    )

    print(
        f"Matched pairs: {fc_results['good_matches_count']}, "
        f"Triangulated points: {tri_results['points_count']}"
    )

    sfm_3d_viz_block.run(
        points_3d=tri_results["points_3d"],
        rotation=em_results["rotation"],
        translation=em_results["translation"],
        figsize_width=10,
        figsize_height=8,
        dpi=120,
    )

    pts = tri_results["points_3d_array"]
    fig = create_sfm_3d_figure(
        pts,
        em_results["rotation"],
        em_results["translation"],
    )
    out_path = OUTPUT_DIR / "sfm_3d_visualization.html"
    if fig is not None:
        fig.write_html(str(out_path))
        print(f"Saved interactive 3D visualization to {out_path}")
    else:
        print("Plotly not available; skipping HTML export.")
