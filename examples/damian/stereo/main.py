"""
Stereo example: SIFT → Feature Comparison → Essential Matrix → Stereo Rectification.
Run from repo root. Requires data in examples/damian/stereo/data/ (e.g. left.png, right.png).
Rectified images are saved to output/.
"""

from pathlib import Path

import cv2

from inference.core.workflows.core_steps.classical_cv.sift.v1 import SIFTBlockV1
from inference.core.workflows.core_steps.classical_cv.feature_comparison.v1 import (
    FeatureComparisonBlockV1,
)
from inference.core.workflows.core_steps.transformations.essential_matrix.v1 import (
    EssentialMatrixBlockV1,
)
from inference.core.workflows.core_steps.transformations.stereo_rectification.v1 import (
    StereoRectificationBlockV1,
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
            parent_id="left",
            camera_intrinsics=camera_intrinsics,
        ),
        image_reference=str(DATA_DIR / "dresden_1.png"),
    )

    image_2 = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="right",
            camera_intrinsics=camera_intrinsics,
        ),
        image_reference=str(DATA_DIR / "dresden_2.png"),
    )

    sift_block = SIFTBlockV1()
    feature_comparison_block = FeatureComparisonBlockV1()
    essential_matrix_block = EssentialMatrixBlockV1()
    stereo_rectification_block = StereoRectificationBlockV1()

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

    rect_results = stereo_rectification_block.run(
        image_1=image_1,
        image_2=image_2,
        camera_intrinsics_1=camera_intrinsics,
        camera_intrinsics_2=camera_intrinsics,
        rotation=em_results["rotation"],
        translation=em_results["translation"],
    )

    rect1 = rect_results["rectified_image_1"].numpy_image
    rect2 = rect_results["rectified_image_2"].numpy_image
    cv2.imwrite(str(OUTPUT_DIR / "rectified_left.png"), rect1)
    cv2.imwrite(str(OUTPUT_DIR / "rectified_right.png"), rect2)
    print(f"Saved rectified images to {OUTPUT_DIR}")
