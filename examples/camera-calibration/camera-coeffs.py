import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2 as cv
import numpy as np


@dataclass
class CameraCoeffs:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def __str__(self) -> str:
        return "\n".join(
            [
                f"fx: {self.fx:.50f}",
                f"fy: {self.fy:.50f}",
                f"cx: {self.cx:.50f}",
                f"cy: {self.cy:.50f}",
                f"k1: {self.k1:.50f}",
                f"k2: {self.k2:.50f}",
                f"p1: {self.p1:.50f}",
                f"p2: {self.p2:.50f}",
                f"k3: {self.k3:.50f}",
            ]
        )


class DirectoryMustExist(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if (
            not values.strip()
            or not Path(values.strip()).exists()
            or not Path(values.strip()).is_dir()
        ):
            raise argparse.ArgumentError(
                argument=self, message="Incorrect directory path"
            )
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera coefficients calculation",
        epilog="Example: python camera-coeffs.py --images /path/to/images",
    )
    parser.add_argument(
        "--images",
        required=True,
        type=str,
        action=DirectoryMustExist,
        help="Path to directory where calibration iamges are stored",
    )
    parser.add_argument(
        "--chessboard-h",
        required=True,
        type=int,
        help="Number of inner corners in chessboard vertically",
    )
    parser.add_argument(
        "--chessboard-w",
        required=True,
        type=int,
        help="Number of inner corners in chessboard horizontally",
    )
    return parser.parse_args()


# adaptation of https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
def extract_chessboard_inner_corners(
    root_dir: str, inner_corners_h: int, inner_corners_w: int
) -> Tuple[int, int, List[np.ndarray]]:
    chessboard_inner_corners = []

    img_w, img_h = None, None
    for root, _, files in os.walk(root_dir):
        for fname in files:
            img = cv.imread(os.path.join(root, fname))
            if img is None:
                continue
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if img_w is None or img_h is None:
                img_h, img_w, *_ = gray.shape

            if img_h != gray.shape[0] or img_w != gray.shape[1]:
                print(
                    f"ERROR - image dimensions {gray.shape[1]}x{gray.shape[0]} do not match {img_w}x{img_h}: {fname}",
                    file=sys.stderr,
                )
                continue

            # https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a
            ret, corners = cv.findChessboardCorners(
                image=gray,
                patternSize=(inner_corners_w, inner_corners_h),
                corners=None,
                flags=None,
            )

            if ret != True:
                print(
                    f"ERROR - chessboard not found on image: {fname}", file=sys.stderr
                )
                continue

            # https://docs.opencv.org/4.11.0/dd/d1a/group__imgproc__feature.html#ga354e0d7c86d0d9da75de9b9701a9a87e
            refined_corners = cv.cornerSubPix(
                image=gray,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            chessboard_inner_corners.append(refined_corners)
    return img_w, img_h, chessboard_inner_corners


def find_coeffs(
    chessboard_inner_corners: List[np.ndarray],
    img_w: int,
    img_h: int,
    inner_corners_h: int,
    inner_corners_w: int,
) -> CameraCoeffs:
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((inner_corners_h * inner_corners_w, 3), np.float32)
    objp[:, :2] = np.mgrid[0:inner_corners_w, 0:inner_corners_h].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [
        objp.copy() for _ in chessboard_inner_corners
    ]  # 3d point in real world space

    # https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=chessboard_inner_corners,
        imageSize=(img_w, img_h),
        cameraMatrix=None,
        distCoeffs=None,
    )

    mean_error = 0
    for i in range(len(objpoints)):
        # https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
        imgpoints2, _ = cv.projectPoints(
            objectPoints=objpoints[i],
            rvec=rvecs[i],
            tvec=tvecs[i],
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs,
        )
        # https://docs.opencv.org/4.11.0/d2/de8/group__core__array.html#ga7c331fb8dd951707e184ef4e3f21dd33
        error = cv.norm(
            src1=chessboard_inner_corners[i], src2=imgpoints2, normType=cv.NORM_L2
        ) / len(imgpoints2)
        mean_error += error

    print("Calibration error: {}".format(mean_error / len(objpoints)), file=sys.stderr)

    return CameraCoeffs(
        fx=cameraMatrix[0][0],
        fy=cameraMatrix[1][1],
        cx=cameraMatrix[0][2],
        cy=cameraMatrix[1][2],
        k1=distCoeffs[0][0],
        k2=distCoeffs[0][1],
        p1=distCoeffs[0][2],
        p2=distCoeffs[0][3],
        k3=distCoeffs[0][4],
    )


if __name__ == "__main__":
    args = parse_args()

    img_w, img_h, chessboard_inner_corners = extract_chessboard_inner_corners(
        root_dir=args.images,
        inner_corners_h=args.chessboard_h,
        inner_corners_w=args.chessboard_w,
    )
    camera_coeffs = find_coeffs(
        chessboard_inner_corners=chessboard_inner_corners,
        img_w=img_w,
        img_h=img_h,
        inner_corners_h=args.chessboard_h,
        inner_corners_w=args.chessboard_w,
    )
    print(camera_coeffs)
