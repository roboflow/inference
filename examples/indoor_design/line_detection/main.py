import numpy as np
import cv2

def _to_h(p):
    return np.array([p[0], p[1], 1.0], dtype=np.float64)

def line_from_points(p1, p2):
    # homogeneous line l = p1 x p2 (in pixel homogeneous coords)
    l = np.cross(_to_h(p1), _to_h(p2))
    n = np.linalg.norm(l[:2])
    if n < 1e-12:
        return None
    return l / n

def point_line_dist(p, l):
    # distance of pixel point p to homogeneous line l (normalized by ||(a,b)||)
    ph = _to_h(p)
    return abs(l @ ph)

def intersect_lines(l1, l2):
    v = np.cross(l1, l2)
    if abs(v[2]) < 1e-12:
        return None
    return v / v[2]  # (u,v,1)

def detect_lines_lsd(img_gray):
    lsd = cv2.createLineSegmentDetector(_refine=cv2.LSD_REFINE_STD)
    lines = lsd.detect(img_gray)[0]  # shape (N,1,4) or None
    if lines is None:
        return []
    lines = lines.reshape(-1, 4)
    # filter short lines
    out = []
    for x1,y1,x2,y2 in lines:
        if np.hypot(x2-x1, y2-y1) >= 30:  # tune
            out.append(((x1,y1),(x2,y2)))
    return out

def ransac_vanishing_point(lines, n_iters=2000, inlier_thresh_px=3.0, min_inliers=50, rng=None):
    """
    Fit one vanishing point from a set of lines using 2-line RANSAC.
    Returns (v, inlier_mask) where v is (u,v,1) homogeneous pixel coords.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    L = []
    for (p1, p2) in lines:
        l = line_from_points(p1, p2)
        if l is not None:
            L.append(l)
    if len(L) < 2:
        return None, None

    L = np.array(L)  # (M,3)
    best_v = None
    best_inliers = None
    best_cnt = 0

    m = len(L)
    for _ in range(n_iters):
        i, j = rng.integers(0, m, size=2)
        if i == j:
            continue
        v = intersect_lines(L[i], L[j])
        if v is None:
            continue

        # Count inliers: vanishing point should lie on the inlier lines
        # distance from v to line l is |l^T v| (since l normalized by ||(a,b)||)
        errs = np.abs(L @ v)
        inliers = errs < inlier_thresh_px
        cnt = int(inliers.sum())

        if cnt > best_cnt:
            best_cnt = cnt
            best_v = v
            best_inliers = inliers

    if best_v is None or best_cnt < min_inliers:
        return None, None

    # Refit vanishing point from all inlier lines in least squares:
    # minimize ||A v|| subject to v3 = 1 (solve for [u,v] in pixels)
    Lin = L[best_inliers]
    # Each line: a u + b v + c = 0
    A = Lin[:, :2]
    c = Lin[:, 2]
    # solve A [u,v] = -c
    uv, *_ = np.linalg.lstsq(A, -c, rcond=None)
    v_refit = np.array([uv[0], uv[1], 1.0], dtype=np.float64)

    return v_refit, best_inliers

def vanishing_points_three(img_bgr, K, debug=False):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines_lsd(gray)

    remaining = lines
    vps = []
    masks = []

    for _ in range(3):
        vp, inliers = ransac_vanishing_point(
            remaining,
            n_iters=3000,
            inlier_thresh_px=4.0,  # tune: 3-6 px
            min_inliers=40
        )
        if vp is None:
            break
        vps.append(vp)
        masks.append(inliers)

        # remove inlier lines for next VP
        # NOTE: inliers are for internal L array; easiest: rebuild mapping by re-evaluating
        # We'll just keep a simple removal by checking line distance to vp.
        new_remaining = []
        for (p1, p2) in remaining:
            l = line_from_points(p1, p2)
            if l is None:
                continue
            if abs(l @ vp) >= 4.0:
                new_remaining.append((p1, p2))
        remaining = new_remaining

    return vps

def manhattan_axes_from_vps(vps, K):
    """
    Convert vanishing points -> directions -> orthonormal Manhattan axes.
    Returns (R_cam_from_manhattan, axes_cam) where axes_cam are unit vectors in camera coords.
    """
    Kinv = np.linalg.inv(K)

    dirs = []
    for vp in vps:
        d = Kinv @ vp  # camera direction (up to scale)
        n = np.linalg.norm(d)
        if n < 1e-12:
            continue
        d = d / n
        dirs.append(d)

    if len(dirs) < 2:
        raise ValueError("Not enough vanishing directions detected.")

    # If we have only 2, compute the third by cross product
    if len(dirs) == 2:
        d3 = np.cross(dirs[0], dirs[1])
        d3 /= (np.linalg.norm(d3) + 1e-12)
        dirs.append(d3)

    # Orthonormalize (Gram-Schmidt + enforce right-handedness)
    x = dirs[0]
    y = dirs[1] - (dirs[1] @ x) * x
    y /= (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z /= (np.linalg.norm(z) + 1e-12)

    # Ensure consistency: pick ordering so axes are close to the original dirs
    # (optional, but helps avoid axis swaps)
    R = np.stack([x, y, z], axis=1)  # columns are axes in camera coords

    # Fix possible reflection
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    return R, (x, y, z)

if __name__ == "__main__":
    img_bgr = cv2.imread("room.jpg")
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    vps = vanishing_points_three(img_bgr, K)
    R_cam_from_manhattan, (x_cam, y_cam, z_cam) = manhattan_axes_from_vps(vps, K)
    print("VPs:", vps)
    print("Axes (camera coords):", x_cam, y_cam, z_cam)
    print("R:", R_cam_from_manhattan)
