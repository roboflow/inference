"""
Segmentation Refinement (Algorithm 4 from Feng, Taguchi, Kamat)
Erosion + region-growing + final merge AHC
"""

from __future__ import annotations

import math
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from examples.plane_detection.ahc import (
    Graph,
    Node,
    Stats3D,
    ahc_cluster,
    nodes_to_label_image,
    point_plane_dist2,
)


def segmentation_refinement(
    points: np.ndarray,
    coarse_nodes: List[Node],
    coarse_planes: Dict[int, Tuple[np.ndarray, float]],
    block_h: int,
    block_w: int,
    TMSE: float,
    TNUM: int,
    TDEG: float,
) -> Tuple[np.ndarray, Dict[int, Tuple[np.ndarray, float]]]:
    """
    Implements Algorithm 4:
      1) Erode border blocks from each segment
      2) Pixel-wise region growing from new boundaries
      3) Build adjacency graph G0 and run AHC again on it

    Returns:
      refined_label_img (M,N)
      refined_planes dict {refined_label -> (n, d)}
    """
    M, N, _ = points.shape

    # --- Build initial coarse label image ---
    coarse_label = nodes_to_label_image(points, coarse_nodes, block_h, block_w)

    # Map segment index -> blocks set (bi,bj)
    seg_blocks: List[Set[Tuple[int, int]]] = [set(n.blocks) for n in coarse_nodes]

    # --- 1) Erode each segment by removing border blocks (Algorithm 4 lines 5-13) ---
    nBH = int(math.ceil(M / block_h))
    nBW = int(math.ceil(N / block_w))

    def block_neighbors(bi: int, bj: int):
        for dbi, dbj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = bi + dbi, bj + dbj
            if 0 <= ni < nBH and 0 <= nj < nBW:
                yield (ni, nj)

    eroded_blocks: List[Set[Tuple[int, int]]] = []
    removed_blocks: List[Set[Tuple[int, int]]] = []

    for k in range(len(seg_blocks)):
        Bk = seg_blocks[k]
        border = set()
        for b in Bk:
            if any(nb not in Bk for nb in block_neighbors(*b)):
                border.add(b)
        eroded = Bk - border
        eroded_blocks.append(eroded)
        removed_blocks.append(border)

    # Initialize refined label image from eroded blocks only
    refined_label = -np.ones((M, N), dtype=int)

    for k, blocks in enumerate(eroded_blocks):
        for (bi, bj) in blocks:
            r0 = bi * block_h
            c0 = bj * block_w
            r1 = min((bi + 1) * block_h, M)
            c1 = min((bj + 1) * block_w, N)
            P = points[r0:r1, c0:c1]
            valid = ~np.isnan(P).any(axis=2)
            refined_label[r0:r1, c0:c1][valid] = k

    # Refit plane per segment after erosion (needed for region-grow threshold)
    seg_plane_n: List[np.ndarray] = []
    seg_plane_d: List[float] = []
    seg_plane_mse: List[float] = []
    seg_stats: List[Stats3D] = []

    for k in range(len(eroded_blocks)):
        mask = (refined_label == k)
        Pk = points[mask]
        if Pk.shape[0] < 3:
            # keep coarse plane as fallback
            # coarse_planes keyed by node.id; coarse_nodes[k].id maps to it
            n0, d0 = coarse_planes.get(coarse_nodes[k].id, (np.array([0.0, 0.0, 1.0]), 0.0))
            seg_plane_n.append(n0)
            seg_plane_d.append(float(d0))
            seg_plane_mse.append(float("inf"))
            seg_stats.append(Stats3D(n=0, s=np.zeros(3), ss=np.zeros((3, 3))))
            continue
        st = Stats3D.from_points(Pk.reshape(-1, 3))
        n_hat, d, mse = st.pca_plane()
        seg_plane_n.append(n_hat)
        seg_plane_d.append(float(d))
        seg_plane_mse.append(float(mse))
        seg_stats.append(st)

    # Boundary queue initialization: enqueue boundary pixels of each eroded segment (Alg 4 lines 10-13)
    Q = deque()
    for k in range(len(eroded_blocks)):
        if not np.any(refined_label == k):
            continue
        ys, xs = np.where(refined_label == k)
        for y, x in zip(ys, xs):
            # boundary if any 4-neighbor not in k
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if yy < 0 or yy >= M or xx < 0 or xx >= N:
                    continue
                if refined_label[yy, xx] != k:
                    Q.append((y, x, k))
                    break

    # --- 2) Region grow from boundaries assigning points to closest plane (Algorithm 4 lines 14-27) ---
    # Track segment adjacency discovered during growth => E0
    seg_adj: Set[Tuple[int, int]] = set()

    def try_assign(y: int, x: int, k: int) -> bool:
        """Attempt to assign (y,x) to segment k based on distance gating."""
        p = points[y, x]
        if np.isnan(p).any():
            return False

        # If already assigned to k, no need.
        if refined_label[y, x] == k:
            return False

        # Distance gate: DIST^2 > 9*MSE(Bk) => reject (Alg 4 line 17)
        n_hat = seg_plane_n[k]
        d = seg_plane_d[k]
        mse = seg_plane_mse[k]
        if not np.isfinite(mse):
            # if we don't have a meaningful MSE, skip gating
            gate_ok = True
        else:
            gate_ok = point_plane_dist2(p, n_hat, d) <= 9.0 * mse

        if not gate_ok:
            return False

        cur = refined_label[y, x]
        if cur == -1:
            refined_label[y, x] = k
            Q.append((y, x, k))
            return True

        # If already assigned to other segment l, connect nodes and pick closer plane (Alg 4 lines 18-22)
        l = int(cur)
        if l != k:
            a, b = (k, l) if k < l else (l, k)
            seg_adj.add((a, b))

            dk = point_plane_dist2(p, seg_plane_n[k], seg_plane_d[k])
            dl = point_plane_dist2(p, seg_plane_n[l], seg_plane_d[l])
            if dk < dl:
                refined_label[y, x] = k
                Q.append((y, x, k))
                return True
        return False

    while Q:
        y, x, k = Q.popleft()
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < M and 0 <= xx < N:
                try_assign(yy, xx, k)

    # Refit planes again after region-growing
    final_seg_stats: List[Stats3D] = []
    for k in range(len(eroded_blocks)):
        mask = (refined_label == k)
        Pk = points[mask]
        if Pk.shape[0] >= 3:
            st = Stats3D.from_points(Pk.reshape(-1, 3))
        else:
            st = Stats3D(n=0, s=np.zeros(3), ss=np.zeros((3, 3)))
        final_seg_stats.append(st)

    # --- 3) Final merge: build small graph G0 and run AHC again (Algorithm 4 line 28) ---
    # Build nodes for segments that exist
    g0_nodes: Dict[int, Node] = {}
    g0_adj: Dict[int, Set[int]] = defaultdict(set)

    # We'll remap "old segment index k" -> "g0 node id"
    remap = {}
    next_id = 0
    for k in range(len(eroded_blocks)):
        if np.any(refined_label == k) and final_seg_stats[k].n >= 3:
            nd = Node(id=next_id, stats=final_seg_stats[k], blocks=set())
            nd.recompute_plane()
            g0_nodes[next_id] = nd
            remap[k] = next_id
            next_id += 1

    for (a, b) in seg_adj:
        if a in remap and b in remap:
            ia, ib = remap[a], remap[b]
            g0_adj[ia].add(ib)
            g0_adj[ib].add(ia)

    for nid in g0_nodes.keys():
        g0_adj[nid] = set(g0_adj[nid])

    G0 = Graph(nodes=g0_nodes, adj=g0_adj)

    merged_nodes, merged_planes_by_nodeid = ahc_cluster(
        G0, TMSE=TMSE, TNUM=TNUM, TDEG=TDEG
    )

    # Convert merged result to new label image:
    # We need mapping from original segment indices to merged cluster labels.
    # We can do this by assigning each g0 node id to an output cluster index.
    g0node_to_cluster = {}
    for cluster_idx, node in enumerate(merged_nodes):
        # node is the extracted node from AHC; but AHC returns extracted nodes only.
        # In this final merge, graph is small and extracted nodes are the final planes.
        g0node_to_cluster[node.id] = cluster_idx

    # Some nodes might not be extracted if they are below TNUM; for refinement graph, we often
    # want to keep them anyway. We'll fall back: any remaining nodes in planes dict are "extracted";
    # otherwise treat them as singleton.
    for nid in g0_nodes.keys():
        if nid not in g0node_to_cluster:
            g0node_to_cluster[nid] = len(g0node_to_cluster)

    # Remap refined_label (old segment k) -> final cluster label
    final_label = -np.ones_like(refined_label)
    for k, g0id in remap.items():
        final_label[refined_label == k] = g0node_to_cluster[g0id]

    # Plane dict by final cluster label
    final_planes: Dict[int, Tuple[np.ndarray, float]] = {}

    # merged_planes_by_nodeid keys are node ids in G0 after merges;
    # but we stored g0node_to_cluster by extracted node ids.
    for node_id, (n_hat, d) in merged_planes_by_nodeid.items():
        cl = g0node_to_cluster.get(node_id, None)
        if cl is not None:
            final_planes[int(cl)] = (n_hat, float(d))

    # If any cluster plane missing (singleton fallbacks), compute from its pixels
    for cl in np.unique(final_label):
        if cl < 0:
            continue
        if cl not in final_planes:
            mask = (final_label == cl)
            P = points[mask]
            if P.shape[0] >= 3:
                st = Stats3D.from_points(P.reshape(-1, 3))
                n_hat, d, _ = st.pca_plane()
                final_planes[int(cl)] = (n_hat, float(d))

    return final_label, final_planes
