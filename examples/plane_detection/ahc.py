"""
Fast Plane Extraction in Organized Point Clouds Using Agglomerative Hierarchical Clustering
Feng, Taguchi, Kamat (MERL TR2014-066 / ICRA 2014)

This script implements the algorithm in 3 separated steps:
  1) Graph Initialization
  2) Agglomerative Hierarchical Clustering (AHC)
  3) Segmentation Refinement (erosion + region-growing + final merge AHC)

Input:
  points: (M, N, 3) float array for organized point cloud (x,y,z)
          Missing points should be NaN (any component NaN => missing).

Output:
  label_img: (M, N) int array; -1 means non-planar/unassigned
  planes: dict {label -> (n, d)} plane equation n·p + d = 0, ||n||=1

Notes:
  - MSE is taken as the smallest eigenvalue of the 3x3 covariance matrix (PCA plane fit).
    This equals the mean squared orthogonal distance to the best-fit plane.
  - For Kinect-like discontinuity check we use Eq.(1) from the paper:
      |za - zb| > 2*alpha*(|za| + 0.5)   (z in millimeters)
  - Thresholding parameters are sensor dependent; defaults are reasonable starting points.
"""

from __future__ import annotations

import heapq
import colorsys
import math
from collections import deque, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple, List, Optional

import click
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------------
# Plane fit utilities (PCA via sufficient statistics)
# ----------------------------

@dataclass
class Stats3D:
    """First + second order statistics to allow O(1) merge and PCA on 3x3 covariance."""
    n: int
    s: np.ndarray      # shape (3,) sum of coordinates
    ss: np.ndarray     # shape (3,3) sum of outer products (x x^T)

    @staticmethod
    def from_points(P: np.ndarray) -> "Stats3D":
        # P: (k,3)
        n = int(P.shape[0])
        s = P.sum(axis=0)
        ss = P.T @ P
        return Stats3D(n=n, s=s, ss=ss)

    def merge(self, other: "Stats3D") -> "Stats3D":
        return Stats3D(
            n=self.n + other.n,
            s=self.s + other.s,
            ss=self.ss + other.ss
        )

    def mean(self) -> np.ndarray:
        return self.s / max(self.n, 1)

    def cov(self) -> np.ndarray:
        """Unbiasedness isn't needed; we want mean squared distances."""
        if self.n <= 0:
            return np.full((3, 3), np.inf)
        mu = self.mean()
        # E[xx^T] - mu mu^T
        return (self.ss / self.n) - np.outer(mu, mu)

    def pca_plane(self) -> Tuple[np.ndarray, float, float]:
        """
        Returns:
          n_hat: unit normal (3,)
          d: plane offset so that n·p + d = 0
          mse: mean squared orthogonal distance
        """
        if self.n < 3:
            return np.array([0.0, 0.0, 1.0]), 0.0, float("inf")
        C = self.cov()
        # symmetric 3x3
        w, V = np.linalg.eigh(C)
        idx = int(np.argmin(w))
        n_hat = V[:, idx]
        # normalize defensively
        norm = np.linalg.norm(n_hat)
        if norm == 0:
            return np.array([0.0, 0.0, 1.0]), 0.0, float("inf")
        n_hat = n_hat / norm
        mu = self.mean()
        d = -float(n_hat @ mu)
        mse = float(w[idx])  # smallest eigenvalue = mean squared distance to plane
        return n_hat, d, mse


def point_plane_dist2(p: np.ndarray, n: np.ndarray, d: float) -> float:
    # n assumed unit
    v = float(n @ p + d)
    return v * v


def angle_between_normals_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    c = float(np.clip(abs(n1 @ n2), 0.0, 1.0))
    return math.degrees(math.acos(c))


# ----------------------------
# Graph representation
# ----------------------------

@dataclass
class Node:
    id: int
    stats: Stats3D
    blocks: Set[Tuple[int, int]]          # (bi,bj) indices of initial blocks; used for refinement
    active: bool = True

    # cached plane fit
    n_hat: Optional[np.ndarray] = None
    d: Optional[float] = None
    mse: float = float("inf")

    def recompute_plane(self):
        self.n_hat, self.d, self.mse = self.stats.pca_plane()


@dataclass
class Graph:
    nodes: Dict[int, Node]
    adj: Dict[int, Set[int]]   # undirected adjacency by node id


# ============================================================
# STEP 1: Graph Initialization (Algorithm 2)
# ============================================================

def graph_initialization(
    points: np.ndarray,
    block_h: int = 10,
    block_w: int = 10,
    TMSE: float = 50.0**2,      # paper uses sensor-dependent; example 50^2 for noisy depth (mm^2)
    TANG_deg: float = 60.0,     # edge reject angle threshold
    alpha: float = 0.02,        # depth discontinuity parameter
) -> Tuple[Graph, np.ndarray, Dict[Tuple[int, int], int]]:
    """
    Returns:
      G: graph with nodes being block groups (some rejected)
      block_id_grid: (nBH, nBW) node_id or -1 for rejected/empty
      block_to_node: mapping from (bi,bj) -> node_id (only for non-rejected)
    """
    assert points.ndim == 3 and points.shape[2] == 3
    M, N, _ = points.shape

    nBH = int(math.ceil(M / block_h))
    nBW = int(math.ceil(N / block_w))

    nodes: Dict[int, Node] = {}
    adj: Dict[int, Set[int]] = defaultdict(set)
    block_id_grid = -np.ones((nBH, nBW), dtype=int)
    block_to_node: Dict[Tuple[int, int], int] = {}

    def has_missing(P: np.ndarray) -> bool:
        return bool(np.isnan(P).any())

    def depth_discontinuous_in_block(bi: int, bj: int) -> bool:
        """
        Implements REJECTNODE depth discontinuity check (Algorithm 2, line 17)
        using Eq.(1) from the paper on z values. We check all valid points in the
        block against their 4-connected neighbors in the full image.
        """
        r0 = bi * block_h
        c0 = bj * block_w
        r1 = min((bi + 1) * block_h, M)
        c1 = min((bj + 1) * block_w, N)

        # iterate pixels in the block; vectorization is possible but kept clear here
        for r in range(r0, r1):
            for c in range(c0, c1):
                p = points[r, c]
                if np.isnan(p).any():
                    return True  # missing data => reject
                z = float(p[2])
                # 4 neighbors
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if rr < 0 or rr >= M or cc < 0 or cc >= N:
                        continue
                    q = points[rr, cc]
                    if np.isnan(q).any():
                        continue
                    z2 = float(q[2])
                    # Eq.(1): |za - zb| > 2*alpha*(|za| + 0.5)
                    if abs(z - z2) > 2.0 * alpha * (abs(z) + 0.5):
                        return True
        return False

    def reject_node(bi: int, bj: int, P: np.ndarray) -> bool:
        # Algorithm 2: missing, depth discontinuity, high MSE
        if has_missing(P):
            return True
        if depth_discontinuous_in_block(bi, bj):
            return True
        stats = Stats3D.from_points(P.reshape(-1, 3))
        n_hat, d, mse = stats.pca_plane()
        if mse > TMSE:
            return True
        return False

    # --- initialize nodes (Algorithm 2 lines 3-8) ---
    next_id = 0
    for bi in range(nBH):
        for bj in range(nBW):
            r0 = bi * block_h
            c0 = bj * block_w
            r1 = min((bi + 1) * block_h, M)
            c1 = min((bj + 1) * block_w, N)
            block_pts = points[r0:r1, c0:c1, :]

            if reject_node(bi, bj, block_pts):
                continue

            P = block_pts.reshape(-1, 3)
            stats = Stats3D.from_points(P)
            node = Node(id=next_id, stats=stats, blocks={(bi, bj)})
            node.recompute_plane()
            nodes[next_id] = node
            block_id_grid[bi, bj] = next_id
            block_to_node[(bi, bj)] = next_id
            next_id += 1

    # --- initialize edges (Algorithm 2 lines 9-13) ---
    def reject_edge(va: Optional[int], vb: Optional[int], vc: Optional[int]) -> bool:
        # Algorithm 2 line 21
        if va is None or vb is None or vc is None:
            return True
        if va not in nodes or vb not in nodes or vc not in nodes:
            return True
        # Algorithm 2 line 22: angle between normals of va and vc
        na = nodes[va].n_hat
        nc = nodes[vc].n_hat
        if na is None or nc is None:
            return True
        ang = angle_between_normals_deg(na, nc)
        return ang > TANG_deg

    # For each non-rejected block vb, consider triples horizontally and vertically
    for bi in range(nBH):
        for bj in range(nBW):
            vb = block_id_grid[bi, bj]
            if vb < 0:
                continue

            # horizontal triple (bj-1, bj, bj+1) => if not reject, connect (bj-1)-(bj) and (bj)-(bj+1)
            if bj - 1 >= 0 and bj + 1 < nBW:
                va = int(block_id_grid[bi, bj - 1])
                vc = int(block_id_grid[bi, bj + 1])
                if not reject_edge(va if va >= 0 else None, vb, vc if vc >= 0 else None):
                    # add edges vb-va and vb-vc if those endpoints exist
                    if va >= 0:
                        adj[vb].add(va)
                        adj[va].add(vb)
                    if vc >= 0:
                        adj[vb].add(vc)
                        adj[vc].add(vb)

            # vertical triple (bi-1, bi, bi+1)
            if bi - 1 >= 0 and bi + 1 < nBH:
                va = int(block_id_grid[bi - 1, bj])
                vc = int(block_id_grid[bi + 1, bj])
                if not reject_edge(va if va >= 0 else None, vb, vc if vc >= 0 else None):
                    if va >= 0:
                        adj[vb].add(va)
                        adj[va].add(vb)
                    if vc >= 0:
                        adj[vb].add(vc)
                        adj[vc].add(vb)

    # Ensure all nodes have an adjacency set
    for nid in nodes.keys():
        adj[nid] = set(adj[nid])

    return Graph(nodes=nodes, adj=adj), block_id_grid, block_to_node


# ============================================================
# STEP 2: Agglomerative Hierarchical Clustering (Algorithm 3)
# ============================================================

def ahc_cluster(
    G: Graph,
    TMSE: float,
    TNUM: int = 800,
) -> Tuple[List[Node], Dict[int, Tuple[np.ndarray, float]]]:
    """
    Performs AHC on graph G and returns:
      B: list of extracted coarse plane nodes
      planes: dict {node.id -> (n_hat, d)}
    """
    # Build min-heap keyed by node.mse (Algorithm 3 line 2)
    heap: List[Tuple[float, int]] = []
    for nid, node in G.nodes.items():
        node.recompute_plane()
        heapq.heappush(heap, (node.mse, nid))

    extracted: List[Node] = []
    planes: Dict[int, Tuple[np.ndarray, float]] = {}

    next_id = max(G.nodes.keys(), default=-1) + 1

    while heap:
        _, vid = heapq.heappop(heap)
        if vid not in G.nodes:
            continue
        v = G.nodes[vid]
        if not v.active:
            continue

        # Find best neighbor merge (Algorithm 3 lines 8-12)
        ubest_id = None
        best_stats = None
        best_mse = float("inf")
        neigh = list(G.adj.get(vid, []))

        for uid in neigh:
            if uid not in G.nodes:
                continue
            u = G.nodes[uid]
            if not u.active:
                continue
            merged_stats = v.stats.merge(u.stats)
            _, _, mse = merged_stats.pca_plane()
            if mse < best_mse:
                best_mse = mse
                ubest_id = uid
                best_stats = merged_stats

        # If no neighbor, treat as extraction attempt (isolated component)
        if ubest_id is None or best_stats is None:
            if v.stats.n > TNUM:
                extracted.append(v)
                planes[v.id] = (v.n_hat.copy(), float(v.d))
            # remove v
            for uid in list(G.adj.get(vid, [])):
                G.adj[uid].discard(vid)
            G.adj.pop(vid, None)
            G.nodes.pop(vid, None)
            continue

        # Merge fail? (Algorithm 3 line 13)
        if best_mse > TMSE:
            # extract if big enough else reject (Algorithm 3 lines 14-17)
            if v.stats.n > TNUM:
                extracted.append(v)
                planes[v.id] = (v.n_hat.copy(), float(v.d))
            # remove v from graph
            for uid in list(G.adj.get(vid, [])):
                G.adj[uid].discard(vid)
            G.adj.pop(vid, None)
            G.nodes.pop(vid, None)
            continue

        # Merge success (Algorithm 3 lines 18-21): edge contraction
        uid = ubest_id
        u = G.nodes[uid]

        new_blocks = v.blocks.union(u.blocks)
        new_node = Node(id=next_id, stats=best_stats, blocks=new_blocks)
        new_node.recompute_plane()

        # neighbors union minus {v,u}
        new_neigh = (G.adj.get(vid, set()) | G.adj.get(uid, set())) - {vid, uid}

        # Remove v and u from neighbors, add new node
        for w in list(G.adj.get(vid, set())):
            if w in G.adj:
                G.adj[w].discard(vid)
        for w in list(G.adj.get(uid, set())):
            if w in G.adj:
                G.adj[w].discard(uid)

        # delete old nodes
        G.adj.pop(vid, None)
        G.adj.pop(uid, None)
        G.nodes.pop(vid, None)
        G.nodes.pop(uid, None)

        # add new node
        G.nodes[next_id] = new_node
        G.adj[next_id] = set()
        for w in new_neigh:
            if w in G.nodes:
                G.adj[next_id].add(w)
                G.adj[w].add(next_id)

        heapq.heappush(heap, (new_node.mse, next_id))
        next_id += 1

    return extracted, planes


def nodes_to_label_image(
    points: np.ndarray,
    extracted: List[Node],
    block_h: int,
    block_w: int,
) -> np.ndarray:
    """Assign each pixel to its coarse extracted segment label using block membership."""
    M, N, _ = points.shape
    label = -np.ones((M, N), dtype=int)
    for seg_idx, node in enumerate(extracted):
        for (bi, bj) in node.blocks:
            r0 = bi * block_h
            c0 = bj * block_w
            r1 = min((bi + 1) * block_h, M)
            c1 = min((bj + 1) * block_w, N)
            # assign only valid points
            P = points[r0:r1, c0:c1]
            valid = ~np.isnan(P).any(axis=2)
            label[r0:r1, c0:c1][valid] = seg_idx
    return label


# ============================================================
# STEP 3: Segmentation Refinement (Algorithm 4)
# ============================================================

def segmentation_refinement(
    points: np.ndarray,
    coarse_nodes: List[Node],
    coarse_planes: Dict[int, Tuple[np.ndarray, float]],
    block_h: int,
    block_w: int,
    TMSE: float,
    TNUM: int = 800,
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

    merged_nodes, merged_planes_by_nodeid = ahc_cluster(G0, TMSE=TMSE, TNUM=TNUM)

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


# ============================================================
# Top-level wrapper (Algorithm 1)
# ============================================================

def fast_plane_extraction(
    points: np.ndarray,
    block_h: int = 10,
    block_w: int = 10,
    TMSE: float = 50.0**2,
    TANG_deg: float = 60.0,
    TNUM: int = 800,
    alpha: float = 0.02,
    do_refine: bool = True,
) -> Tuple[np.ndarray, Dict[int, Tuple[np.ndarray, float]]]:
    """
    Implements Algorithm 1: FASTPLANEEXTRACTION(F)
    """
    # Step 1
    G, _, _ = graph_initialization(
        points,
        block_h=block_h,
        block_w=block_w,
        TMSE=TMSE,
        TANG_deg=TANG_deg,
        alpha=alpha,
    )

    # Step 2
    coarse_nodes, coarse_planes = ahc_cluster(G, TMSE=TMSE, TNUM=TNUM)
    if not do_refine:
        label_img = nodes_to_label_image(points, coarse_nodes, block_h, block_w)
        # map segment index -> plane
        planes = {k: coarse_planes[coarse_nodes[k].id] for k in range(len(coarse_nodes)) if coarse_nodes[k].id in coarse_planes}
        return label_img, planes

    # Step 3
    refined_label, refined_planes = segmentation_refinement(
        points,
        coarse_nodes=coarse_nodes,
        coarse_planes=coarse_planes,
        block_h=block_h,
        block_w=block_w,
        TMSE=TMSE,
        TNUM=TNUM,
    )
    return refined_label, refined_planes


# ============================================================
# Example usage
# ============================================================


@click.command()
@click.option(
    "--input-image-path",
    "-ii",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to input image (.png).",
)
@click.option(
    "--organized-point-cloud-path",
    "-opc",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to organized point cloud (.npy).",
)
@click.option(
    "--output-path",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to save organized point cloud (.npy).",
)
@click.option(
    "--do-refine",
    "-d",
    type=bool,
    default=True,
    help="Whether to refine the planes.",
)
def main(input_image_path: Path, organized_point_cloud_path: Path, output_path: Path, do_refine: bool):
    points = np.load(organized_point_cloud_path)

    if points.shape[2] != 3:
        raise ValueError("Organized point cloud must have 3 channels")

    if points.shape[0] == 0 or points.shape[1] == 0:
        raise ValueError("Organized point cloud must have non-zero height and width")

    label_img, planes = fast_plane_extraction(
        points,
        block_h=10, block_w=10,
        TMSE=50.0**2,
        TANG_deg=60.0,
        TNUM=800,
        alpha=0.02,
        do_refine=do_refine
    )

    print("Found planes:", len(planes))
    for k, (n, d) in planes.items():
        print(k, "n=", n, "d=", d)

    np.save(output_path, label_img)

    # Visualize input image with label overlay
    _visualize_planes_plotly(input_image_path, label_img, opacity=0.5)


def _get_label_colors(n_labels: int) -> np.ndarray:
    """Generate n_labels distinct RGB colors (0-255)."""
    colors = []
    for i in range(max(n_labels, 1)):
        hue = (i * 0.618033988749895) % 1.0  # golden ratio for spreading
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(colors, dtype=np.uint8)


def _visualize_planes_plotly(
    input_image_path: Path,
    label_img: np.ndarray,
    opacity: float = 0.5,
) -> None:
    """Visualize input image with label overlay using plotly."""
    img = cv2.imread(str(input_image_path))
    if img is None:
        print(f"Could not load image: {input_image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H_label, W_label = label_img.shape
    H_img, W_img = img_rgb.shape[:2]
    if (H_img, W_img) != (H_label, W_label):
        img_rgb = cv2.resize(img_rgb, (W_label, H_label), interpolation=cv2.INTER_LINEAR)

    img_float = img_rgb.astype(np.float64) / 255.0

    unique_labels = np.unique(label_img)
    unique_labels = unique_labels[unique_labels >= 0]
    n_labels = len(unique_labels)
    colors = _get_label_colors(n_labels)

    overlay = img_float.copy()
    for i, lbl in enumerate(unique_labels):
        mask = label_img == lbl
        if not np.any(mask):
            continue
        color = colors[i].astype(np.float64) / 255.0
        alpha = opacity * mask.astype(np.float64)[:, :, np.newaxis]
        overlay = overlay * (1 - alpha) + color * alpha

    overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Input image", "Plane overlay"))
    fig.add_trace(go.Image(z=img_rgb), row=1, col=1)
    fig.add_trace(go.Image(z=overlay_uint8), row=1, col=2)
    fig.update_layout(
        width=1200,
        height=600,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.show()


if __name__ == "__main__":
    main()
