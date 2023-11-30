import numpy as np
import bisect
from collections import OrderedDict
import math
import matplotlib.tri as tri
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import linemerge
from sklearn.cluster import DBSCAN
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d import _C


class ConcaveHull:

    def __init__(self):
        self.triangles = {}
        self.crs = {}

    def loadpoints(self, points):
        # self.points = np.array(points)
        self.points = points

    def edge(self, key, triangle):
        '''Calculate the length of the triangle's outside edge
        and returns the [length, key]'''
        pos = triangle[1].index(-1)
        if pos == 0:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][1]]
        elif pos == 1:
            x1, y1 = self.points[triangle[0][1]]
            x2, y2 = self.points[triangle[0][2]]
        elif pos == 2:
            x1, y1 = self.points[triangle[0][0]]
            x2, y2 = self.points[triangle[0][2]]
        length = ((x1-x2)**2+(y1-y2)**2)**0.5
        rec = [length, key]
        return rec

    def triangulate(self):

        if len(self.points) < 2:
            raise Exception('CountError: You need at least 3 points to Triangulate')

        temp = list(zip(*self.points))
        x, y = list(temp[0]), list(temp[1])
        del (temp)

        triang = tri.Triangulation(x, y)

        self.triangles = {}

        for i, triangle in enumerate(triang.triangles):
            self.triangles[i] = [list(triangle), list(triang.neighbors[i])]

    def calculatehull(self, tol=50):

        self.tol = tol

        if len(self.triangles) == 0:
            self.triangulate()

        # All triangles with one boundary longer than the tolerance (self.tol)
        # is added to a sorted deletion list.
        # The list is kept sorted from according to the boundary edge's length
        # using bisect
        deletion = []
        self.boundary_vertices = set()
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, neigh in enumerate(triangle[1]):
                    if neigh == -1:
                        if pos == 0:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][1])
                        elif pos == 1:
                            self.boundary_vertices.add(triangle[0][1])
                            self.boundary_vertices.add(triangle[0][2])
                        elif pos == 2:
                            self.boundary_vertices.add(triangle[0][0])
                            self.boundary_vertices.add(triangle[0][2])
            if -1 in triangle[1] and triangle[1].count(-1) == 1:
                rec = self.edge(i, triangle)
                if rec[0] > self.tol and triangle[1].count(-1) == 1:
                    bisect.insort(deletion, rec)

        while len(deletion) != 0:
            # The triangles with the longest boundary edges will be
            # deleted first
            item = deletion.pop()
            ref = item[1]
            flag = 0

            # Triangle will not be deleted if it already has two boundary edges
            if self.triangles[ref][1].count(-1) > 1:
                continue

            # Triangle will not be deleted if the inside node which is not
            # on this triangle's boundary is already on the boundary of
            # another triangle
            adjust = {0: 2, 1: 0, 2: 1}
            for i, neigh in enumerate(self.triangles[ref][1]):
                j = adjust[i]
                if neigh == -1 and self.triangles[ref][0][j] in self.boundary_vertices:
                    flag = 1
                    break
            if flag == 1:
                continue

            for i, neigh in enumerate(self.triangles[ref][1]):
                if neigh == -1:
                    continue
                pos = self.triangles[neigh][1].index(ref)
                self.triangles[neigh][1][pos] = -1
                rec = self.edge(neigh, self.triangles[neigh])
                if rec[0] > self.tol and self.triangles[rec[1]][1].count(-1) == 1:
                    bisect.insort(deletion, rec)

            for pt in self.triangles[ref][0]:
                self.boundary_vertices.add(pt)

            del self.triangles[ref]

        self.polygon()

    def polygon(self):

        edgelines = []
        for i, triangle in self.triangles.items():
            if -1 in triangle[1]:
                for pos, value in enumerate(triangle[1]):
                    if value == -1:
                        if pos == 0:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][1]]
                        elif pos == 1:
                            x1, y1 = self.points[triangle[0][1]]
                            x2, y2 = self.points[triangle[0][2]]
                        elif pos == 2:
                            x1, y1 = self.points[triangle[0][0]]
                            x2, y2 = self.points[triangle[0][2]]
                        line = LineString([(x1, y1), (x2, y2)])
                        edgelines.append(line)

        bound = linemerge(edgelines)

        self.boundary = Polygon(bound.coords)


def find_largest_cluster_3D(points, eps=0.2, min_samples=50):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    n_clusters_ = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    dense_cluster = np.zeros((0, 3))
    cur_len = 0
    for c in range(n_clusters_):
        cluster = points[clustering.labels_ == c]
        m_ = np.mean(cluster, axis=0)
        C = cluster - m_[None, :]
        V = np.cov(C.T)
        eigenValues, eigenVectors = np.linalg.eig(V)
        idx = eigenValues.argsort()[::-1]
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        proj_of_u_on_v = np.dot(C, eigenVectors[:, 0])
        max_id = np.argmax(proj_of_u_on_v)
        min_id = np.argmin(proj_of_u_on_v)
        max_len = np.linalg.norm(cluster[max_id]-cluster[min_id])
        if max_len > cur_len:
            dense_cluster = cluster
            cur_len = max_len
    return dense_cluster


def compute_normal_ordered_points(boundary_points):
    boundary_points_shift = np.roll(boundary_points, -2, axis=0)
    diffs = boundary_points_shift - boundary_points
    diffs = np.roll(diffs, 1, axis=0)
    thetas = np.arctan2(diffs[:, 1], diffs[:, 0])
    normal_thetas = thetas - np.pi / 2
    normal_vecs = np.stack((np.cos(normal_thetas), np.sin(normal_thetas), np.zeros((len(normal_thetas))))).T
    return normal_vecs


def sort_boundary_point_from_extreme(boundary_points, approx_extreme_pts_idx, approx_far_pts_idx):
    ordered_boundary_points = np.roll(boundary_points, -approx_extreme_pts_idx, axis=0)
    ordered_boundary_points = ordered_boundary_points if approx_extreme_pts_idx - \
        approx_far_pts_idx > 0 else ordered_boundary_points[::-1, :]
    normal_vecs = compute_normal_ordered_points(ordered_boundary_points)
    normal_vecs = normal_vecs if approx_extreme_pts_idx-approx_far_pts_idx > 0 else -normal_vecs
    return ordered_boundary_points, normal_vecs


def sample_farthest_points(
    points: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    K: Union[int, List, torch.Tensor] = 50,
    random_start_point: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative farthest point sampling algorithm [1] to subsample a set of
    K points from a given pointcloud. At each iteration, a point is selected
    which has the largest nearest neighbor distance to any of the
    already selected points.

    Farthest point sampling provides more uniform coverage of the input
    point cloud compared to uniform random sampling.

    [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        on Point Sets in a Metric Space", NeurIPS 2017.

    Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

    Returns:
        selected_points: (N, K, D), array of selected values from points. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            0.0 for batch elements where k_i < max(K).
        selected_indices: (N, K) array of selected indices. If the input
            K is a tensor, then the shape will be (N, max(K), D), and padded with
            -1 for batch elements where k_i < max(K).
    """
    N, P, D = points.shape
    device = points.device

    # Validate inputs
    if lengths is None:
        lengths = torch.full((N,), P, dtype=torch.int64, device=device)
    else:
        if lengths.shape != (N,):
            raise ValueError("points and lengths must have same batch dimension.")
        if lengths.max() > P:
            raise ValueError("A value in lengths was too large.")

    # TODO: support providing K as a ratio of the total number of points instead of as an int
    if isinstance(K, int):
        K = torch.full((N,), K, dtype=torch.int64, device=device)
    elif isinstance(K, list):
        K = torch.tensor(K, dtype=torch.int64, device=device)

    if K.shape[0] != N:
        raise ValueError("K and points must have the same batch dimension")

    # Check dtypes are correct and convert if necessary
    if not (points.dtype == torch.float32):
        points = points.to(torch.float32)
    if not (lengths.dtype == torch.int64):
        lengths = lengths.to(torch.int64)
    if not (K.dtype == torch.int64):
        K = K.to(torch.int64)

    # Generate the starting indices for sampling
    start_idxs = torch.zeros_like(lengths)
    if random_start_point:
        for n in range(N):
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            start_idxs[n] = torch.randint(high=lengths[n], size=(1,)).item()

    with torch.no_grad():
        # pyre-fixme[16]: `pytorch3d_._C` has no attribute `sample_farthest_points`.
        idx = _C.sample_farthest_points(points, lengths, K, start_idxs)
    sampled_points = masked_gather(points, idx)

    return sampled_points, idx


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


def sample_points_along_ordered_vertices(convex_hull_vertex, pts_dis=0.1):
    points = np.zeros((0, 3))
    for i in range(convex_hull_vertex.shape[0]):
        start_pts = convex_hull_vertex[i-1]
        end_pts = convex_hull_vertex[i]
        D = np.linalg.norm(end_pts-start_pts)
        n_pts = round(D/pts_dis)
        for j in range(n_pts):
            points = np.vstack((points, ((n_pts-j)/n_pts)*start_pts + (j/n_pts)*end_pts))
    return points
