import numpy as np
import torch
import open3d as o3d
import copy
import shapely
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from utils.geometry_utils import T_pcd, keypoints_to_spheres
from .view_selection_utils import ConcaveHull, find_largest_cluster_3D, sort_boundary_point_from_extreme, sample_farthest_points, compute_normal_ordered_points, sample_points_along_ordered_vertices


def inpainting_view_selection(data, o3d_pcd=None, vis=False):
    im_size = (512, 384)
    W, H = im_size
    downprojected_unseen, extreme_pts, ordered_boundary_points, normal_vecs, coverage = check_unseen_part(
        data, vis=False)
    print('coverage', coverage)
    d_to_check = 200
    kps_distance = 50
    d_to_check = d_to_check if ordered_boundary_points.shape[0] > d_to_check else int(
        ordered_boundary_points.shape[0]/2)
    keypoints_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
        ordered_boundary_points[:d_to_check:kps_distance, :]))
    keypoints_pcd.normals = o3d.utility.Vector3dVector(normal_vecs[:d_to_check:kps_distance, :])

    # import pdb;pdb.set_trace()
    extreme_pts[-1] = 0.15
    extreme_cams = []
    extreme_cam = None
    mid_u = im_size[0]/2
    z_min = 0
    # find extreme cam
    for i in range(data['w_T_cams'].shape[0]):
        axis_aligned_T_cam = data['axis_aligned_T_w'] @ data['w_T_cams'][i]
        z_cam = axis_aligned_T_cam[2, 2]
        pts_current_frame = T_pcd(np.linalg.inv(axis_aligned_T_cam), extreme_pts[None, :])
        pts_canvas = ((data['K']) @ pts_current_frame.T).T
        pts_canvas = np.round((pts_canvas/pts_canvas[:, -1][..., None])[:, :2]).astype(np.int16)
        if pts_canvas[0][0] > 0 and pts_canvas[0][0] < im_size[0]-0.5 and pts_canvas[0][1] > 0 and pts_canvas[0][1] < im_size[1]-0.5:
            extreme_cams.append(data['w_T_cams'][i])
            if z_cam < z_min:
                z_min = z_cam
                extreme_cam = data['w_T_cams'][i]
    extreme_cams = np.array(extreme_cams)  # w_T_cams
    if vis:
        cfs_extreme_cams = [o3d.geometry.TriangleMesh.create_coordinate_frame(
            0.5) for _ in range(extreme_cams.shape[0])]
        for ii, cf_ in enumerate(cfs_extreme_cams):
            cf_.transform(extreme_cams[ii])
        cf_extreme_cam = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
        cf_extreme_cam.transform(extreme_cam)
        o3d.visualization.draw_geometries([o3d_pcd, cf_extreme_cam] + cfs_extreme_cams)

    K = data['K']

    floor_T_camera = data['axis_aligned_T_w']
    camera_T_floor = np.linalg.inv(floor_T_camera)

    axis_aligned_T_start_cam = data['axis_aligned_T_w'] @ extreme_cam
    # first 2 camera: move back
    z_direction = axis_aligned_T_start_cam[:3, 2]
    target_poses = []
    move_back_step = 0.2
    see_floor = False
    direction = None
    while not see_floor:
        axis_aligned_T_start_cam[:3, 3] -= z_direction * move_back_step
        target_poses.append(copy.copy(axis_aligned_T_start_cam))
        pts_current_frame = T_pcd(np.linalg.inv(axis_aligned_T_start_cam), np.array(keypoints_pcd.points))
        pts_canvas = ((data['K']) @ pts_current_frame.T).T
        pts_canvas = np.round((pts_canvas/pts_canvas[:, -1][..., None])[:, :2]).astype(np.int16)
        mask = np.logical_and(pts_canvas[:, 0] < W-0.5, pts_canvas[:, 0] >= 0)
        mask = np.logical_and(mask, pts_canvas[:, 1] >= 0)  # v
        mask = np.logical_and(mask, pts_canvas[:, 1] < H-0.5)  # v
        direction = 'left' if np.mean(pts_canvas[:, 0]) < W/2 else 'right'
        if sum(mask) >= mask.shape[0]/2:
            see_floor = True
    if len(target_poses) < 2:
        axis_aligned_T_start_cam[:3, 3] -= z_direction * move_back_step
        target_poses.append(copy.copy(axis_aligned_T_start_cam))

    # then rotate until see most of the keypoints
    see_most = False
    max_rotate = 3 + len(target_poses)
    while not see_most:
        if direction == 'left':
            axis_aligned_T_start_cam[:3, :3] = R.from_euler(
                'z', 10, degrees=True).as_matrix() @ axis_aligned_T_start_cam[:3, :3]
            target_poses.append(copy.copy(axis_aligned_T_start_cam))
        else:
            axis_aligned_T_start_cam[:3, :3] = R.from_euler(
                'z', -10, degrees=True).as_matrix() @ axis_aligned_T_start_cam[:3, :3]
            target_poses.append(copy.copy(axis_aligned_T_start_cam))
        pts_current_frame = T_pcd(np.linalg.inv(axis_aligned_T_start_cam), np.array(keypoints_pcd.points))
        pts_canvas = ((data['K']) @ pts_current_frame.T).T
        pts_canvas = np.round((pts_canvas/pts_canvas[:, -1][..., None])[:, :2]).astype(np.int16)
        mask = np.logical_and(pts_canvas[:, 0] < W-0.5, pts_canvas[:, 0] >= 0)
        mask = np.logical_and(mask, pts_canvas[:, 1] >= 0)  # v
        mask = np.logical_and(mask, pts_canvas[:, 1] < H-0.5)  # v
        if sum(mask) == mask.shape[0] or len(target_poses) == max_rotate:
            see_most = True
    return target_poses


def check_unseen_part(scene_data, vis=False):
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(0.3)
    hitpts = scene_data['partial_2d']
    hitpts[:, -1] = 0
    # first fit a concave hull to the largest cluster in the ray casted hitpoints
    filtered = find_largest_cluster_3D(hitpts, eps=0.5, min_samples=3)
    hitpts = filtered
    try:
        ch = ConcaveHull()
        ch.loadpoints(filtered[:, :2])
        ch.calculatehull(tol=0.9)
    except:
        return None, None, None, None, 0
    boundary_points = np.vstack(ch.boundary.exterior.coords.xy).T
    boundary_points = np.hstack((boundary_points, hitpts[0, -1]*np.ones((boundary_points.shape[0], 1))))

    _, pts_idx = sample_farthest_points(torch.Tensor(scene_data['droid_2d_pts_all']).unsqueeze(0).cuda(), K=3000)
    pts_idx = pts_idx[0].cpu().numpy()
    downprojected = scene_data['droid_2d_pts_all'][pts_idx]
    downprojected = T_pcd(scene_data['axis_aligned_T_w'], downprojected)
    if vis:
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(boundary_points),
            lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(boundary_points.shape[0]-1)]),
        )
        line_set.paint_uniform_color([1, 0, 0])
        hit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hitpts))
        hit_pcd.paint_uniform_color([0, 0, 0])
        o3d.visualization.draw_geometries([cf, line_set, hit_pcd,
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downprojected))])
    # remove downprojected points that lies inside the concave hull
    mp = map(shapely.Point, zip(downprojected[:, 0], downprojected[:, 1]))
    geoms = np.array([i for i in mp])
    inside_poly = shapely.contains(ch.boundary, geoms)
    downprojected_unseen = downprojected[~inside_poly]
    # sample points on the concave hull boundary and compute normal
    boundary_points = sample_points_along_ordered_vertices(boundary_points, pts_dis=0.01)
    # normal estimation
    normal_vecs = compute_normal_ordered_points(boundary_points)

    if vis:
        boundary_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boundary_points))
        boundary_pcd.paint_uniform_color([1, 0, 0])
        boundary_pcd.normals = o3d.utility.Vector3dVector(normal_vecs)
        o3d.visualization.draw_geometries([cf, boundary_pcd,
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hitpts)),
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downprojected_unseen))])
    # remove points that lies along the estimated boundary normal so that we keep only most of the unseen area
    vec = downprojected_unseen[:, None, :].repeat(
        boundary_points.shape[0], 1) - boundary_points[None, :, :].repeat(downprojected_unseen.shape[0], 0)
    vec_norm = np.min(np.linalg.norm(vec, axis=-1), axis=1)
    downprojected_unseen = downprojected_unseen[vec_norm != 0]
    vec = downprojected_unseen[:, None, :].repeat(
        boundary_points.shape[0], 1) - boundary_points[None, :, :].repeat(downprojected_unseen.shape[0], 0)
    norms_ = normal_vecs[None, :, :].repeat(downprojected_unseen.shape[0], 0)
    dis_mask = np.linalg.norm(vec, axis=-1) < 0.3
    vec = vec/np.linalg.norm(vec, axis=-1)[..., None]
    dot_p = (vec[:, :, None, :] @ norms_[..., None])[:, :, 0, 0]
    angle_mask = np.abs(np.rad2deg(np.arccos(dot_p))) < 10
    remove_mask = np.logical_and(dis_mask, angle_mask)
    remove_mask = np.any(remove_mask, axis=1)
    downprojected_unseen = downprojected_unseen[~remove_mask]
    if downprojected_unseen.shape[0] == 0:
        return None, None, None, None, 1
    #
    downprojected_unseen = find_largest_cluster_3D(downprojected_unseen, eps=0.2, min_samples=20)
    if vis:
        hit_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(hitpts))
        hit_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries(
            [cf, hit_pcd, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downprojected))])
        o3d.visualization.draw_geometries([cf, hit_pcd, o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(downprojected_unseen))])

    coverage = (downprojected.shape[0] - downprojected_unseen.shape[0])/downprojected.shape[0]
    # find approximated extreme points in the ray casted points, and find the extreme point that is closest to the uncovered area
    m_ = np.mean(hitpts, axis=0)
    C = hitpts - m_[None, :]
    V = np.cov(C.T)
    eigenValues, eigenVectors = np.linalg.eig(V)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    proj_of_u_on_v = np.dot(C, eigenVectors[:, 0])
    max_id = np.argmax(proj_of_u_on_v)
    min_id = np.argmin(proj_of_u_on_v)
    extreme_pts = np.vstack((hitpts[max_id][None, :], hitpts[min_id][None, :]))
    far_extreme_pts = extreme_pts[np.argmax(np.min(cdist(extreme_pts, downprojected_unseen), axis=-1))]
    # the point that is closer to unseen part
    extreme_pts = extreme_pts[np.argmin(np.min(cdist(extreme_pts, downprojected_unseen), axis=-1))]
    if vis:
        kps = keypoints_to_spheres(extreme_pts[None, :], radius=0.1)
        o3d.visualization.draw_geometries([cf, hit_pcd, kps, o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(downprojected_unseen))])
    # Fit a concave hull to the full downprojected points
    downprojected_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downprojected))
    downprojected_pcd, _ = downprojected_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    downprojected = np.asarray(downprojected_pcd.points)

    ch = ConcaveHull()
    ch.loadpoints(downprojected[:, :2])
    ch.calculatehull(tol=0.5)
    boundary_points = np.vstack(ch.boundary.exterior.coords.xy).T
    boundary_points = np.hstack((boundary_points, hitpts[0, -1]*np.ones((boundary_points.shape[0], 1))))
    boundary_points = sample_points_along_ordered_vertices(boundary_points, pts_dis=0.01)
    approx_far_pts_idx = np.argmin(cdist(far_extreme_pts[None, :], boundary_points)[0])
    approx_extreme_pts_idx = np.argmin(cdist(extreme_pts[None, :], boundary_points)[0])
    # order the boundary of the downpojected points, the first point is extreme point and the order goes along unseen part
    ordered_boundary_points, normal_vecs = sort_boundary_point_from_extreme(
        boundary_points, approx_extreme_pts_idx, approx_far_pts_idx)
    reversed_ordered_boundary_points, reversed_normal_vecs = sort_boundary_point_from_extreme(
        boundary_points, approx_far_pts_idx, approx_extreme_pts_idx)
    # check the direction with wall, avoid going into walls
    wall_pts = scene_data['droid_2d_pts_all'][scene_data['m2f_seg'] == 1]
    _, pts_idx = sample_farthest_points(torch.Tensor(wall_pts).unsqueeze(0).cuda(), K=3000)
    wall_pts = wall_pts[pts_idx[0].cpu().numpy()]
    wall_pts = T_pcd(scene_data['axis_aligned_T_w'], wall_pts)
    d_to_check = 200 if boundary_points.shape[0] > 200 else int(boundary_points.shape[0]/2)
    ordered_pts = ordered_boundary_points[:d_to_check:10, :]
    rev_ordered_pts = reversed_ordered_boundary_points[:d_to_check:10, :]
    if np.sum(np.min(cdist(ordered_pts, wall_pts), axis=1)) < np.sum(np.min(cdist(rev_ordered_pts, wall_pts), axis=1)):
        ordered_boundary_points, reversed_ordered_boundary_points = reversed_ordered_boundary_points, ordered_boundary_points
        normal_vecs, reversed_normal_vecs = reversed_normal_vecs, normal_vecs
        extreme_pts, far_extreme_pts = far_extreme_pts, extreme_pts
        approx_extreme_pts_idx, approx_far_pts_idx = approx_far_pts_idx, approx_extreme_pts_idx

    # import pdb;pdb.set_trace()
    if vis:
        trajectory = keypoints_to_spheres(ordered_boundary_points[:200:10, :], radius=0.03)
        approx_kps = keypoints_to_spheres(
            np.vstack((boundary_points[approx_far_pts_idx][None, :], boundary_points[approx_extreme_pts_idx][None, :])), radius=0.1)
        boundary_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(boundary_points))
        boundary_pcd.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([cf, hit_pcd, kps, trajectory, approx_kps,
                                          boundary_pcd, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(downprojected))])

    return downprojected_unseen, extreme_pts, ordered_boundary_points, normal_vecs, coverage
