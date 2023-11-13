#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:56:16 2023

@author: jj
"""
import numpy as np
import numpy.linalg
import os
import open3d as o3d
import scipy
from PIL import Image
import sys
import copy

def T_pcd(T, pcd):
    '''
    simple function for transforming 3D points, T @ pcd
    :param T: 4x4 transformation
    :param pcd: Nx3 point cloud array
    :return: Nx3
    '''
    return (T @ np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1).T).T[:, :3]

def emulate_lidar_hitpts(pcd, w_T_cams, axis_aligned_T_w, m2f_seg, vacuum_height = 0.15, cams_used_ray_cast=None):
    '''
    emulate Lidar hit points 
    :param pcd: reconstructed point cloud in the original frame, usually the first camera frame
    :type pcd: o3d.geometry.PointCloud
    :param w_T_cams: (M,4,4), downsampled camera poses from the video, used to downproject to RVC height for ray casting
    :param axis_aligned_T_w: the estimated 4x4 transformation to transform reconstructed scene to a frame with z-axis pointing up
    :param m2f_seg: (N,) segmentation labels for the reconstructed point cloud, used for filter out floor points, 
                            if none, filter out floor points with height
    :param vacuum_height: predefined height for emulating LiDAR hit points, defaults to 0.15
    :param cams_used_ray_cast: to specify which cams to use for ray casting, defaults to None, optional

    '''
    # downsample to avoid oom
    pcd, iddx, aa = pcd.voxel_down_sample_and_trace(0.005, pcd.get_min_bound(), pcd.get_max_bound(), False)
    idx = np.array([i[0] for i in aa])
    m2f_seg = m2f_seg[idx] if m2f_seg is not None else None
    #
    w_T_vacuums = np.zeros(w_T_cams.shape)
    floor_z_in_w = axis_aligned_T_w[2,:3]
    
    for i in range(w_T_cams.shape[0]):
        axis_aligned_T_cam = axis_aligned_T_w @ w_T_cams[i]
        axis_aligned_T_vacuum = np.eye(4)
        axis_aligned_T_vacuum[:3,-1] = axis_aligned_T_cam[:3,-1]
        axis_aligned_T_vacuum[2,-1] = vacuum_height
        axis_aligned_T_vacuum[:3,1] = np.array([0,0,-1]) #y
        z_ = axis_aligned_T_cam[:3,2]
        z = z_ - np.dot(z_,np.array([0,0,-1]))*np.array([0,0,-1]) 
        axis_aligned_T_vacuum[:3,2] = z/np.linalg.norm(z)
        x = np.cross(axis_aligned_T_vacuum[:3,1],axis_aligned_T_vacuum[:3,2])
        axis_aligned_T_vacuum[:3,0] = x/np.linalg.norm(x)
        w_T_vacuums[i] = np.linalg.inv(axis_aligned_T_w) @ axis_aligned_T_vacuum
    
    # filter floor
    if m2f_seg is not None:
        pts = np.array(pcd.points)[m2f_seg!=2] 
        pts_2d_all_w_seg = m2f_seg[m2f_seg!=2]
    else:
        # if no segmentation info, filter out floor with height
        aligned_pts = T_pcd(axis_aligned_T_w, np.array(pcd.points)) 
        aligned_pts = aligned_pts[aligned_pts[:,-1] > 0.1]
        pts = T_pcd(np.linalg.inv(axis_aligned_T_w), aligned_pts) 
        pts_2d_all_w_seg = -np.ones(pts.shape[0])
    
    pts_3d_axis_aligned = T_pcd(axis_aligned_T_w, pts)
    z_ = np.array([0,0,1])
    pts_2d_all_axis_aligned = pts_3d_axis_aligned - (pts_3d_axis_aligned @ z_[:,None]) * z_[None,:]
    pts_2d_all_w = T_pcd(np.linalg.inv(axis_aligned_T_w), pts_2d_all_axis_aligned)
    
    fov = 180
    num_rays = 300
    delta = 3
    ray_angles = np.linspace(-np.deg2rad(fov/2),  np.deg2rad(fov/2), num_rays)
    ray_directions = np.array([np.sin(ray_angles), np.zeros_like(ray_angles), np.cos(ray_angles)]).T
    sensor_pos = np.array([0,0,0])
    
    cam_lidar_2d = np.zeros((0,3))
    cam_lidar_2d_seg = np.zeros((0,1))
    record_used_cam = True if cams_used_ray_cast is None else False
    cams_used_ray_cast = [] if cams_used_ray_cast is None else cams_used_ray_cast
    for i in range(w_T_vacuums.shape[0]):
        axis_aligned_T_vacuum_ = axis_aligned_T_w @ w_T_vacuums[i]
        if record_used_cam:
            collision = scipy.spatial.distance.cdist(pts_2d_all_axis_aligned[:,:2],axis_aligned_T_vacuum_[:2,-1][None,:],'euclidean') < 0.1
            if np.sum(np.any(collision,axis=-1)) > 10: continue
            cams_used_ray_cast.append(i)
        else:
            if not i in cams_used_ray_cast: continue
        
        vacuum_pts = T_pcd(np.linalg.inv(w_T_vacuums[i]),pts)
        
        vec = vacuum_pts-sensor_pos[None,...]
        sub_pts = vacuum_pts[np.isclose(vec[:,1],0,atol=1e-1)]
        sub_labels = pts_2d_all_w_seg[np.isclose(vec[:,1],0,atol=1e-1)]
        vec = vec[np.isclose(vec[:,1],0,atol=1e-1)]
        vec_unit = vec/np.linalg.norm(vec,axis=-1)[...,None]

        angles = np.rad2deg(np.arccos(ray_directions @ vec_unit.T)) # num_rays x num_points
        hit_mask = np.isclose(angles,0,atol=delta) # num_rays x num_points
        pts_distance = np.linalg.norm(vec, axis=-1)[None,:].repeat(hit_mask.shape[0],axis=0) 
        masked_distance = copy.copy(pts_distance)
        masked_distance[~hit_mask] = 99999999 # some large number
        idx = np.argmin(masked_distance,axis=1)
        idx = idx[np.min(masked_distance,axis=-1) != 99999999]
        idx = np.unique(idx)
        hit_pts = sub_pts[idx]
        hit_pts_labels = sub_labels[idx]
        if not hit_pts.shape[0] == 0: 
            cam_lidar_2d = np.vstack((cam_lidar_2d,T_pcd(w_T_vacuums[i],hit_pts)))
            cam_lidar_2d_seg = np.append(cam_lidar_2d_seg, hit_pts_labels)
    return cam_lidar_2d, cam_lidar_2d_seg, pts_2d_all_w, pts_2d_all_w_seg, w_T_vacuums, cams_used_ray_cast

