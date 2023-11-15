#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:07:18 2023

@author: jj
"""
import numpy as np
import open3d as o3d
import os
from run_redwood import viola_matching
import argparse
import torch
import pytorch3d
from pytorch3d.ops import sample_farthest_points

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--scene_name",
        default='arcore-dataset-2023-10-25-16-57-43',
        help="folder name to the APP output"
    )
    parser.add_argument(
        "--data_path",
        default='/labdata/selim/video2floorplan/galaxy_scans/office',
        help="directory to where all recordings are saved"
    )       
    parser.add_argument(
        "--lidar_path",
        default='/labdata/junjee/data/our_itw/office/lidar/office.ply',
        help="path to 2D LiDAR scan"
    )     
    parser.add_argument(
        "--pose_init_method",
        default='grid',
        choices=['grid','img_cross_correlation'],
        help="method for pose initialization between 2D point clouds"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()    
    
    print('start viola...')
    # load and process LiDAR map
    lidar_pcd = o3d.io.read_point_cloud(args.lidar_path)
    template_2d = np.array(lidar_pcd.points)
    _, pts_idx = sample_farthest_points(torch.Tensor(template_2d).unsqueeze(0), K=1800)
    pts_idx = pts_idx[0].numpy()
    template_2d = template_2d[pts_idx]
    template_2d[:,-1] = 1
    
    ## load simplerecon result and run viola matching
    data = np.load(f'{args.data_path}/{args.scene_name}/viola/viola_input.npy',allow_pickle=True).item()
    result = viola_matching(data, template_2d, init_method=args.pose_init_method, vis=False)
    est_T = result['est_T']
    
    os.chmod(f'{args.data_path}/{args.scene_name}/viola', 0o777)
    np.save(f'{args.data_path}/{args.scene_name}/viola/est_T.npy',est_T)
    
    # visualization
    est_patial_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((est_T @ data['partial_2d'].T).T))
    est_patial_pcd.paint_uniform_color([1,0,0])
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
            
    o3d.visualization.draw_geometries([est_patial_pcd,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_2d))])
    est_pose_3d = np.eye(4)
    est_pose_3d[:2,:2] = est_T[:2,:2]
    est_pose_3d[:2,-1] = est_T[:2,-1]
    
    try:
        recon = o3d.io.read_triangle_mesh(f'{args.data_path}/{args.scene_name}/viola/{args.scene_name}.ply')
        if np.asarray(recon.triangles).shape[0] == 0:
            recon = o3d.io.read_point_cloud(f'{args.data_path}/{args.scene_name}/viola/{args.scene_name}.ply')
    except:
        recon = o3d.io.read_point_cloud(f'{args.data_path}/{args.scene_name}/viola/{args.scene_name}.ply')
    recon.transform(data['axis_aligned_T_w'])
    recon.transform(est_pose_3d)
    template_points = template_2d
    template_points[:,-1] = 0
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_points)),recon])
    import pdb;pdb.set_trace()

