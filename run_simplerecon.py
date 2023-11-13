#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:07:18 2023

@author: jj
"""
import numpy as np
import open3d as o3d
import os
import torch
from utils import geometry_utils, optimization_utils, cross_correlation_utils
import torch.optim as optim
import time
import argparse
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
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()    
    
    vis = False
    iterations = 180
    print('start viola...')
    ## load simplerecon result
    data = np.load(f'{args.data_path}/{args.scene_name}/viola/viola_input.npy',allow_pickle=True).item()
    w_T_cams = data['w_T_cams'] #w_T_cam
    axis_aligned_T_w = data['axis_aligned_T_w']
    m2f_seg = data['m2f_seg']
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['pts']))
    scale = 1
    ## simulate LiDAR hit points
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(1)
    pts_2d, _, _, _, _, _ = geometry_utils.emulate_lidar_hitpts(pcd, w_T_cams, axis_aligned_T_w, m2f_seg, vacuum_height = 0.2/scale)
    _, pts_idx = sample_farthest_points(torch.Tensor(pts_2d).unsqueeze(0), K=500)
    pts_idx = pts_idx[0].numpy()
    pts_2d = pts_2d[pts_idx]
    partial_2d = geometry_utils.T_pcd(axis_aligned_T_w, pts_2d)
    
    partial_2d *= scale
    partial_2d[:,-1] = 1
    data['partial_2d'] = partial_2d
    
    # process LiDAR map
    lidar_pcd = o3d.io.read_point_cloud(args.lidar_path)
    template_2d = np.array(lidar_pcd.points)
    _, pts_idx = sample_farthest_points(torch.Tensor(template_2d).unsqueeze(0), K=1800)
    pts_idx = pts_idx[0].numpy()
    template_2d = template_2d[pts_idx]
    template_2d[:,-1] = 1
    data['template_2d'] = template_2d
    # 
    if vis:
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_2d)), cf])
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_2d)), 
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_2d)), cf])

    print('pose init')
    img_cor_t0 = time.time()
    #init_poses = cross_correlation_utils.get_cross_correlation_matches(partial_2d, template_2d, False, False)
    init_poses = optimization_utils.grid_initialization(template_2d,grid_n = 10)
    print('pose init time', time.time()-img_cor_t0)
    if vis:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
        sp_loc = geometry_utils.keypoints_to_spheres((init_poses @ np.array([0,0,1])[None,:,None])[...,0], radius=0.1)
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['template_2d'])),sp_loc,cf])
    #
    template_tensor = torch.Tensor(template_2d).unsqueeze(0).repeat(init_poses.shape[0],1,1).cuda()
    partial_2d = torch.Tensor(partial_2d).cuda()
    
    torch.cuda.empty_cache()
    est_4D = optimization_utils.T24D(torch.tensor(init_poses)).float().cuda()
    est_4D = est_4D.detach().requires_grad_().cuda()
    optimizer = optim.Adam([est_4D],lr=0.1)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.5)
    T_vis = {}
    T_vis['partial_2d'] = partial_2d
    T_vis['template_2d'] = template_2d
    #T_vis['T_intermediate'] = T_intermediate
    #np.save(f'{path}/{seq_name}_2D_points.npy',T_vis)
    print('optimization...')
    for it in range(iterations):
        optimizer.zero_grad()
        
        T = optimization_utils.Out4D2T(est_4D)
        est_t = (T @ partial_2d.T).permute(0,2,1)
        T_previous = T.clone().detach()        
        batch_loss = optimization_utils.chamfer_loss(est_t, template_tensor)
        loss = torch.sum(batch_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
                
        if it > 120:
            if optimization_utils.check_converge(T_previous, optimization_utils.Out4D2T(est_4D).clone().detach()):
                break
        torch.cuda.empty_cache()
    T = optimization_utils.Out4D2T(est_4D)
    est_t = (T @ partial_2d.T).permute(0,2,1)
    batch_loss = optimization_utils.chamfer_loss(est_t, template_tensor)
    #idx = torch.argmin(batch_loss)
    idx = torch.argmin(batch_loss)
    est_T = T[idx].detach().cpu().numpy()
    
    est_patial_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((est_T @ data['partial_2d'].T).T))
    
    est_patial_pcd.paint_uniform_color([1,0,0])
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    
    os.chmod(f'{args.data_path}/{args.scene_name}/viola', 0o777)
    np.save(f'{args.data_path}/{args.scene_name}/viola/est_T.npy',est_T)
            
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
    recon.transform(axis_aligned_T_w)
    recon.transform(est_pose_3d)
    template_points = template_2d
    template_points[:,-1] = 0
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_points)),recon])
    import pdb;pdb.set_trace()

