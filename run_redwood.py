#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 00:41:49 2023

@author: jj
"""

import numpy as np
import open3d as o3d
import os
import scipy
import torch
from utils import geometry_utils, optimization_utils, cross_correlation_utils
from preprocess.postprocess_m2f import reduced_colormap
import torch.optim as optim
import time
import argparse
import pytorch3d
from pytorch3d.ops import sample_farthest_points

def pose_optimization(template_2d, partial_2d, init_poses, iterations=180):
    template_tensor = torch.Tensor(template_2d).unsqueeze(0).repeat(init_poses.shape[0],1,1).cuda()
    partial_tensor = torch.Tensor(partial_2d).cuda() 
    
    torch.cuda.empty_cache()
    est_4D = optimization_utils.T24D(torch.tensor(init_poses)).float().cuda()
    est_4D = est_4D.detach().requires_grad_().cuda()
    optimizer = optim.Adam([est_4D],lr=0.1)  
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.5)
    print('optimization...')
    for it in range(iterations):
        optimizer.zero_grad()
        
        T = optimization_utils.Out4D2T(est_4D)
        est_t = (T @ partial_tensor.T).permute(0,2,1)
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
    est_t = (T @ partial_tensor.T).permute(0,2,1)
    batch_loss = optimization_utils.chamfer_loss(est_t, template_tensor)
    best_idx = torch.argmin(batch_loss)
    
    result = {}
    result['est_T'] = T[best_idx].detach().cpu().numpy()
    result['min_loss'] = batch_loss[best_idx].item()
    result['final_losses'] = batch_loss.detach().cpu().numpy()
    result['final_poses'] = T.detach().cpu().numpy()
    return result
    

def viola_matching(data, template_2d, init_method='grid', vis=False):
    iterations = 180
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
    data['template_2d'] = template_2d
    # 
    if vis:
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_2d)), cf])
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_2d)), 
                                           o3d.geometry.PointCloud(o3d.utility.Vector3dVector(partial_2d)), cf])

    print('pose initialization:', init_method)
    img_cor_t0 = time.time()
    if init_method=='grid':
        init_poses = optimization_utils.grid_initialization(template_2d,grid_n = 10)
    elif init_method=='img_cross_correlation':
        init_poses = cross_correlation_utils.get_cross_correlation_matches(partial_2d, template_2d, False, False)
    else:
        raise NotImplementedError
    print('pose init time', time.time()-img_cor_t0)
    if vis:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
        sp_loc = geometry_utils.keypoints_to_spheres((init_poses @ np.array([0,0,1])[None,:,None])[...,0], radius=0.1)
        o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['template_2d'])),sp_loc,cf])
    #
    
    result = pose_optimization(template_2d, partial_2d, init_poses, iterations)
    result['template_2d'] = template_2d
    result['partial_2d'] = partial_2d
    return result

def scene_completion_decision(result, rot_tr=20, t_tr=0.3, l_tr=20):
    sorted_pose_idx = np.argsort(result['final_losses'])
    best_pose_loss = result['final_losses'][sorted_pose_idx[0]]
    best_pose = result['final_poses'][sorted_pose_idx[0]]
    for i in range(sorted_pose_idx.shape[0]):
        rot_diff, t_diff = optimization_utils.rot_err_2d(result['final_poses'][sorted_pose_idx[i]], best_pose)
        if rot_diff>rot_tr or t_diff>rot_tr:
            second_pose_loss = result['final_losses'][sorted_pose_idx[i]]
            use_inpaint = True if abs(second_pose_loss-best_pose_loss) < l_tr else  False
            break
    return use_inpaint
    

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--data_path",
        default='/home/jj/work/data/viola_sample/redwood/loft_short',
        help="directory to short video sequences, should contain /image /depth"
    )        
    parser.add_argument(
        "--lidar_path",
        default='/home/jj/work/data/viola_sample/redwood/loft_lidar_dense.mat',
        help="path to 2D LiDAR scan"
    )     
    parser.add_argument(
        "--pose_init_method",
        default='img_cross_correlation',
        choices=['grid','img_cross_correlation'],
        help="method for pose initialization between 2D point clouds"
    )
    parser.add_argument(
        "--scene_completion_activation",
        default='off',
        choices=['on','off','criterion'],
        help="to activate scene completion module or not, criterion is based on the decision method proposed in Section IV-A"
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()    
    print('start viola...')
    # load lidar data
    floor_plan_lidar_data = scipy.io.loadmat(args.lidar_path)
    w_T_redwood = floor_plan_lidar_data['w_T_c']
    template = floor_plan_lidar_data['lidar_points']
    _, pts_idx = sample_farthest_points(torch.Tensor(template).unsqueeze(0), K=1800)
    pts_idx = pts_idx[0].numpy()
    template_2d = template[pts_idx]
    template_2d[:,-1] = 1
    
    ## load open3d result and run viola matching, viola_input.npy is generated with preprocess/redwood_open3d_m2f.py
    data = np.load(f'{args.data_path}/viola_input.npy',allow_pickle=True).item()
    result = viola_matching(data, template_2d, init_method=args.pose_init_method, vis=False)
    est_T = result['est_T']
    
    # visualization 2d
    est_patial_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((est_T @ data['partial_2d'].T).T))
    est_patial_pcd.paint_uniform_color([1,0,0])
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([est_patial_pcd,o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_2d))])
    # visualization 3d
    est_pose_3d = np.eye(4)
    est_pose_3d[:2,:2] = est_T[:2,:2]
    est_pose_3d[:2,-1] = est_T[:2,-1]
    
    o3d_recon = o3d.io.read_point_cloud(f'{args.data_path}/scene/integrated.ply')
    semantic_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data['pts']))
    semantic_pcd.colors = o3d.utility.Vector3dVector(reduced_colormap[data['m2f_seg'].astype(int),:])
    
    o3d_recon.transform(data['axis_aligned_T_w'])
    o3d_recon.transform(est_pose_3d)
    semantic_pcd.transform(data['axis_aligned_T_w'])
    semantic_pcd.transform(est_pose_3d)
    template_points = template_2d
    template_points[:,-1] = 0
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_points)),o3d_recon])
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(template_points)),semantic_pcd])
    
    import pdb;pdb.set_trace()
    
    ## scene completion via inpainting
    if args.scene_completion_activation=='on' or (args.scene_completion_activation=='criterion' and scene_completion_decision(result)):
        raise NotImplementedError #TODO: inpainting module
        