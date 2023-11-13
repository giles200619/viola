#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 01:07:37 2023

@author: jj
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def T24D(tensor):
    '''
    Convert 3x3 transformation to a 6D representation
    :param tensor: size[B, 3 ,3]
    :type tensor: torch.Tensor
    :return: size[B, 6] tensor
    '''
    # sixD = torch.zeros(1,9)
    B = tensor.shape[0]
    sixD = torch.rand((B, 6), requires_grad=False)
    sixD.to(tensor.device)
    sixD[:, 0:2] = tensor[:, 0:2, 0].reshape(B, 2)
    sixD[:, 2:4] = tensor[:, 0:2, 1].reshape(B, 2)
    sixD[:, 4:6] = tensor[:, 0:2, 2].reshape(B, 2)
    return sixD


def Out4D2T(x, check_det=True):
    '''
    Convert 6D representation to 3x3 transformation
    :param x: size[B, 6]
    :type x: torch.Tensor
    :param check_det: ensure the transformed matrix is a valid SE(2) transformation, defaults to True
    :return: size[B, 3 ,3] tensor
    '''
    B = x.shape[0]
    b1 = x[:, 0:2]
    a2 = x[:, 2:4]
    b1 = b1/torch.norm(b1, p=2, dim=-1).unsqueeze(-1).repeat(1, 2)
    c = torch.bmm(b1.unsqueeze(1), a2.unsqueeze(-1)).repeat(1, 2, 1).squeeze()  # (b1 @ a2)
    b2 = a2 - c*b1
    b2 = b2/torch.norm(b2, p=2, dim=-1).unsqueeze(-1).repeat(1, 2)

    t = x[:, 4:6]
    b1 = b1.view(B, 2, -1)
    b2 = b2.view(B, 2, -1)

    t = t.view(B, 2, -1)
    T = torch.cat((b1, b2, t), -1)
    if check_det:
        mask = torch.linalg.det(T[:, :2, :2]) < 0
        T[mask, :2, 1] *= -1
    return torch.cat((T, torch.Tensor([[0, 0, 1]]).unsqueeze(0).repeat(B, 1, 1).to(x.device)), 1)


def rot_err_2d(est_T, Tgt):
    '''
    measure rotation difference in angle and tranlsation difference between two transformation
    :param est_T: size[4, 4] np.array
    :param Tgt: size[4, 4] np.array
    '''
    rot1 = np.eye(3)
    rot2 = np.eye(3)
    rot1[:2, :2] = est_T[:2, :2]
    rot2[:2, :2] = Tgt[:2, :2]
    angle1 = R.from_matrix(rot1).as_euler('zyx', degrees=True)
    angle2 = R.from_matrix(rot2).as_euler('zyx', degrees=True)
    err_ = np.sum(np.abs(angle1 - angle2))
    err_ = err_ if err_ <= 180 else 360-err_
    return err_,  np.linalg.norm(est_T[:2, -1]-Tgt[:2, -1])

def chamfer_distance_with_batch(p1, p2, use_mean=False):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :return: one-directional Chamfer Distance of batches of two point sets and closest point indices
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.norm(p1 - p2, 2, dim=3)
    min_dist, inds = torch.min(dist, dim=2)

    if use_mean:
        min_dist = torch.mean(min_dist)

    return min_dist, inds

def chamfer_loss(partial, template):
    '''
    Calculate one directional Chamfer Distance from partial to template
    :param partial: size[B, N, D]
    :param template: size[B, M, D]
    :return: sum of mean Chamfer Distance 
    '''
    batch_loss ,_ = chamfer_distance_with_batch(partial, template)
    return batch_loss.sum(-1)

def check_converge(T_previous, T_now):
    '''
    check convergence for early stop 
    :param T_previous: size[B, 4, 4]
    :param T_now: size[B, 4, 4]
    :return: bool
    '''
    diff = T_previous @ torch.linalg.inv(T_now)
    max_distance = torch.max(torch.norm(diff[:,:2,-1]))
    max_angle_dif = torch.max(torch.abs(diff[:,0,0]-1))
    return max_distance<0.02 and max_angle_dif<0.005
   

def grid_initialization(template_2d,grid_n = 10):
    '''
    sample initial poses on a grid
    :param template_2d: size[N, 2] 2D LiDAR Map to be registered to
    :param grid_n: DESCRIPTION, defaults to 10
    :return: size[grid_n*grid_n*2, 3, 3] initial poses
    '''
    template_center = np.asarray([((np.max(template_2d[:,0])+np.min(template_2d[:,0]))/2), 
                                  ((np.max(template_2d[:,1])+np.min(template_2d[:,1]))/2),1])
    
    area_x = np.max(template_2d[:,0])-np.min(template_2d[:,0])
    area_y = np.max(template_2d[:,1])-np.min(template_2d[:,1])
    
    xv, yv = np.meshgrid(np.linspace(0, area_x, grid_n), np.linspace(0, area_y, grid_n))
    xv -= area_x/2
    yv -= area_y/2
    
    delta_xy = np.array([xv.reshape(-1), yv.reshape(-1)]).T 
    delta_xy += template_center[None,:2]
    
    n_rotation = 2
    init_poses = np.eye(3)[None,...].repeat(grid_n*grid_n*n_rotation,axis=0)
    init_poses[:,:2,2] = np.repeat(delta_xy, n_rotation,axis=0) # template_T_partial
    init_poses[1::n_rotation,:2,:2] = R.from_euler('z', 180, degrees=True).as_matrix()[:2,:2]
    #init_poses[2::n_rotation,:2,:2] = R.from_euler('z', 180, degrees=True).as_matrix()[:2,:2]
    #init_poses[3::n_rotation,:2,:2] = R.from_euler('z', 270, degrees=True).as_matrix()[:2,:2]
    return init_poses