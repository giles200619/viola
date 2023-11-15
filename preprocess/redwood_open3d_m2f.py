#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import numpy as np
import shutil
import glob
from natsort import os_sorted
import open3d as o3d
import argparse
import subprocess
import json
from postprocess_m2f import postprocess_m2f_aug, postprocess_m2f_noaug, fuse_m2f_aug, fuse_m2f_noaug, est_gravity

redwood_W, redwood_H = (640,480)
redwood_intrinsic = np.array([[525,0,319.5],
                              [0,525,239.5],
                              [0,0,1]])

def run_open3d(data_path, path_to_open3d):
    os.makedirs(data_path, exist_ok=True)
    json_template = f'{path_to_open3d}/examples/python/reconstruction_system/config/tutorial.json'
    f = open(json_template)
    data = json.load(f)
    data['name'] = f'redwood {data_path}'
    data['path_dataset'] = data_path
    data['path_intrinsic'] = os.path.join(data_path,'intrinsic.json')
    json_object = json.dumps(data, indent=4)
    with open(os.path.join(data_path,'open3d_config.json'), "w") as outfile:
        outfile.write(json_object)
    
    W, H = (redwood_W, redwood_H)
    intrinsic = redwood_intrinsic
    o3d_K = o3d.camera.PinholeCameraIntrinsic(W,H,intrinsic[:3,:3])
    o3d.io.write_pinhole_camera_intrinsic(os.path.join(data_path,'intrinsic.json'),o3d_K)
    
    cwd = os.getcwd()
    os.chdir(f'{path_to_open3d}/examples/python/reconstruction_system')
    subprocess.run(["python","run_system.py","--config", os.path.join(data_path,'open3d_config.json'),
                    "--make", "--register", "--refine", "--integrate"
                    ])
    os.chdir(cwd)
    
def run_m2f(data_path, m2f_path, no_aug=False):    
    cwd = os.getcwd()
    os.chdir(f'{m2f_path}/demo')
    if no_aug:
        subprocess.run(["python",f"{m2f_path}/demo/demo_noaug.py","--config-file",
                        f"{m2f_path}/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
                        "--input", f'{data_path}/key_frames',
                        "--opts", "MODEL.WEIGHTS", f"{m2f_path}/model_weights/model_final_47429163_0.pkl"
                        ])
        os.chdir(cwd)
        postprocess_m2f_noaug(m2f_result_path=f'{data_path}/mask2former', path_to_csv='./')
    else:
        subprocess.run(["python",f"{m2f_path}/demo/demo.py","--config-file",
                        f"{m2f_path}/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
                        "--input", f'{data_path}/key_frames',
                        "--opts", "MODEL.WEIGHTS", f"{m2f_path}/model_weights/model_final_47429163_0.pkl"
                        ])
        os.chdir(cwd)
        postprocess_m2f_aug(m2f_result_path=f'{data_path}/mask2former', path_to_csv='./')
    
def read_trajectory(filename):
    traj = []
    with open(filename, 'r') as f:
        metastr = f.readline();
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape = (4, 4))
            for i in range(4):
                matstr = f.readline();
                mat[i, :] = np.fromstring(matstr, dtype = float, sep=' \t')
            #traj.append(CameraPose(metadata, mat))
            data = {}
            data['metadata'] = ' '.join(map(str, metadata))
            data['pose'] = mat # w_T_cam
            traj.append(data)
            metastr = f.readline()
    return traj

def get_parser():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument(
        "--data_path",
        default='/home/jj/work/data/viola_sample/redwood/loft_short',
        help="directory to short video sequences, should contain /image /depth"
    )       
    parser.add_argument(
        "--open3d_path",
        default='/home/jj/work/Open3D/',
        help="path to open3d"
    )
    parser.add_argument(
        "--m2f_path",
        default='/home/jj/work/Mask2Former/',
        help="path to Mask2former"
    )     
    parser.add_argument(
        "--skip_every_n_frames",
        type=int,
        default=15,
        help="skip every n frames to run semantic segmentation, increase to speed up but might reduce accuracy",
    )
    parser.add_argument(
        '--no_aug', 
        action='store_true', 
        help='disable image augmentation before semantic segmentation to speed up'
    )
    parser.add_argument(
        "--N_pts_semantic_pcd",
        type=int,
        default=200000,
        help="downsample reconstructed points to N for fusing semantic segmentation",
    )
    parser.add_argument(
        "--fusing_chunk_size",
        type=int,
        default=40000,
        help="decrease if cpu ram oom during 3D semantic fusion",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()    
    
    # step 1: run open3d reconstruction
    run_open3d(args.data_path, args.open3d_path)
    
    # step 2: extract key frames and run mask2former 
      #extract key frames
    os.makedirs(os.path.join(args.data_path,'key_frames'), exist_ok=True)
    rgb_ls = os_sorted(glob.glob(os.path.join(args.data_path,'image','*')))[::args.skip_every_n_frames]
    for rgb_file in rgb_ls:
        shutil.copy(rgb_file, os.path.join(args.data_path,'key_frames'))
      #run mask2former 
    run_m2f(args.data_path, args.m2f_path, args.no_aug)
    
    # step 3: fuse 2D segmentation via voting
    print('fusing 2D segmentation to 3D...')
    pcd = o3d.io.read_point_cloud(f'{args.data_path}/scene/integrated.ply')
     #extract corresponding poses of the key frames
    o3d_traj = read_trajectory(f'{args.data_path}/scene/trajectory.log')[::args.skip_every_n_frames]
    w_T_cams = []
    for i in range(len(o3d_traj)): w_T_cams.append(o3d_traj[i]['pose'])
    w_T_cams = np.asarray(w_T_cams)
    
    if args.no_aug:
        m2f_mask_ls = os_sorted(glob.glob(os.path.join(args.data_path,'mask2former','m2f_notta_semantics','*')))
        seg_pcd, seg_labels, original_colors = fuse_m2f_noaug(pcd, w_T_cams, redwood_intrinsic, m2f_mask_ls, redwood_H, redwood_W,
                                number_of_points=args.N_pts_semantic_pcd, chunck_size=args.fusing_chunk_size, vis=False)
    else:
        m2f_prob_ls = os_sorted(glob.glob(os.path.join(args.data_path,'mask2former','m2f_probabilities','*')))
        seg_pcd, seg_labels, original_colors = fuse_m2f_aug(pcd, w_T_cams, redwood_intrinsic, m2f_prob_ls, redwood_H, redwood_W,
                                number_of_points=args.N_pts_semantic_pcd, chunck_size=args.fusing_chunk_size, vis=False)
    #
    viola_input = {}
    viola_input['pts'] = np.asarray(seg_pcd.points)
    viola_input['colors'] = original_colors
    viola_input['m2f_seg'] = seg_labels
    viola_input['w_T_cam0'] = w_T_cams[0]
    viola_input['w_T_cams'] = w_T_cams
    viola_input['axis_aligned_T_w'] = est_gravity(np.asarray(seg_pcd.points), seg_labels, 
                                                  plane_threshold=0.02, vis=False, min_wall_pts = 100, ransac_iter=5000)
    np.save(os.path.join(args.data_path, 'viola_input.npy'), viola_input)
    
    # visualize
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    pcd.transform(viola_input['axis_aligned_T_w'])
    o3d.visualization.draw_geometries([pcd, cf])
    
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
    seg_pcd.transform(viola_input['axis_aligned_T_w'])
    o3d.visualization.draw_geometries([seg_pcd, cf])