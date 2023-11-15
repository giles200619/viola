#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import os
import numpy as np
from pathlib import Path
import glob
import torch
from PIL import Image
import open3d as o3d
from PIL import ImageColor
from tqdm import trange
from natsort import os_sorted
from pytorch3d.ops import sample_farthest_points


reduced_ = [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
            ]
reduced_colormap = np.asarray([np.asarray(ImageColor.getcolor(x, "RGB"))/255 for x in reduced_])

def get_thing_semantics(sc_classes='extended', path='./'):
    thing_semantics = [False]
    for cllist in [x.strip().split(',') for x in Path(f"{path}/scannet_{sc_classes}_things.csv").read_text().strip().splitlines()]:
        thing_semantics.append(bool(int(cllist[1])))
    return thing_semantics

def convert_from_mask_to_semantics_and_instances_no_remap(original_mask, segments, _coco_to_scannet, is_thing, instance_ctr, instance_to_semantic):
    id_to_class = torch.zeros(1024).int()
    instance_mask = torch.zeros_like(original_mask)
    invalid_mask = original_mask == 0
    for s in segments:
        id_to_class[s['id']] = s['category_id']
        if is_thing[s['category_id']]:
            instance_mask[original_mask == s['id']] = instance_ctr
            instance_to_semantic[instance_ctr] = s['category_id']
            instance_ctr += 1
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(original_mask.shape), instance_mask, invalid_mask, instance_ctr, instance_to_semantic


def postprocess_m2f_aug(m2f_result_path, path_to_csv):
    src_folder = Path(m2f_result_path)#Path('/home/jj/work/room_recon/data/redwood_apartment/mask2former/')
    rgb_folder_name = 'key_frames'
    sc_classes ='extended'

    coco_to_scannet = {}
    thing_semantics = get_thing_semantics(sc_classes, path_to_csv)
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"{path_to_csv}/scannet_{sc_classes}_to_coco.csv").read_text().strip().splitlines()]):
        for c in cllist[1:]:
            coco_to_scannet[c.split('/')[1]] = cidx + 1
    instance_ctr = 1
    instance_to_semantic = {}
    instance_ctr_notta = 1
    segment_ctr = 1
    instance_to_semantic_notta = {}
    (src_folder / "m2f_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_semantics").mkdir(exist_ok=True)
    (src_folder / "m2f_feats").mkdir(exist_ok=True)
    (src_folder / "m2f_probabilities").mkdir(exist_ok=True)
    (src_folder / "m2f_invalid").mkdir(exist_ok=True)
    (src_folder / "m2f_segments").mkdir(exist_ok=True)
    
    if not len(os.listdir(str((src_folder / "m2f_segments")))) == len(os.listdir(str((src_folder.parent.absolute() / rgb_folder_name)))):
        for idx, fpath in enumerate(sorted(list((src_folder.parent.absolute() / rgb_folder_name).iterdir()), key=lambda x: x.stem)):
            print(idx,fpath)
            data = torch.load(gzip.open(src_folder / f'{fpath.stem}.ptz'), map_location='cpu')
            probability, confidence, confidence_notta = data['probabilities'], data['confidences'], data['confidences_notta']
        
            semantic, instance, invalid_mask, instance_ctr, instance_to_semantic = convert_from_mask_to_semantics_and_instances_no_remap(data['mask'], data['segments'], coco_to_scannet, thing_semantics, instance_ctr, instance_to_semantic)
            semantic_notta, instance_notta, _, instance_ctr_notta, instance_to_semantic_notta = convert_from_mask_to_semantics_and_instances_no_remap(data['mask_notta'], data['segments_notta'], coco_to_scannet, thing_semantics,
                                                                                                                                                      instance_ctr_notta, instance_to_semantic_notta)
            segment_mask = torch.zeros_like(data['mask'])
            for s in data['segments']:
                segment_mask[data['mask'] == s['id']] = segment_ctr
                segment_ctr += 1
            Image.fromarray(segment_mask.numpy().astype(np.uint16)).save(src_folder / "m2f_segments" / f"{fpath.stem}.png")
            Image.fromarray(semantic.numpy().astype(np.uint16)).save(src_folder / "m2f_semantics" / f"{fpath.stem}.png")
            Image.fromarray(instance.numpy()).save(src_folder / "m2f_instance" / f"{fpath.stem}.png")
            Image.fromarray(semantic_notta.numpy().astype(np.uint16)).save(src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png")
            Image.fromarray(instance_notta.numpy()).save(src_folder / "m2f_notta_instance" / f"{fpath.stem}.png")
            Image.fromarray(invalid_mask.numpy().astype(np.uint8) * 255).save(src_folder / "m2f_invalid" / f"{fpath.stem}.png")
            np.savez_compressed(src_folder / "m2f_probabilities" / f"{fpath.stem}.npz", probability=probability.float().numpy(), confidence=confidence.float().numpy(), confidence_notta=confidence_notta.float().numpy())
            
def postprocess_m2f_noaug(m2f_result_path, path_to_csv):
    src_folder = Path(m2f_result_path)#Path('/home/jj/work/room_recon/data/redwood_apartment/mask2former/')
    rgb_folder_name = 'key_frames'
    sc_classes ='extended'

    coco_to_scannet = {}
    thing_semantics = get_thing_semantics(sc_classes, path_to_csv)
    for cidx, cllist in enumerate([x.strip().split(',') for x in Path(f"{path_to_csv}/scannet_{sc_classes}_to_coco.csv").read_text().strip().splitlines()]):
        for c in cllist[1:]:
            coco_to_scannet[c.split('/')[1]] = cidx + 1
    instance_ctr_notta = 1
    instance_to_semantic_notta = {}
    (src_folder / "m2f_notta_instance").mkdir(exist_ok=True)
    (src_folder / "m2f_notta_semantics").mkdir(exist_ok=True)
    
    if not len(os.listdir(str((src_folder / "m2f_notta_semantics")))) == len(os.listdir(str((src_folder.parent.absolute() / rgb_folder_name)))):
        for idx, fpath in enumerate(sorted(list((src_folder.parent.absolute() / rgb_folder_name).iterdir()), key=lambda x: x.stem)):
            print(idx,fpath)
            if os.path.exists(src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png"): continue
            data = torch.load(gzip.open(src_folder / f'{fpath.stem}.ptz'), map_location='cpu')
            semantic_notta, instance_notta, _, instance_ctr_notta, instance_to_semantic_notta = convert_from_mask_to_semantics_and_instances_no_remap(data['mask_notta'], data['segments_notta'], coco_to_scannet, thing_semantics,
                                                                                                                                                      instance_ctr_notta, instance_to_semantic_notta)
            Image.fromarray(semantic_notta.numpy().astype(np.uint16)).save(src_folder / "m2f_notta_semantics" / f"{fpath.stem}.png")
            Image.fromarray(instance_notta.numpy()).save(src_folder / "m2f_notta_instance" / f"{fpath.stem}.png")

def T_pcd(T, pcd):
    '''
    simple function for transforming 3D points, T @ pcd
    :param T: 4x4 transformation
    :param pcd: Nx3 point cloud array
    :return: Nx3
    '''
    return (T @ np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1).T).T[:, :3]

def fuse_m2f_aug(pcd, Ts, K, m2f_prob_ls, H, W,
                        number_of_points=100000, chunck_size=10000, vis=False):    
    assert Ts.shape[0]==len(m2f_prob_ls), "Poses should correspond to the segmented images"
    pts = np.asarray(pcd.points)
    print(f'downsampling reconstructed points from {pts.shape[0]} to {number_of_points}...')
    _, pts_idx = sample_farthest_points(torch.Tensor(pts).unsqueeze(0).cuda(), K=number_of_points)
    pts_idx = pts_idx[0].cpu().numpy()
    pts = pts[pts_idx]
    original_colors = np.asarray(pcd.colors)[pts_idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    
    seg_labels = np.zeros((0,))
    colors = np.zeros((0,3))
                
    chunks = int(number_of_points/chunck_size) if number_of_points%chunck_size==0 else int(number_of_points/chunck_size)+1
    n_frames = Ts.shape[0]
    print('start fusing...')
    for chunk in trange(chunks):
        
        full_pts = pts[chunck_size*chunk:chunck_size*(chunk+1),:] if chunk!=chunks-1 else pts[chunck_size*chunk:,:]
        full_conf = np.zeros((full_pts.shape[0],n_frames))
        full_prob = np.zeros((full_pts.shape[0],n_frames,32))
        
        for i in range(Ts.shape[0]):
            data = np.load(m2f_prob_ls[i])
            probability = data['probability']
            confidence = data['confidence']
            
            pcd_current_frame = T_pcd(np.linalg.inv(Ts[i]), full_pts)
            pcd_canvas = ((K) @ pcd_current_frame.T).T
            pcd_canvas = np.round((pcd_canvas/pcd_canvas[:,-1][...,None])[:,:2]).astype(np.int16)
            
            # mask visible points
            mask = pcd_current_frame[:,-1] > 0
            mask = np.logical_and(mask, pcd_canvas[:,0] >= 0)
            mask = np.logical_and(mask , pcd_canvas[:,0] < W-0.5) #u
            mask = np.logical_and(mask, pcd_canvas[:,1] >= 0) #v
            mask = np.logical_and(mask, pcd_canvas[:,1] < H-0.5) #v
            diameter = 1
            camera = [0, 0, 0]
            radius = diameter * 5000
            pcd_cur_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_current_frame[mask]))
            try:
                _, pt_map = pcd_cur_.hidden_point_removal(camera, radius)
            except:
                continue
            mask_ = np.zeros(pcd_current_frame[mask].shape[0])
            mask_[pt_map] = 1        
            # mask occluded points
            mask[mask] = mask_==1
            
            uv_coord = pcd_canvas[mask]
            full_prob[mask,i,:] = probability[uv_coord[:,1], uv_coord[:,0]]
            full_conf[mask,i] = confidence[uv_coord[:,1], uv_coord[:,0]]
            
        full_conf_ = np.sum(full_conf, axis=-1)[...,None]
        full_conf_[full_conf_==0] = 1
        norm_conf = full_conf/full_conf_ # 
        weighted_prob = np.sum(full_prob * norm_conf[...,None], axis=1)
        
        seg = np.argmax(weighted_prob,axis=-1)
        color = reduced_colormap[seg.astype(int),:]
        
        seg_labels = np.concatenate((seg_labels, seg), axis=0)
        colors = np.vstack((colors, color))
        
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if vis:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd, cf])
    return pcd, seg_labels, original_colors

def fuse_m2f_noaug(pcd, Ts, K, m2f_mask_ls, H, W,
                        number_of_points=100000, chunck_size=10000, vis=False):    
    assert Ts.shape[0]==len(m2f_mask_ls), "Poses should correspond to the segmented images"
    pts = np.asarray(pcd.points)
    print(f'downsampling reconstructed points from {pts.shape[0]} to {number_of_points}...')
    _, pts_idx = sample_farthest_points(torch.Tensor(pts).unsqueeze(0).cuda(), K=number_of_points)
    pts_idx = pts_idx[0].cpu().numpy()
    pts = pts[pts_idx]
    original_colors = np.asarray(pcd.colors)[pts_idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    
    seg_labels = np.zeros((0,))
    colors = np.zeros((0,3))
                
    chunks = int(number_of_points/chunck_size) if number_of_points%chunck_size==0 else int(number_of_points/chunck_size)+1
    n_frames = Ts.shape[0]
    print('start fusing...')
    for chunk in trange(chunks):

        full_pts = pts[chunck_size*chunk:chunck_size*(chunk+1),:] if chunk!=chunks-1 else pts[chunck_size*chunk:,:]
        semantic_votes = np.zeros((full_pts.shape[0],50)) # max number of class
    
        for i in range(Ts.shape[0]):
        
            semantic_mask = np.asarray(Image.open(m2f_mask_ls[i]))
            
            pcd_current_frame = T_pcd(np.linalg.inv(Ts[i]), full_pts)
            pcd_canvas = ((K) @ pcd_current_frame.T).T
            pcd_canvas = np.round((pcd_canvas/pcd_canvas[:,-1][...,None])[:,:2]).astype(np.int16)
            
            # mask visible points
            mask = pcd_current_frame[:,-1] > 0
            mask = np.logical_and(mask, pcd_canvas[:,0] >= 0)
            mask = np.logical_and(mask , pcd_canvas[:,0] < W-0.5) #u
            mask = np.logical_and(mask, pcd_canvas[:,1] >= 0) #v
            mask = np.logical_and(mask, pcd_canvas[:,1] < H-0.5) #v
            diameter = 1
            camera = [0, 0, 0]
            radius = diameter * 5000
            pcd_cur_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_current_frame[mask]))
            _, pt_map = pcd_cur_.hidden_point_removal(camera, radius)
            mask_ = np.zeros(pcd_current_frame[mask].shape[0])
            mask_[pt_map] = 1        
            # mask occluded points
            mask[mask] = mask_==1
            
            uv_coord = pcd_canvas[mask]
            labels = semantic_mask[uv_coord[:,1], uv_coord[:,0]]
            semantic_votes[mask, labels] += 1
        
        seg = np.argmax(semantic_votes,axis=-1)
        color = reduced_colormap[seg.astype(int),:]
        
        seg_labels = np.concatenate((seg_labels, seg), axis=0)
        colors = np.vstack((colors, color))
        
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if vis:
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd, cf])
    return pcd, seg_labels, original_colors

def est_gravity(pts, seg_all, plane_threshold=0.02, vis=False, min_wall_pts = 100, ransac_iter=5000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))  # in camera frame
    floor_pts = pts[seg_all==2] # 2 floor, 1 wall
    if floor_pts.shape[0]>100:
        floor_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(floor_pts))
        plane_model, inliers = floor_pcd.segment_plane(distance_threshold=plane_threshold,
                                             ransac_n=3,
                                             num_iterations=ransac_iter)
        [a, b, c, d] = plane_model
        
        if vis: 
            inlier_cloud = floor_pcd.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            cf = o3d.geometry.TriangleMesh.create_coordinate_frame()
            o3d.visualization.draw_geometries([pcd, inlier_cloud, cf])
        
        z = np.asarray([a,b,c]) if d>0 else -np.asarray([a,b,c]) # floor normal
        x = np.asarray([1,0,0]) - a*z
        x /= np.linalg.norm(x)
        y = np.cross(z,x)
        y /= np.linalg.norm(y)
        R_ = np.asarray([x,y,z])
        w_T_c = np.eye(4)
        w_T_c[:3, :3] = R_
        w_T_c[:3,3] = [0,0,d] if d>0 else [0,0,-d] #axis_aligned_T_droid_frame
        return w_T_c
    else: 
        raise ValueError('Cannot use floor to estimate the gravity direction')