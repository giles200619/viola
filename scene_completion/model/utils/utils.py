import torch
import cv2
import os
import json
import numpy as np
import time
import pymeshlab
import imageio
import open3d as o3d

from PIL import Image
import matplotlib.pyplot as plt

from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline


def load_sd_inpaint(args, models_path):
    model_path = os.path.join(models_path, "stable-diffusion-2-inpainting")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Next Image"
    })

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe


def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img


def generate_first_image(args):
    model_path = os.path.join(args.models_path, "stable-diffusion-2-1")
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Start Image"
    })

    return pipe(args.prompt).images[0]


def save_settings(args):
    with open(os.path.join(args.out_path, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    dirs = np.stack(
        ((i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], np.ones_like(i)), axis=-1
    )
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return np.concatenate((rays_o, rays_d), axis=-1)


def get_pcd(K, img, depth, mask=None, c2w=np.eye(4)):
    H, W = depth.shape
    # Create the point cloud
    depth = depth.reshape(-1)
    rays = get_rays_np(H, W, K, c2w)

    u = np.arange(0, W)
    v = np.arange(0, H)
    u, v = np.meshgrid(u, v)
    u, v = u.reshape(-1), v.reshape(-1)

    uv = np.stack((u, v, np.ones_like(u)), axis=-1)
    xy = (np.linalg.inv(K) @ np.transpose(uv)) * depth[None, :]
    x = np.transpose(xy)[:, 0]
    y = np.transpose(xy)[:, 1]
    depth = np.sqrt(x ** 2 + y ** 2 + depth ** 2)

    rays = rays.reshape(-1, 6)
    rays0, raysd = rays[:, :3], rays[:, 3:]
    raysd = raysd / np.linalg.norm(raysd, axis=-1, keepdims=True)
    pts = rays0 + raysd * (depth[:, None])
    colors = img.reshape(-1, img.shape[-1])

    if mask is not None:
        indices = np.where(mask.reshape(-1))
        pts = pts[indices]
        colors = colors[indices]

    return pts, colors


def get_o3d_pointcloud(points: np.ndarray, colors: np.ndarray, scale: float = 1e-3, uint_colors: bool = True):

    points = points * scale
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    if uint_colors:
        colors = colors / 255.
    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pcd


def get_colormap_pil(image: np.ndarray, cmap='inferno_r'):

    cm = plt.get_cmap(cmap)
    colored_image_np = cm(image / image.max()) * (image > 0)[..., None]

    return Image.fromarray((colored_image_np[:, :, :3] * 255).astype(np.uint8))
