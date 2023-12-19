import os
from PIL import Image
import torch
import numpy as np
import open3d as o3d
from datetime import datetime
import pyrender
import trimesh
from sklearn.neighbors import NearestNeighbors

from .model.iron_depth.predict_depth import predict_iron_depth, load_iron_depth_model
from .model.pointersect.pointersect import PointersectInference
from .model.utils.utils import load_sd_inpaint, get_pcd, get_o3d_pointcloud, get_colormap_pil
from .scene_completion_config import SceneCompletionConfig
import sys
sys.path.append('..')
from mask2former.demo.demo_function import run_m2f_segmentation


class DepthOutpainter:
    def __init__(self, K, im_size, init_pcd, camera_T_floor) -> None:
        """initializes the depth outpainter

        :param args: cli arguments
        :param K: camera intrinsics
        :param im_size: input image dimensions
        :param init_pcd: input point cloud
        :param camera_T_floor: floor frame in the first camera frame
        """

        self.args = SceneCompletionConfig().get_config()

        self.models_path = 'checkpoints'
        self.iron_depth_type = 'scannet'
        self.iron_depth_iters = 20
        self.im_size = im_size
        self.W, self.H = im_size
        self.vis_images = True

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.camera_T_floor = camera_T_floor
        self.setup_models()
        self.counter = 0

        self.scene_pcd = o3d.geometry.PointCloud()
        self.scene_pcd += init_pcd
        self.scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.K = K
        self.K_torch = torch.from_numpy(self.K).float()
        self.inpainting_size = (512, 512, 3)

        self.ground_floor_depth = False
        floor_size = 5.
        floor_grid_dim = 64
        self.floor_points = np.argwhere(np.zeros((floor_grid_dim, floor_grid_dim)) == 0)
        self.floor_points = self.floor_points - floor_grid_dim / 2.
        self.floor_points /= ((floor_grid_dim / 2.) / floor_size)
        self.floor_points = np.hstack((self.floor_points, np.zeros_like(self.floor_points)[:, :1]))
        self.floor_id = 2  # semantic id of the floor class defined in mask2former

    def setup_models(self):
        # construct inpainting stable diffusion pipeline
        self.inpaint_pipe = load_sd_inpaint(self.args, self.models_path)

        # construct depth model
        self.iron_depth_n_net, self.iron_depth_model = load_iron_depth_model(
            self.iron_depth_type, self.iron_depth_iters, self.models_path, self.args.device)

        self.pointersect_model = PointersectInference(im_size=self.im_size)

    def render_pointcloud_pointersect(self, pointcloud_c_o3d: o3d.geometry.PointCloud, K: torch.tensor, H_c2w: torch.tensor):
        """renders point cloud pointcloud_c_o3d at poses H_c2w

        :param pointcloud_c_o3d: scene point cloud
        :param K: (3, 3)
        :param H_c2w: (B, N, 4, 4)
        :return: rendered rgb and depth images
        """
        rendered_rgb_np, rendered_depth_np = self.pointersect_model.render_pointcloud(
            o3d_pcd=pointcloud_c_o3d,
            intrinsics=K,
            H_c2w=H_c2w
        )

        return rendered_rgb_np, rendered_depth_np

    def render_pointcloud(self, pointcloud_c_o3d: o3d.geometry.PointCloud, K: np.ndarray, H_c2w: np.ndarray):
        # reconstruct color mesh
        recon_pts = np.asarray(pointcloud_c_o3d.points)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pointcloud_c_o3d, depth=9)
        vertices = np.asarray(mesh.vertices)
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(recon_pts)
        min_v_to_p = y_nn.kneighbors(vertices)[0]

        mesh.remove_vertices_by_mask(min_v_to_p > 0.05)
        # render image
        tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                   vertex_normals=np.asarray(mesh.vertex_normals),
                                   vertex_colors=np.asarray(mesh.vertex_colors))
        py_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
        # scene = pyrender.Scene()
        scene = pyrender.Scene(ambient_light=[1, 1, 1], bg_color=[0, 0, 0])
        camera = pyrender.IntrinsicsCamera(fx=K[0][0], fy=K[1][1], cx=K[0][2], cy=K[1][2],
                                           znear=0.05, zfar=100.0)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=1e3)

        scene.add(py_mesh)
        opengl_c2w = H_c2w @ np.array([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])
        scene.add(light, pose=opengl_c2w)

        scene.add(camera, pose=opengl_c2w)
        # pyrender.Viewer(scene, use_raymond_lighting=True)
        # render scene
        im_size = self.im_size
        r = pyrender.OffscreenRenderer(viewport_width=im_size[0],
                                       viewport_height=im_size[1])
        rgb, depth = r.render(scene)
        r.delete()
        return rgb, depth

    def inpaint(self, rendered_image_pil, inpaint_mask_pil):

        # m = np.asarray(inpaint_mask_pil).astype(np.uint8)
        # rendered_image_numpy = np.asarray(rendered_image_pil)
        # rendered_image_pil = Image.fromarray(cv2.inpaint(rendered_image_numpy, m, 3, cv2.INPAINT_TELEA))

        inpainted_image_pil = self.inpaint_pipe(
            prompt=self.args.prompt,
            negative_prompt=self.args.negative_prompt,
            num_images_per_prompt=1,
            image=rendered_image_pil,
            mask_image=inpaint_mask_pil,
            guidance_scale=self.args.guidance_scale,
            num_inference_steps=self.args.num_inference_steps
        ).images[0]

        return inpainted_image_pil

    def predict_depth(self, image_pil, input_depth_torch=None, input_mask_torch=None, fix_input_depth=True):
        # use the IronDepth method to predict depth: https://github.com/baegwangbin/IronDepth
        # taken from: https://github.com/lukasHoel/text2room

        predicted_depth, _ = predict_iron_depth(
            image=image_pil,
            K=self.K,
            device=self.args.device,
            model=self.iron_depth_model,
            n_net=self.iron_depth_n_net,
            input_depth=input_depth_torch,
            input_mask=input_mask_torch,
            fix_input_depth=fix_input_depth
        )

        return predicted_depth

    def view_completion(self, pose_c):
        """completes and fuses the scene point cloud from the input viewpoint

        :param pose_c: target viewpoint pose w.r.t. the first camera frame (4, 4)
        """

        # Point cloud rendering
        # pose_c = pose_c[None, None]
        # if not self.scene_pcd.has_normals():
        #     self.scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        rendered_rgb_np, rendered_depth_np = self.render_pointcloud(self.scene_pcd, self.K, pose_c)
        canvas = np.zeros(self.inpainting_size, dtype=rendered_rgb_np.dtype)
        canvas[:rendered_rgb_np.shape[0], :rendered_rgb_np.shape[1]] = rendered_rgb_np
        # rendered_rgb_pil = Image.fromarray((canvas * 255).astype(np.uint8))
        rendered_rgb_pil = Image.fromarray(canvas)
        if self.vis_images:
            rendered_rgb_pil.save(f'rendered_rgb_{self.counter}.png')
        mask_np = (np.asarray(rendered_rgb_pil) == 0).all(-1).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_np)

        # Inpainting
        inpainted_rgb_pil = self.inpaint(rendered_rgb_pil, mask_pil)
        if self.vis_images:
            inpainted_rgb_pil.save(f'inpainted_rgb_{self.counter}.png')

        inpainted_rgb_np = np.array(inpainted_rgb_pil)
        inpainted_rgb_pil = inpainted_rgb_pil.crop((0, 0, self.W, self.H))

        # Floor grounding
        if self.ground_floor_depth:
            rendered_depth_np = self.floor_grounding(inpainted_rgb_pil, rendered_depth_np, pose_c)

        # Depth estimation
        rendered_depth_torch = torch.from_numpy(rendered_depth_np).float().to(self.device)
        rendered_mask_torch = rendered_depth_torch == 0
        depth_estimated_torch = self.predict_depth(inpainted_rgb_pil, rendered_depth_torch, rendered_mask_torch)
        depth_estimated_np = depth_estimated_torch.cpu().detach().numpy()
        if self.vis_images:
            get_colormap_pil(depth_estimated_np).save(f'depth_{self.counter}.png')

        # 3D projection
        mask_np_cropped = np.asarray(mask_pil.crop((0, 0, self.W, self.H)))
        inpainted_pts, inpainted_colors = get_pcd(
            self.K, inpainted_rgb_np, depth_estimated_np, mask=np.logical_and(depth_estimated_np > 0, mask_np_cropped))
        inpainted_pcd = get_o3d_pointcloud(inpainted_pts, inpainted_colors, scale=1)
        inpainted_pcd.transform(pose_c)

        ori_normals = np.asarray(self.scene_pcd.normals)
        self.scene_pcd += inpainted_pcd
        self.scene_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.scene_pcd.orient_normals_towards_camera_location()
        self.scene_pcd.normals = o3d.utility.Vector3dVector(np.vstack((ori_normals, np.asarray(self.scene_pcd.normals)[ori_normals.shape[0]:,:])))
        self.scene_pcd = self.scene_pcd.voxel_down_sample(voxel_size=0.005)
        self.scene_pcd, _ = self.scene_pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)
        self.counter += 1

    def floor_grounding(self, inpainted_rgb_pil: Image.Image, rendered_depth_np: np.ndarray, pose_c: np.ndarray):
        """grounds depth estimation by anchoring floor pixels with depth rendering of the predicted floor plane

        :param inpainted_rgb_pil: image of the inpainted image
        :param rendered_depth_np: rendered depth of the scene point cloud
        :param pose_c: pose to render the target view
        :return: updated depth render with grounded floor
        """
        floor_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.floor_points))
        floor_pcd.transform(self.camera_T_floor)
        floor_pcd.paint_uniform_color([0, 0, 0])
        plane_model, inliers = floor_pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
        [a, b, c, d] = plane_model
        floor_pcd.normals = o3d.utility.Vector3dVector(plane_model[:3][None,:].repeat(self.floor_points.shape[0],axis=0))
        _, rendered_floor_depth_np = self.render_pointcloud(floor_pcd, self.K_torch, pose_c)

        semantic_seg = run_m2f_segmentation(self.args, np.array(inpainted_rgb_pil), './preprocess/')
        floor_seg = semantic_seg == self.floor_id

        rendered_depth_np += rendered_floor_depth_np * (floor_seg * (rendered_depth_np == 0))
        return rendered_depth_np

    def get_fused_pointcloud(self):
        return self.scene_pcd
