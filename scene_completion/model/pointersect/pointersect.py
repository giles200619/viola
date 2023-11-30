import os
import torch
import open3d as o3d

from .inference import infer
from .inference.structures import PointCloud, Camera


class PointersectInference:
    def __init__(self, im_size) -> None:
        self.model_pth_filename = os.path.join('checkpoints/pointersect_epoch700.pth')
        self.data_device = torch.device('cpu')
        self.model_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.output_camera_setting = {
            'fov': 60,
            'width_px': im_size[0],
            'height_px': im_size[1],
            'ray_offsets': 'center',
        }

        self.pr_setting = dict(
            ray_radius=None,  # 0.1,  # if <0, set to grid_width/grid_size *2
            grid_size=None,  # 100,
            grid_center=None,  # 0,
            grid_width=None,
            # 2.2 * mesh_scale,  # full width, it is for mesh range: [-scale, scale] + some margin for numerical error
        )

        self.model_loading_settings = dict(
            loss_name='test_epoch_loss_hit',  # 'valid_epoch_loss_hit',  # training test set is not the test set
            loss_smooth_window=100,  # 1,
            loss_smooth_std=30.,  # 1.,
        )

    def render_pointcloud(self,
                          o3d_pcd: o3d.geometry.PointCloud,
                          intrinsics: torch.tensor,
                          H_c2w: torch.tensor,
                          ):

        input_point_cloud = PointCloud.from_o3d_pcd(o3d_pcd=o3d_pcd)
        *b_shape, _, _ = H_c2w.shape

        output_cameras = Camera(H_c2w=H_c2w,
                                intrinsic=intrinsics.expand(*b_shape, 3, 3),
                                width_px=self.output_camera_setting['width_px'],
                                height_px=self.output_camera_setting['height_px'],
                                )

        pointersect_result = infer.render_point_cloud_camera_using_pointersect(
            model_filename=self.model_pth_filename,
            k=30,
            point_cloud=input_point_cloud,
            output_cameras=output_cameras,
            output_camera_setting=None,
            model_loading_settings=self.model_loading_settings,
            pr_setting=self.pr_setting,
            model_device=self.model_device,
            data_device=self.data_device,
            print_out=True,
        )  # (b, q, h, w)

        rendered_rgbd = pointersect_result.get_rgbd_image(output_cameras)
        if b_shape == [1, 1]:
            rendered_rgb_np = rendered_rgbd.rgb[0, 0].detach().numpy()
            rendered_depth_np = rendered_rgbd.depth[0, 0].detach().numpy()

        else:
            rendered_rgb_np = rendered_rgbd.rgb.detach().numpy()
            rendered_depth_np = rendered_rgbd.depth.detach().numpy()

        return rendered_rgb_np, rendered_depth_np
