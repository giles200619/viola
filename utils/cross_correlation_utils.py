from scipy.io import loadmat
import open3d as o3d
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
from copy import deepcopy
from tqdm import tqdm
import argparse
import sys
sys.path.append('./')

def get_cross_correlation_matches(partial_2d: np.ndarray,
                                  template_2d: np.ndarray,
                                  nms: bool = False,
                                  vis_images: bool = False,
                                  ) -> np.ndarray:
    """get pose estimates aligning the partial_2d point cloud to template_2d using image-based normalized cross-correlation

    :param partial_2d: 2d projection of camera reconstruction points in homogeneous coordinates (N, 3)
    :param template_2d: 2d lidar point cloud in homogeneous coordinates (M, 3)
    :param nms: use non-max suppression, defaults to False
    :param vis_images: visualize images, defaults to False
    :return: a list of pose estimates
    """

    resolution = 100
    lidar_im_uint = image_from_points(template_2d, resolution=resolution)

    pad = 400
    lidar_im_uint = cv2.copyMakeBorder(lidar_im_uint, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    n_rots = 360
    rot_angles_deg = np.linspace(0, 360, n_rots + 1)[:-1]

    res_bests = []
    max_locs_all = []
    im_sizes = np.zeros((len(rot_angles_deg)))
    k_at_each_angle = 100
    for ii, rot_angle in (enumerate(rot_angles_deg)):
        partial_pts = partial_2d @ R.from_euler('z', rot_angle, degrees=True).as_matrix().T
        partial_im_uint = image_from_points(partial_pts)

        res_i = cv2.matchTemplate(lidar_im_uint, partial_im_uint, cv2.TM_CCOEFF_NORMED)
        max_locs = k_largest_index_argsort(res_i, k_at_each_angle)
        res_bests.append(res_i[max_locs[:, 0], max_locs[:, 1]])
        max_locs_all.append(max_locs[:, ::-1])

        im_size = partial_im_uint.shape[0]
        im_sizes[ii] = im_size

    res_bests = np.stack(res_bests)
    max_locs_all = np.stack(max_locs_all)

    # import pdb; pdb.set_trace()
    tls = max_locs_all.reshape(-1, 2)
    all_sizes = im_sizes[:, None].repeat(k_at_each_angle, -1).reshape(-1)
    brs = tls + all_sizes[:, None]
    boxes = np.hstack((tls, brs))

    # # Get the indices of best 100
    # inds = k_largest_index_argsort(res_bests, 100)
    # tls = max_locs_all[inds[:, 0], inds[:, 1]]
    # brs = tls + im_sizes[inds[:, 0]][:, None]
    # boxes = np.hstack((tls, brs))
    boxes, nms_inds = non_max_suppression_fast(boxes, 0.75)
    nms_inds_np = np.array(nms_inds)
    angle_inds = nms_inds_np // k_at_each_angle

    # if nms:
    #     boxes, nms_inds = non_max_suppression_fast(boxes, 0.9)
    #     inds = inds[nms_inds]

    pred_tforms = []

    for box, ix in zip(boxes, angle_inds):
        # rot_angle = rot_angles_deg[ix[0]]
        rot_angle = rot_angles_deg[ix]
        partial_pts = partial_2d @ R.from_euler('z', rot_angle, degrees=True).as_matrix().T

        tl = box[:2]
        br = box[2:4]
        tl = tl.astype(int)
        br = br.astype(int)
        im_size = br[0] - tl[0]
        # assert (tl == max_locs_all[ix[0], ix[1]]).all()

        if vis_images:
            partial_im_uint = image_from_points(partial_pts)
            partial_im_uint_3ch = partial_im_uint[..., None].repeat(3, -1)
            partial_im_uint_3ch[..., 1:] = 0

            lidar_im_uint_3ch = lidar_im_uint[..., None].repeat(3, -1)
            cv2.rectangle(lidar_im_uint_3ch, tl, br, (0, 0, 255), 2)
            im_sz = partial_im_uint_3ch.shape[:2]
            lidar_im_uint_3ch[tl[1]:tl[1]+im_sz[0], tl[0]:tl[0]+im_sz[1]] = partial_im_uint_3ch
            Image.fromarray(lidar_im_uint_3ch).show()

        lidar_pt = tl / float(resolution) + template_2d.min(0)[:2] - (pad / float(resolution))
        rot_partial_pt = partial_pts.min(0)[:2]
        pred_trans = lidar_pt - rot_partial_pt
        pred_ori = rot_angle
        pred_tform = R.from_euler('z', pred_ori, degrees=True).as_matrix()
        pred_tform[:2, -1] = pred_trans
        pred_tforms.append(pred_tform)

    res_inds = np.array([(nms_inds_np // k_at_each_angle), (nms_inds_np % k_at_each_angle)]).T
    res_scores = res_bests[res_inds[:, 0], res_inds[:, 1]]
    sorted_inds = np.argsort(res_scores)[:200]
    pred_tforms_np = np.stack(pred_tforms)
    pred_tforms_np = pred_tforms_np[sorted_inds]

    return pred_tforms_np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='redwood_processed/apartment_long_0.mat')
    parser.add_argument('--vis3d', action='store_true')
    parser.add_argument('--vis_images', action='store_true')
    parser.add_argument('--nms', action='store_true')

    return parser.parse_args()


def main():
    args = get_args()
    data = loadmat(args.data_path)
    print('Processing...', args.data_path)

    partial_2d = data['partial_2d']
    template_2d = data['template_2d']

    # Position and orientation error bounds for successful runs
    error_bound_pos = 0.5  # [m]
    error_bound_ori = 10  # [deg]
    error_bound = np.array([error_bound_pos, error_bound_ori])

    # Position and orientation errors
    errors = []

    pred_tforms = get_cross_correlation_matches(partial_2d, template_2d, args.nms, args.vis_images)

    for pred_tform in pred_tforms:

        pred_rot = np.eye(3)
        pred_rot[:2, :2] = pred_tform[:2, :2]
        pred_ori = R.from_matrix(pred_rot).as_euler('xyz')[-1] * 180 / np.pi % 360

        gt_rot = np.eye(3)
        gt_rot[:2, :2] = data['gt_pose_2d'][:2, :2]
        gt_ori = R.from_matrix(gt_rot).as_euler('xyz')[-1] * 180 / np.pi % 360
        ori_error = abs((gt_ori - pred_ori + 180) % 360 - 180)

        pred_trans = pred_tform[:2, -1]
        pos_error = np.linalg.norm(data['gt_pose_2d'][:2, -1] - pred_trans)

        error_i = [pos_error, ori_error]
        errors.append(error_i)

    print('Any successful:', (np.stack(errors) < error_bound).all(-1).any())

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def image_from_points(points, resolution=100):
    pts_pixelized = ((points - points.min(0)) * resolution).astype(int)
    im_size = (((points.max(0) - points.min(0)) * resolution).astype(int) + 1)[:2]
    image = np.zeros(im_size[::-1])

    k = 5
    if k > 1:
        vecs = np.argwhere(np.zeros((k, k)) == 0) - (k-1)//2
        pts_pixelized = pts_pixelized[None][..., :2] + vecs[:, None]
        pts_pixelized = pts_pixelized.reshape(-1, 2).clip(0, [im_size-1])

    image[pts_pixelized[:, 1], pts_pixelized[:, 0]] = 1

    im_uint = (image * 255).astype(np.uint8)
    return im_uint


# Credit: https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the idxs list
    while len(idxs) > 0:

        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    return boxes[pick].astype("int"), pick

if __name__ == "__main__":
    main()
