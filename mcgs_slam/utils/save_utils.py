import os
import cv2
import torch
import numpy as np
import lietorch
import droid_backends
from utils.plot_depth_map import colorize_np


def save_pc_one(poses, disps, images, intrinsics, index, cam, output, suffix):
    points = droid_backends.iproj(lietorch.SE3(poses).inv().data, disps, intrinsics).cpu()
    images = images.cpu()[:, [2, 1, 0]].permute(0, 2, 3, 1)

    thresh = 0.01 * torch.ones_like(disps.mean(dim=[1, 2]))
    count = droid_backends.depth_filter(poses, disps, intrinsics, index, thresh)
    masks = ((count >= 3) & (disps > .5*disps.mean(dim=[1, 2], keepdim=True)))
    masks = masks.reshape(-1)
    points_masked = points.reshape(-1, 3)[masks.cpu()].numpy()
    colors_masked = images.reshape(-1, 3)[masks.cpu()].numpy()
    pcwrite(output + f"/pc_{cam}{suffix}.ply", points_masked, colors_masked)


def iproj_single(poses, intrinsic, disps, index):
    points = droid_backends.iproj(poses.inv().data[0], disps, intrinsic).cpu()
    thresh = 0.02 * torch.ones_like(disps.mean(dim=[1, 2]))
    count = droid_backends.depth_filter(poses.data[0], disps, intrinsic, index.cuda(), thresh)
    masks = ((count >= 3) & (disps > .5*disps.mean(dim=[1, 2], keepdim=True)))
    return points, masks

def save_pc(args, video, output, suffix=''):
    index = torch.arange(video.total_counter, device="cpu")
    poses = torch.index_select(video.globuf.poses_all, 0, index)
    disps = torch.index_select(video.globuf.disps_all, 0, index)
    images = torch.index_select(video.globuf.images_all, 0, index)
    save_pc_one(poses.cuda(), disps.cuda(), images, video.intrinsics[0, 0], index.cuda(), 1, output, suffix)
    for ix in range(1, args.multi-1):
        posesi = (video.T_ci_c0[ix] * lietorch.SE3(poses.cuda()[None])).data[0]
        imagesi = torch.index_select(video.globuf.images_all_list[ix], 0, index)
        dispsi = torch.index_select(video.globuf.disps_all_list[ix], 0, index)
        save_pc_one(posesi, dispsi.cuda(), imagesi, video.intrinsics[0,ix+1], index.cuda(), ix+1, output, suffix)

def pcwrite(filename, points, colors):
    """Save a point cloud to a polygon .ply file.
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (points.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(points.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            points[i, 0], points[i, 1], points[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))
