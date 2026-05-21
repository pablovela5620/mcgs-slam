import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def visualization(video, device="cuda:0"):
    """ visualization frontend """

    torch.cuda.set_device(device)
    visualization.video = video
    visualization.cameras = {}
    visualization.points = {}
    visualization.factors = {}
    visualization.warmup = 1
    visualization.scale = 2.0
    visualization.ix = 0

    visualization.filter_thresh = 0.005
    visualization.count = 2
    visualization.show_inactive_frame = True

    def increase_filter_errtol(vis):
        visualization.filter_thresh *= 2
        print("- increase vis filter to", visualization.filter_thresh)
        with visualization.video.get_lock():
            visualization.video.dirty[:visualization.video.counter.value] = True

    def decrease_filter_errtol(vis):
        visualization.filter_thresh *= 0.5
        print("- decrease vis filter to", visualization.filter_thresh)
        with visualization.video.get_lock():
            visualization.video.dirty[:visualization.video.counter.value] = True

    def increase_filter_count(vis):
        visualization.count += 1
        print("- increase vis filter count to", visualization.count)
        with visualization.video.get_lock():
            visualization.video.dirty[:visualization.video.counter.value] = True

    def decrease_filter_count(vis):
        visualization.count -= 1
        print("- decrease vis filter count to", visualization.count)
        with visualization.video.get_lock():
            visualization.video.dirty[:visualization.video.counter.value] = True

    def disable_inactive_frames(vis):
        visualization.show_inactive_frame = not visualization.show_inactive_frame
        with visualization.video.get_lock():
            visualization.video.dirty[:visualization.video.counter.value] = True
        
    def save_pc(vis):
        point_cloud = o3d.geometry.PointCloud()
        for v in visualization.points.values():
            point_cloud += v
        filename = "result_f{}_c{}.pcd".format(visualization.filter_thresh, visualization.count)
        o3d.io.write_point_cloud(filename, point_cloud)
        print("saved point cloud to", filename)

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()

        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if video.counter.value >= (video.buffer - 15):
                return
            if len(dirty_index) == 0:
                return

            # vis factors
            if len(visualization.factors) > 0:
                vis.remove_geometry(visualization.factors[0])
                del visualization.factors[0]

            if video.num_factors.value > 0 and visualization.show_inactive_frame:
                positions = SE3(video.poses).inv().data[:,:3].cpu().numpy()
                lines = torch.cat((video.ii, video.jj), dim=-1).cpu().numpy()[:video.num_factors.value]
                # print("!!!!!!!!!!!!!!vis factors", video.num_factors.value, positions.shape, lines.shape)
                factors = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(positions),
                    lines=o3d.utility.Vector2iVector(lines))
                factors.paint_uniform_color((0.0, 0.0, 0.0))
                vis.add_geometry(factors)
                visualization.factors[0] = factors

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index.cpu())
            images = images[:,[2,1,0]].permute(0,2,3,1) / 255.0
            def process_one(poses, disps, posesall, dispsall, intrinsics):
                points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics).cpu()
                thresh = visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
                count = droid_backends.depth_filter(posesall, dispsall, intrinsics, dirty_index, thresh)
                masks = ((count >= visualization.count) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
                return points, masks
            points, masks = process_one(poses, disps, video.poses, video.disps, video.intrinsics[0, 0])

            if video.multi:
                masks_multi, points_multi, images_multi = [], [], []
                for ix in range(1, video.multi-1):
                    imagesi = torch.index_select(video.images_list[ix], 0, dirty_index.cpu())
                    images_multi.append(imagesi[:,[2,1,0]].permute(0,2,3,1) / 255.0)
                    posesiall = (video.T_ci_c0[ix] * SE3(video.poses[None])).data[0]
                    posesi = torch.index_select(posesiall, 0, dirty_index)
                    dispsi = torch.index_select(video.disps_list[ix], 0, dirty_index)
                    pointsi, masksi = process_one(posesi, dispsi, posesiall, video.disps_list[ix], video.intrinsics[0, ix+1])
                    points_multi.append(pointsi)
                    masks_multi.append(masksi)

            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item() + video.globuf.offset.value

                if ix in visualization.cameras:
                    vis.remove_geometry(visualization.cameras[ix])
                    del visualization.cameras[ix]

                if ix in visualization.points:
                    vis.remove_geometry(visualization.points[ix])
                    del visualization.points[ix]

                ### add camera actor ###
                if i == (len(dirty_index)-1):
                    cam_actor = create_camera_actor(False, scale=0.2*visualization.scale)
                else:
                    cam_actor = create_camera_actor(True, scale=0.1*visualization.scale)
                cam_actor.transform(pose)

                if visualization.show_inactive_frame or i == (len(dirty_index)-1):
                    vis.add_geometry(cam_actor)
                    visualization.cameras[ix] = cam_actor

                def add(masks, points, images):
                    mask = masks[i].reshape(-1)
                    pts = points[i].reshape(-1, 3)[mask.cpu()].numpy()
                    clr = images[i].reshape(-1, 3)[mask.cpu()].numpy()
                    return create_point_actor(pts, clr)

                point_actor = add(masks, points, images)
                if video.multi:
                    for ix in range(video.multi-2):
                        point_actor += add(masks_multi[ix], points_multi[ix], images_multi[ix])

                ## add point actor ###
                vis.add_geometry(point_actor)
                visualization.points[ix] = point_actor

            # hack to allow interacting with vizualization during inference
            if len(visualization.cameras) >= visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

            visualization.ix += 1

    ### create Open3D visualization ###
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("S"), increase_filter_errtol)
    vis.register_key_callback(ord("A"), decrease_filter_errtol)
    vis.register_key_callback(ord("Q"), increase_filter_count)
    vis.register_key_callback(ord("W"), decrease_filter_count)
    vis.register_key_callback(ord("P"), save_pc)
    vis.register_key_callback(ord("C"), disable_inactive_frames)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(visualization.scale))

    vis.run()
    vis.destroy_window()
