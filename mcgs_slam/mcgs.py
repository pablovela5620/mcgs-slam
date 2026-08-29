import os
import cv2
import torch
import lietorch
from prior_mask import mask_invalid_depth
import numpy as np
from typing import TypedDict

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend

from collections import OrderedDict
from torch.multiprocessing import Process
from tqdm import trange

from gs_backend import GSBackEnd
from utils.utils import load_config

class CameraPacket(TypedDict):
    """One camera's tracking data passed to the Gaussian backend."""

    viz_idx: torch.Tensor
    tstamp: torch.Tensor
    poses: torch.Tensor
    images: torch.Tensor
    normals: torch.Tensor
    depths: torch.Tensor
    intrinsics: torch.Tensor
    cam_idx: int
    pose_updates: torch.Tensor | lietorch.SE3 | None
    scale_updates: torch.Tensor | None


def build_camera_packet(
    *,
    viz_idx: torch.Tensor,
    cam_idx: int,
    poses_camera_from_world_n7: torch.Tensor,
    images_rgb_n3hw: torch.Tensor,
    depth_metres_nhw: torch.Tensor,
    normals_n3hw: torch.Tensor,
    intrinsics_n4: torch.Tensor,
    scale_factor: float = 0.2,
    pose_updates: torch.Tensor | lietorch.SE3 | None = None,
    scale_updates: torch.Tensor | None = None,
) -> CameraPacket:
    """Build one Gaussian-backend packet from trusted camera observations.

    Args:
        viz_idx: Int64 mapper keyframe indices with shape ``[n]``.
        cam_idx: Camera index used for the unique ``+500*cam_idx`` viewpoint key.
        poses_camera_from_world_n7: Float32 lietorch ``T_camera_from_world``
            vectors with shape ``[n, 7]`` and metric translations.
        images_rgb_n3hw: UInt8 RGB images with shape ``[n, 3, h, w]``.
        depth_metres_nhw: Float32 metric depth with shape ``[n, h, w]``.
        normals_n3hw: Float32 camera-frame normals with shape ``[n, 3, h, w]``;
            a zero normal marks invalid prior geometry.
        intrinsics_n4: Float32 virtual-pinhole ``[fx, fy, cx, cy]`` values with
            shape ``[n, 4]``.
        scale_factor: Translation and depth scale into Gaussian-map units.
        pose_updates: Optional pose correction used only by the legacy global path.
        scale_updates: Optional scale correction used only by the legacy global path.

    Returns:
        Canonical CPU packet consumed by :class:`GSBackEnd`.
    """
    poses_scaled_n7: torch.Tensor = poses_camera_from_world_n7.detach().to(
        device="cpu", dtype=torch.float32
    ).clone()
    poses_scaled_n7[:, :3] *= scale_factor
    depth_scaled_nhw: torch.Tensor = depth_metres_nhw.detach().to(
        device="cpu", dtype=torch.float32
    ) * scale_factor
    return {
        "viz_idx": viz_idx.detach().to(device="cpu", dtype=torch.long),
        "tstamp": viz_idx.detach().to(device="cpu", dtype=torch.long) + 500 * cam_idx,
        "poses": poses_scaled_n7,
        "images": images_rgb_n3hw.detach().to(device="cpu", dtype=torch.uint8),
        "normals": normals_n3hw.detach().to(device="cpu", dtype=torch.float32),
        "depths": mask_invalid_depth(
            depth_scaled_nhw,
            normals_n3hw.detach().to(device="cpu", dtype=torch.float32),
        ),
        "intrinsics": intrinsics_n4.detach().to(device="cpu", dtype=torch.float32),
        "cam_idx": cam_idx,
        "pose_updates": pose_updates.to(device="cpu") if pose_updates is not None else None,
        "scale_updates": scale_updates.to(device="cpu") if scale_updates is not None else None,
    }


class Mcgs:
    def __init__(self, args, video=None, rr_logger=None, scale_factor: float = 0.2):
        super(Mcgs, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.config = load_config(args.config)
        self.scale_factor = scale_factor
        self.rr = rr_logger

        # store images, depth, poses, intrinsics (shared between processes)
        if video is None:
            self.video = DepthVideo(args, args.image_size, args.buffer)
        else:
            self.video = video

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh, args=args)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)
        
        # 3dgs
        self.gs = GSBackEnd(self.config, self.args.output, args.gsvis, rr_logger=rr_logger)

        # visualizer
        if args.vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

    def load_weights(self, weights):
        """ load trained model weights """

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
    
    def _scale_poses(self, poses, scale_factor: float):
        scaled_poses = poses.clone()
        scaled_poses[:, :3] *= scale_factor
        return scaled_poses

    def _camera_packet(
        self,
        viz_idx: torch.Tensor,
        cam_idx: int,
        dposes: torch.Tensor | lietorch.SE3 | None,
        dscales: torch.Tensor | None,
    ) -> CameraPacket:
        """Build the canonical Gaussian-backend packet for one rig camera."""
        poses_c0_w = lietorch.SE3(self.video.poses[viz_idx].to(device="cpu")[None])
        poses_ci_w = (self.video.T_ci_c0[cam_idx].cpu() * poses_c0_w).data[0]
        return build_camera_packet(
            viz_idx=viz_idx,
            cam_idx=cam_idx,
            poses_camera_from_world_n7=poses_ci_w,
            images_rgb_n3hw=self.video.images_up_list[cam_idx][viz_idx.cpu()],
            depth_metres_nhw=(
                1.0 / self.video.disps_up_list[cam_idx][viz_idx.cpu()].to(device="cpu")
            ),
            normals_n3hw=self.video.normals_list[cam_idx][viz_idx.cpu()],
            intrinsics_n4=self.video.K(cam_idx)[0, viz_idx].to(device="cpu")[:, :4] * 8,
            scale_factor=self.scale_factor,
            pose_updates=dposes,
            scale_updates=dscales,
        )

    def call_gs(self, viz_idx, dposes=None, dscale=None):

        if self.rr is not None:
            self.rr.log_keyframe(self.video, viz_idx)

        for cam_idx in range(self.video.multi):
            self.gs.process_track_data(self._camera_packet(viz_idx, cam_idx, dposes, dscale))

        # One splat snapshot per keyframe update (cadence inside the logger),
        # after every camera's packet has refined the map.
        if self.rr is not None:
            self.rr.log_gaussians(self.gs.gaussians)

    def call_global_gs(self, viz_idx, dposes=None, dscale=None):

        if self.rr is not None:
            self.rr.log_keyframe(self.video, viz_idx)

        packets: list[CameraPacket] = [
            self._camera_packet(viz_idx, cam_idx, dposes, dscale)
            for cam_idx in range(self.video.multi)
        ]
        tensor_keys: tuple[str, ...] = (
            "viz_idx", "tstamp", "poses", "images", "normals", "depths", "intrinsics"
        )
        final_data: dict[str, object] = {
            key: torch.cat([packet[key] for packet in packets], dim=0)
            for key in tensor_keys
        }
        final_data["cam_idx"] = torch.cat([
            torch.full_like(packet["viz_idx"], packet["cam_idx"], dtype=torch.long)
            for packet in packets
        ])
        final_data['pose_updates'] = lietorch.SE3(self._scale_poses(dposes.to(device='cpu').data, self.scale_factor)) if dposes is not None else None
        final_data['scale_updates'] = dscale.to(device='cpu') if dscale is not None else None
            
        self.gs.process_global_track_data(final_data, self.video.multi)

        if self.rr is not None:
            self.rr.log_gaussians(self.gs.gaussians, force=True)

    def track(self, t, tstamp, image, intrinsics):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(t, tstamp, image, intrinsics)

            # local bundle adjustment
            viz_idx = self.frontend()

            if self.video.counter.value >= (self.video.buffer - 15):
                window = 35
                self.frontend.release_buffer(window=window)
                self.video.release_buffer(window=window)
        
        if len(viz_idx):
            self.call_gs(viz_idx)

    def save_kf_poses(self, args, video, filename='traj_mcgs.txt'):
        N = video.total_counter
        kf_poses = lietorch.SE3(video.globuf.poses_all[:N][None]).inv().data.cpu().numpy()[0]    # poses_wc
        kf_stamps = sorted(video.globuf.kf_stamps_all.values())
        traj_file = os.path.join(args.output, filename)
        with open(traj_file, 'w') as f:
            for stamp, pose in zip(kf_stamps, kf_poses):
                pose = [stamp] + list(pose)
                pose = [str(i) for i in pose]
                pose = " ".join(pose)
                f.write(pose + "\n")
        print('saved pose file to', traj_file)

    def global_pose_ba(self):
        """ terminate the visualization process, return poses [t, q] """
        torch.cuda.empty_cache()

        for iter in trange(5, desc='Global BA outer loop'):
            print("#"*64, f" iter {iter} ", "#"*64)
            self.backend.pose_ba(iter, 6)
            torch.cuda.empty_cache()

        # self.video.globuf.poses_all = self.video.poses.cpu()
        # self.video.globuf.disps_all = self.video.disps.cpu()
        # self.video.globuf.images_all = self.video.images
        # if self.video.multi:
        #     self.video.globuf.images_all_list = self.video.images_list
        #     self.video.globuf.disps_all_list = [d.cpu() for d in self.video.disps_list]
    
    def terminate(self):
        
        del self.frontend

        # global bundle adjustment
        poses_pre = self.video.poses[:self.video.counter.value].clone()
        
        self.global_pose_ba()
        del self.backend
        
        poses_pos = self.video.poses[:self.video.counter.value].clone()
        
        dposes = lietorch.SE3(poses_pos).inv() * lietorch.SE3(poses_pre)
        dscale = torch.ones(self.video.counter.value, 1)
        torch.cuda.empty_cache()

        # final refinement
        self.call_global_gs(torch.arange(0, self.video.counter.value, device='cuda'), dposes, dscale)
        self.gs.finalize()
        
        self.gs.eval_rendering_kf()
