import os
import cv2
import torch
import lietorch
import numpy as np

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

class Mcgs:
    def __init__(self, args, video=None):
        super(Mcgs, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.config = load_config(args.config)
        self.scale_factor = 0.2

        # store images, depth, poses, intrinsics (shared between processes)
        if video is None:
            self.video = DepthVideo(args, args.image_size, args.buffer, stereo=args.stereo)
        else:
            self.video = video

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh, args=args)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)
        
        # 3dgs
        self.gs = GSBackEnd(self.config, self.args.output, args.gsvis)

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
    
    def _scale_poses(self, poses, scale_factor=0.2):
        scaled_poses = poses.clone()
        scaled_poses[:, :3] *= scale_factor
        return scaled_poses

    def call_gs(self, viz_idx, dposes=None, dscale=None):
        
        data = {'viz_idx':  viz_idx.to(device='cpu'),
                'tstamp':   self.video.tstamp[viz_idx].to(device='cpu'),
                'poses':    self._scale_poses(self.video.poses[viz_idx].to(device='cpu'), scale_factor = self.scale_factor),
                'images':   self.video.images_up[viz_idx.cpu()],
                'normals':  self.video.normals[viz_idx.cpu()],
                'depths':   self.scale_factor / self.video.disps_up[viz_idx.cpu()].to(device='cpu'),
                'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu')[:, 0, :4] * 8,
                'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                'scale_updates': dscale.to(device='cpu') if dscale is not None else None}
        
        self.gs.process_track_data(data)
        
        if self.video.multi:
            for i in range(1, self.video.multi-1):
                
                T_ci_c0 = self.video.T_ci_c0[i]
                
                data = {'viz_idx':  viz_idx.to(device='cpu'),
                        'tstamp':   self.video.tstamp[viz_idx].to(device='cpu') + 500 * i,
                        'poses':    self._scale_poses((T_ci_c0.cpu() * lietorch.SE3(self.video.poses[viz_idx].to(device='cpu')[None])).data[0], 
                                                        scale_factor = self.scale_factor),
                        'images':   self.video.images_up_list[i][viz_idx.cpu()],
                        'normals':  self.video.normals_list[i][viz_idx.cpu()],
                        'depths':   self.scale_factor / self.video.disps_up_list[i][viz_idx.cpu()].to(device='cpu'),
                        'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu')[:, i + 1, :4] * 8,
                        'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                        'scale_updates': dscale.to(device='cpu') if dscale is not None else None}
            
                self.gs.process_track_data(data)
    
    def call_global_gs(self, viz_idx, dposes=None, dscale=None):
        
        multi_cam_data = {
            'viz_idx': [],
            'tstamp': [],
            'poses': [],
            'images': [],
            'normals': [],
            'depths': [],
            'intrinsics': []
        }

        multi_cam_data['viz_idx'].append(viz_idx.to(device='cpu'))
        multi_cam_data['tstamp'].append(self.video.tstamp[viz_idx].to(device='cpu'))
        multi_cam_data['poses'].append(
            self._scale_poses(self.video.poses[viz_idx].to(device='cpu'), scale_factor = self.scale_factor)
        )
        multi_cam_data['images'].append(self.video.images_up[viz_idx.cpu()])
        multi_cam_data['normals'].append(self.video.normals[viz_idx.cpu()])
        multi_cam_data['depths'].append(self.scale_factor / self.video.disps_up[viz_idx.cpu()].to(device='cpu'))
        multi_cam_data['intrinsics'].append(self.video.intrinsics[viz_idx].to(device='cpu')[:, 0, :4] * 8)

        if self.video.multi:
            for i in range(1, self.video.multi - 1):
                T_ci_c0 = self.video.T_ci_c0[i]

                multi_cam_data['viz_idx'].append(viz_idx.to(device='cpu'))
                multi_cam_data['tstamp'].append(self.video.tstamp[viz_idx].to(device='cpu') + 500 * i)
                multi_cam_data['poses'].append(self._scale_poses((T_ci_c0.cpu() * lietorch.SE3(self.video.poses[viz_idx].to(device='cpu')[None])).data[0], 
                                                        scale_factor = self.scale_factor))
                multi_cam_data['images'].append(self.video.images_up_list[i][viz_idx.cpu()])
                multi_cam_data['normals'].append(self.video.normals_list[i][viz_idx.cpu()])
                multi_cam_data['depths'].append(self.scale_factor / self.video.disps_up_list[i][viz_idx.cpu()].to(device='cpu'))
                multi_cam_data['intrinsics'].append(self.video.intrinsics[viz_idx].to(device='cpu')[:, i + 1, :4] * 8)
                
        final_data = {
            k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
            for k, v in multi_cam_data.items()
        }
        
        final_data['pose_updates'] = lietorch.SE3(self._scale_poses(dposes.to(device='cpu').data, self.scale_factor)) if dposes is not None else None
        final_data['scale_updates'] = dscale.to(device='cpu') if dscale is not None else None
            
        self.gs.process_global_track_data(final_data, self.video.multi - 1)

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
