import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3

from utils.utils import Log, clone_obj
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.slam_utils import update_pose, to_se3_vec, get_loss_normal, get_loss_mapping_rgbd
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from camera_packet import CameraPacket, GlobalCameraPacket
try:
    from gaussian.gui import gui_utils, slam_gui
except ImportError:  # GUI deps (glfw/OpenGL/imgviz) are optional; rerun is the default viewer
    gui_utils = slam_gui = None


@dataclass(slots=True)
class CameraProjection:
    """Pinhole calibration and projection matrix for one camera."""

    K: list[float | int]
    """Camera parameters ``[fx, fy, cx, cy, width, height]``."""
    matrix: torch.Tensor
    """Transposed 4x4 projection matrix on the requested device."""


def build_camera_projection(
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    device: str = "cuda",
    znear: float = 0.01,
    zfar: float = 100.0,
) -> CameraProjection:
    """Build one camera's projection from pixel intrinsics and image size."""
    height, width = image_hw
    fx, fy, cx, cy = (float(value) for value in intrinsics[:4])
    K: list[float | int] = [fx, fy, cx, cy, width, height]
    matrix = getProjectionMatrix2(
        znear=znear,
        zfar=zfar,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        W=width,
        H=height,
    ).transpose(0, 1).to(device=device)
    return CameraProjection(K=K, matrix=matrix)

class GSBackEnd(mp.Process):
    def __init__(self, config, save_dir, use_gui=False, rr_logger=None):
        super().__init__()
        self.config = config
        self.rr = rr_logger
        
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui
        self.camera_projections: dict[int, CameraProjection] = {}

        if self.use_gui and gui_utils is None:
            raise RuntimeError("--gsvis requires glfw/imgviz/PyOpenGL, which are not in the pixi env")

        self.opt_params = munchify(config["opt_params"])
        self.config["Training"]["monocular"] = False

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.cameras_extent = float(self.config["Dataset"].get("cameras_extent", 6.0))
        self.gaussians.init_lr(self.cameras_extent)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        self.set_hyperparams()

        if self.use_gui:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(3)

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.lambda_dnormal = self.config["Training"]["lambda_dnormal"]


    def _log_renders(self, viewpoints_by_cam):
        """Log a splat-render vs ground-truth pair per camera to Rerun."""
        with torch.no_grad():
            for cam_idx, vp in viewpoints_by_cam.items():
                render_pkg = render(vp, self.gaussians, self.background)
                self.rr.log_render_comparison(cam_idx, render_pkg["render"], vp.original_image)

    def _projection_for_camera(
        self,
        cam_idx: int,
        intrinsics: torch.Tensor,
        image_hw: tuple[int, int],
    ) -> CameraProjection:
        """Return the cached projection for one camera, building it on first use."""
        if cam_idx not in self.camera_projections:
            dataset_config = self.config["Dataset"]
            self.camera_projections[cam_idx] = build_camera_projection(
                intrinsics,
                image_hw,
                znear=float(dataset_config.get("znear", 0.01)),
                zfar=float(dataset_config.get("zfar", 100.0)),
            )
        return self.camera_projections[cam_idx]

    def process_track_data(self, packet: CameraPacket):
        H, W = packet["images"].shape[-2:]
        cam_idx = int(packet.get('cam_idx', 0))
        projection = self._projection_for_camera(cam_idx, packet["intrinsics"][0], (H, W))
        w2c = SE3(packet["poses"]).matrix().cuda()
        for i, view_id_tensor in enumerate(packet["view_ids"]):
            view_id: int = int(view_id_tensor.item())
            frame_id: int = int(packet["frame_ids"][i].item())
            viewpoint = Camera.init_from_tracking(packet["images"][i]/255.0, packet["depths"][i], packet["normals"][i], w2c[i], view_id, projection.matrix, projection.K, frame_id, cam_idx=cam_idx)
            if view_id not in self.current_window:
                self.current_window = [view_id] + self.current_window[:-1] if len(self.current_window) > 10 else [view_id] + self.current_window
                if not self.initialized:
                    self.reset()
                    self.viewpoints[view_id] = viewpoint
                    self.add_next_kf(view_id, viewpoint, depth_map=packet["depths"][0].numpy(), init=True)
                    self.initialize_map(view_id, viewpoint)
                    self.initialized = True
                elif view_id not in self.viewpoints:
                    self.viewpoints[view_id] = viewpoint
                    self.add_next_kf(view_id, viewpoint, depth_map=packet["depths"][i].numpy())
                else:
                    self.viewpoints[view_id] = viewpoint

        self.map(self.current_window, iters=10)
        # self.map(self.current_window, iters=1, prune=True)

        if self.rr is not None:
            self._log_renders({cam_idx: viewpoint})

        if self.use_gui:
            keyframes = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    kf_window=current_window_dict,
                    gtcolor=viewpoint.original_image,
                    gtdepth=viewpoint.depth.numpy()))

    def process_global_track_data(self, packet: GlobalCameraPacket):
        if packet["pose_updates"] is not None:
            with torch.no_grad():
                view_ids = packet["view_ids"]
                pose_updates = packet["pose_updates"]
                scale_updates = packet["scale_updates"]
                if scale_updates is None:
                    raise ValueError("scale_updates are required when pose_updates are supplied")
                packet_indices = (
                    view_ids.unsqueeze(1) == self.gaussians.unique_kfIDs.unsqueeze(0)
                ).nonzero()[:, 0]
                pose_indices = packet_indices % len(pose_updates)
                updates = pose_updates.cuda()[pose_indices]
                updates_scale = scale_updates.cuda()[pose_indices]
                
                xyz = self.gaussians.get_xyz
                xyz = (updates * xyz) / updates_scale
                self.gaussians._xyz[:] = xyz

                scale = self.gaussians.get_scaling
                scale = scale / updates_scale
                self.gaussians._scaling[:] = self.gaussians.scaling_inverse_activation(scale)
 
                rot = SO3(self.gaussians.get_rotation)
                rot = SO3(updates.data[:,3:]) * rot
                self.gaussians._rotation[:] = rot.data

        H, W = packet["images"].shape[-2:]
        w2c = SE3(packet["poses"]).matrix().cuda()
        for i, view_id_tensor in enumerate(packet["view_ids"]):
            view_id: int = int(view_id_tensor.item())
            frame_id: int = int(packet["frame_ids"][i].item())
            cam_idx = int(packet["cam_indices"][i])
            projection = self._projection_for_camera(cam_idx, packet["intrinsics"][i], (H, W))
            viewpoint = Camera.init_from_tracking(packet["images"][i]/255.0, packet["depths"][i], packet["normals"][i], w2c[i], view_id, projection.matrix, projection.K, frame_id, cam_idx=cam_idx)
            if view_id not in self.current_window:
                self.current_window = [view_id] + self.current_window[:-1] if len(self.current_window) > 10 else [view_id] + self.current_window
                if not self.initialized:
                    self.reset()
                    self.viewpoints[view_id] = viewpoint
                    self.add_next_kf(view_id, viewpoint, depth_map=packet["depths"][0].numpy(), init=True)
                    self.initialize_map(view_id, viewpoint)
                    self.initialized = True
                elif view_id not in self.viewpoints:
                    self.viewpoints[view_id] = viewpoint
                    self.add_next_kf(view_id, viewpoint, depth_map=packet["depths"][i].numpy())
                else:
                    self.viewpoints[view_id] = viewpoint

        self.map(self.current_window, iters=10)
        # self.map(self.current_window, iters=1, prune=True)

        if self.rr is not None:
            self._log_renders({vp.cam_idx: vp for vp in self.viewpoints.values()})

        if self.use_gui:
            keyframes = [self.viewpoints[kf_idx] for kf_idx in self.current_window]
            current_window_dict = {}
            current_window_dict[self.current_window[0]] = self.current_window[1:]
            self.q_main2vis.put(
                gui_utils.GaussianPacket(
                    gaussians=clone_obj(self.gaussians),
                    current_frame=viewpoint,
                    keyframes=keyframes,
                    kf_window=current_window_dict,
                    gtcolor=viewpoint.original_image,
                    gtdepth=viewpoint.depth.numpy()))

    def finalize(self):
        self.color_refinement(iteration_total=self.gaussians.max_steps)
        self.gaussians.save_ply(f'{self.save_dir}/3dgs_final.ply')

        if self.rr is not None:
            self.rr.set_refine_iter(self.gaussians.max_steps)
            self.rr.log_text(f"final map: {self.gaussians.get_xyz.shape[0]} gaussians")
            self.rr.log_gaussians(self.gaussians, force=True, cap=False)
            self._log_renders({vp.cam_idx: vp for vp in self.viewpoints.values()})

        poses_cw = []
        for view in self.viewpoints.values():
            T_w2c = np.eye(4)
            T_w2c[0:3, 0:3] = view.R.cpu().numpy()
            T_w2c[0:3, 3] = view.T.cpu().numpy()
            poses_cw.append(np.hstack(([view.tstamp], to_se3_vec(T_w2c))))
        poses_cw.sort(key=lambda x: x[0])
        return np.stack(poses_cw)

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx, cam_idx=0):
        projection = self.camera_projections[cam_idx]
        eval_rendering(gtimages, gtdepthdir, traj, self.gaussians,self.save_dir, self.background,
            projection.matrix, projection.K, kf_idx, iteration="after_opt")
    
    def eval_rendering_kf(self) -> dict[str, float]:
        return eval_rendering_kf(
            self.viewpoints,
            self.gaussians,
            self.save_dir,
            self.background,
            iteration="after_opt",
        )

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.current_window = []
        self.initialized = False
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(viewpoint, self.gaussians, self.background)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"]
            )
            loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        Log("Initialized map")
        return render_pkg

    def map(self, current_window, iters, prune=False):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx not in current_window_set:
                random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            viewpoints = viewpoint_stack + [random_viewpoint_stack[idx] for idx in torch.randperm(len(random_viewpoint_stack))[:2]]
            for viewpoint in viewpoints:
                render_pkg = render(viewpoint, self.gaussians, self.background)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                loss_mapping += self.lambda_dnormal * get_loss_normal(depth, viewpoint) / 10.
                loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                self.gaussian_reset = 501
                if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                # self.gaussians.update_learning_rate(self.iteration_count)

    def color_refinement(self, iteration_total):
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(view.uid)})
            if self.config["Training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        for iteration in (pbar := trange(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(viewpoint_cam, self.gaussians, self.background)
            image, depth = render_pkg["render"], render_pkg["depth"]
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b

            gt_image = viewpoint_cam.original_image.cuda()
            loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += get_loss_mapping_rgbd(self.config, image, depth, viewpoint_cam)
            if iteration < 7000:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam)
            else:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam) / 2
            loss.backward()
            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            # skip iteration_total: finalize() logs the final map right after
            if self.rr is not None and iteration % self.rr.refine_every == 0 and iteration < iteration_total:
                self.rr.set_refine_iter(iteration)
                self.rr.log_gaussians(self.gaussians, force=True)

            if self.use_gui and iteration % 50 == 0:
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

            pbar.set_description(f"Global GS Refinement lr {lr:.3E} loss {loss.item():.3f}")

        Log("Map refinement done")
