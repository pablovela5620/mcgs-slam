"""Integrate MCGS-SLAM rendered depth maps into a fused TSDF mesh.

Consumes a single MCGS-SLAM output folder:

    {result}/traj_mcgs.txt                          # cam0 keyframe poses (c2w)
    {result}/renders/image_after_opt/cam{i}/*.jpg   # per-camera color renders
    {result}/renders/depth_after_opt/cam{i}/*.png   # per-camera depth renders

Per-camera intrinsics and the camera rig extrinsics (T_cami_cam0) are read from
the calibration yaml (calib/<seq>.yml). traj_mcgs.txt stores only the front
camera (cam0) trajectory; the pose of every other camera is reconstructed as

    T_world->cami = T_cami_cam0 @ inv(T_cam0->world)

The renders live in a frame scaled by `scale_factor` (0.2) relative to the
metric tracking frame of traj_mcgs.txt, so that factor is folded into the depth
scale and the fused mesh comes out in the same metric frame as traj_mcgs.txt.

With --per_camera a separate mesh is also written for each camera.
"""

import os
import gc
import argparse
import time
import numpy as np
import open3d as o3d
import cv2
import yaml
from glob import glob
from tqdm import trange
from scipy.spatial.transform import Rotation as R


def release_gpu_cache():
    """Best-effort free of Open3D / Torch cached GPU memory."""
    gc.collect()
    try:
        o3d.core.cuda.release_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def to_se3_matrix(pvec):
    """[timestamp, tx, ty, tz, qx, qy, qz, qw] -> 4x4 pose matrix."""
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:8]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose


def pose7_to_matrix(vec):
    """[tx, ty, tz, qx, qy, qz, qw] -> 4x4 pose matrix."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(vec[3:7]).as_matrix()
    T[:3, 3] = vec[0:3]
    return T


def load_calib(calib_path):
    """Read per-camera intrinsics and rig extrinsics from calib/<seq>.yml.

    Returns (intrinsics, T_cami_cam0) where intrinsics is (C, >=4) with rows
    [fx, fy, cx, cy, ...] and T_cami_cam0 is a list of 4x4 matrices.
    """
    with open(calib_path, 'r') as f:
        params = yaml.safe_load(f)
    intrinsics = np.array(params['intrinsic'], dtype=np.float64)
    raw = params.get('T_cami_cam0', [[0, 0, 0, 0, 0, 0, 1]])
    T_cami_cam0 = [pose7_to_matrix(np.asarray(t, dtype=np.float64)) for t in raw]
    return intrinsics, T_cami_cam0


def discover_cameras(result):
    """Return sorted camera folder names ('cam0', 'cam1', ...) under renders/."""
    depth_root = f'{result}/renders/depth_after_opt'
    cams = sorted(glob(f'{depth_root}/cam*'),
                  key=lambda d: int(os.path.basename(d)[3:]))
    return [os.path.basename(c) for c in cams]


def extract_mesh(vbg, weight):
    """Try GPU marching cubes; fall back to CPU (after freeing GPU cache) on OOM."""
    release_gpu_cache()
    try:
        return vbg.extract_triangle_mesh(weight_threshold=weight).to_legacy()
    except RuntimeError as e:
        msg = str(e)
        if ('assistance mesh structure' not in msg
                and 'Marching Cubes' not in msg
                and 'out of memory' not in msg):
            raise
        print("  GPU mesh extraction failed, moving VBG to CPU...")
        release_gpu_cache()
        vbg_cpu = vbg.cpu()
        del vbg
        release_gpu_cache()
        return vbg_cpu.extract_triangle_mesh(weight_threshold=weight).to_legacy()


def build_camera_part(result, cam_name, cam_intrinsic, T_ci_c0, traj, args):
    """Build one integration part (rgb / depth / extrinsic / intrinsic) for a camera.

    `cam_intrinsic` is the [fx, fy, cx, cy, ...] row from the calib yaml at the
    full sensor resolution; it is rescaled here to the render resolution.
    `T_ci_c0` maps cam0 poses into this camera's frame.
    """
    depth_files = sorted(glob(f'{result}/renders/depth_after_opt/{cam_name}/*'))
    color_files = sorted(glob(f'{result}/renders/image_after_opt/{cam_name}/*'))
    n = min(len(depth_files), len(color_files), len(traj))
    if n == 0:
        raise RuntimeError(f"No renders / poses found for {cam_name}")
    depth_files, color_files = depth_files[:n], color_files[:n]

    # Rescale intrinsics from the sensor resolution to the render resolution.
    sample = cv2.imread(depth_files[0], cv2.IMREAD_ANYDEPTH)
    h_render, w_render = sample.shape[:2]
    sx = w_render / args.orig_size[0]
    sy = h_render / args.orig_size[1]
    fx, fy = cam_intrinsic[0] * sx, cam_intrinsic[1] * sy
    cx, cy = cam_intrinsic[2] * sx, cam_intrinsic[3] * sy
    intrinsic = o3d.core.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                                dtype=o3d.core.Dtype.Float64)

    # Each render k corresponds to keyframe k; cam0 pose comes from traj_mcgs.txt.
    extrinsic = []
    for k in range(n):
        T_c2w0 = to_se3_matrix(traj[k])             # cam0 camera-to-world
        T_w2ci = T_ci_c0 @ np.linalg.inv(T_c2w0)    # world-to-this-camera
        extrinsic.append(o3d.core.Tensor(T_w2ci, dtype=o3d.core.Dtype.Float64))

    if args.stride > 1:
        depth_files = depth_files[::args.stride]
        color_files = color_files[::args.stride]
        extrinsic = extrinsic[::args.stride]

    return {'name': cam_name, 'rgb': color_files, 'depth': depth_files,
            'extrinsic': extrinsic, 'intrinsic': intrinsic}


def integrate_multi(parts, args):
    """Integrate (rgb, depth, extrinsic) triplets from one or more cameras into one VBG.

    Each part carries its own intrinsic. The render frame is scaled by
    `scale_factor` w.r.t. the metric trajectory frame, so that factor is folded
    into the effective depth scale.
    """
    device = o3d.core.Device(args.device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=args.voxel_size,
        block_count=200000,
        device=device,
    )

    eff_depth_scale = args.depth_scale * args.scale_factor
    total_frames = sum(len(p['rgb']) for p in parts)
    pbar = trange(total_frames, desc="Integration")
    start = time.time()
    for p in parts:
        intrinsic = p['intrinsic']
        for i in range(len(p['rgb'])):
            pbar.set_description(f"{p['name']} {i + 1}/{len(p['rgb'])}")
            depth_path = p['depth'][i]
            color_path = p['rgb'][i]
            pose = p['extrinsic'][i]

            # GS-rendered depth saturates at the uint16 ceiling in low-coverage
            # regions (sky, image borders); zero those pixels and an optional
            # border margin so they are not fused as false geometry.
            dep_raw = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            dep_raw[dep_raw >= args.depth_clip] = 0
            if args.crop_border > 0:
                b = args.crop_border
                dep_raw[:b, :] = 0
                dep_raw[-b:, :] = 0
                dep_raw[:, :b] = 0
                dep_raw[:, -b:] = 0

            dep_m = dep_raw / eff_depth_scale
            if not ((dep_m > 0) & (dep_m < args.depth_max)).any():
                pbar.update(1)
                continue

            depth = o3d.t.geometry.Image(o3d.core.Tensor(dep_raw[..., None])).to(device)
            color = o3d.t.io.read_image(color_path).to(device)
            frustum = vbg.compute_unique_block_coordinates(
                depth, intrinsic, pose, eff_depth_scale, args.depth_max)
            vbg.integrate(frustum, depth, color, intrinsic, pose,
                          eff_depth_scale, args.depth_max)
            pbar.update(1)
    pbar.close()
    print(f"Integration took {time.time() - start:.2f}s")
    return vbg


def main():
    parser = argparse.ArgumentParser(
        description='Integrate MCGS-SLAM rendered depth maps into a TSDF mesh.')
    parser.add_argument('--result', type=str, required=True,
                        help='Path to the MCGS-SLAM output folder (e.g. output/100613)')
    parser.add_argument('--calib', type=str, default=None,
                        help='Path to calib yaml. Default: calib/<seq>.yml.')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='Voxel size (m)')
    parser.add_argument('--depth_scale', type=float, default=6553.5,
                        help='Depth PNG uint16 -> render-frame units divisor')
    parser.add_argument('--depth_max', type=float, default=100.0,
                        help='Maximum depth in the metric trajectory frame (m)')
    parser.add_argument('--depth_clip', type=int, default=65535,
                        help='Depth PNG values >= this are dropped (GS saturation '
                             'ceiling, i.e. unreliable sky / low-coverage pixels)')
    parser.add_argument('--crop_border', type=int, default=0,
                        help='Zero out this many border pixels of each depth map '
                             'before fusion (0 = keep full frame)')
    parser.add_argument('--scale_factor', type=float, default=0.2,
                        help='Render-frame / tracking-frame scale (MCGS-SLAM uses 0.2)')
    parser.add_argument('--orig_size', type=int, nargs=2, default=[1920, 1280],
                        metavar=('W', 'H'),
                        help='Full sensor resolution the calib intrinsics refer to')
    parser.add_argument('--weight', type=float, default=[0.001], nargs='+',
                        help='Weight threshold(s) for mesh extraction')
    parser.add_argument('--stride', type=int, default=1, help='Use every N-th keyframe')
    parser.add_argument('--device', type=str, default='cpu:0',
                        help="Open3D device, e.g. 'cuda:0' or 'cpu:0'")
    parser.add_argument('--per_camera', action='store_true',
                        help='Also save a separate mesh for each camera')
    args = parser.parse_args()

    seq = os.path.basename(os.path.normpath(args.result))
    calib_path = args.calib or os.path.join('calib', f'{seq}.yml')
    if not os.path.exists(calib_path):
        raise SystemExit(f"Calib yaml not found: {calib_path} (pass --calib)")
    print(f"Calib: {calib_path}")
    intrinsics, T_cami_cam0 = load_calib(calib_path)

    traj_path = f'{args.result}/traj_mcgs.txt'
    if not os.path.exists(traj_path):
        raise SystemExit(f"Trajectory not found: {traj_path}")
    traj = np.loadtxt(traj_path)
    if traj.ndim == 1:
        traj = traj[None]

    cams = discover_cameras(args.result)
    if not cams:
        raise SystemExit(
            f"No renders/depth_after_opt/cam* folders under {args.result}")
    print(f"Found {len(cams)} cameras: {', '.join(cams)}  ({len(traj)} keyframes)")

    parts = []
    for cam_name in cams:
        cam_idx = int(cam_name[3:])
        T_ci_c0 = T_cami_cam0[cam_idx] if cam_idx < len(T_cami_cam0) else np.eye(4)
        part = build_camera_part(args.result, cam_name, intrinsics[cam_idx],
                                 T_ci_c0, traj, args)
        parts.append(part)
        print(f"  {cam_name}: {len(part['rgb'])} frames")

    vbg = integrate_multi(parts, args)
    for w in args.weight:
        mesh = extract_mesh(vbg, w)
        out = f'{args.result}/tsdf_mesh_w{w:g}.ply'
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"TSDF saved to {out}")

    if args.per_camera:
        del vbg
        release_gpu_cache()
        for p in parts:
            print(f"\n[per-camera] integrating {p['name']} ({len(p['rgb'])} frames)...")
            vbg_p = integrate_multi([p], args)
            for w in args.weight:
                mesh_p = extract_mesh(vbg_p, w)
                out = f'{args.result}/tsdf_mesh_{p["name"]}_w{w:g}.ply'
                o3d.io.write_triangle_mesh(out, mesh_p)
                print(f"  saved {out}")
            del vbg_p
            release_gpu_cache()


if __name__ == '__main__':
    main()
