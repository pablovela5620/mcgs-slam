"""Interactive visualizer for MCGS-SLAM TSDF meshes: show all cameras together,
press 1..N to isolate one camera.

Consumes a single MCGS-SLAM output folder (the same one passed to
`tsdf_integrate.py`):

    {result}/tsdf_mesh_w*.ply            # fused mesh (all cameras)
    {result}/tsdf_mesh_cam{i}_w*.ply     # per-camera mesh (tsdf_integrate --per_camera)
    {result}/traj_mcgs.txt               # cam0 keyframe poses (c2w)
    {result}/renders/depth_after_opt/cam{i}/   # per-camera render folders

Per-camera trajectories are reconstructed from traj_mcgs.txt and the camera rig
extrinsics (T_cami_cam0) in calib/<seq>.yml.

Keys:
    A or 0  -> show ALL (fused mesh + every camera trajectory)
    1..9    -> show only that camera's mesh + trajectory
    H       -> print the legend again
"""

import os
import argparse
import numpy as np
import open3d as o3d
import yaml
from glob import glob
from scipy.spatial.transform import Rotation as R


CAM_COLORS = [
    [0.0, 0.4, 1.0],   # 1 blue
    [1.0, 0.2, 0.2],   # 2 red
    [0.2, 0.8, 0.2],   # 3 green
    [1.0, 0.6, 0.0],   # 4 orange
    [0.6, 0.2, 0.8],   # 5 purple
    [0.0, 0.8, 0.8],   # 6 cyan
    [0.9, 0.9, 0.0],   # 7 yellow
    [0.5, 0.5, 0.5],   # 8 gray
    [0.9, 0.4, 0.7],   # 9 pink
]


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
    """Return the list of 4x4 T_cami_cam0 rig extrinsics from calib/<seq>.yml."""
    with open(calib_path, 'r') as f:
        params = yaml.safe_load(f)
    raw = params.get('T_cami_cam0', [[0, 0, 0, 0, 0, 0, 1]])
    return [pose7_to_matrix(np.asarray(t, dtype=np.float64)) for t in raw]


def discover_cameras(result):
    """Return sorted camera folder names ('cam0', 'cam1', ...) under renders/."""
    cams = sorted(glob(f'{result}/renders/depth_after_opt/cam*'),
                  key=lambda d: int(os.path.basename(d)[3:]))
    return [os.path.basename(c) for c in cams]


def create_frustum_lineset(c2w, color, size=0.06):
    points = np.array([
        [0.0,  0.0, 0.0],
        [ 1.0, -0.5, 2.0],
        [-1.0, -0.5, 2.0],
        [ 1.0,  0.5, 2.0],
        [-1.0,  0.5, 2.0],
    ]) * size
    pts_h = np.hstack([points, np.ones((5, 1))])
    pts_world = (c2w @ pts_h.T)[:3].T
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts_world)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


def create_traj_polyline(c2w_list, color):
    if len(c2w_list) < 2:
        return None
    pts = np.array([T[:3, 3] for T in c2w_list])
    lines = [[i, i + 1] for i in range(len(pts) - 1)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return ls


NORMAL_PERMS = ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']
_AXIS_IDX = {'x': 0, 'y': 1, 'z': 2}


def paint_mesh_by_normals(mesh, perm='xyz'):
    normals = np.asarray(mesh.vertex_normals)
    colors = (normals + 1.0) * 0.5
    idx = [_AXIS_IDX[c] for c in perm]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, idx])


def load_mesh(path, gray=False, normal=None):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    if gray:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
        mesh.paint_uniform_color([0.75, 0.75, 0.75])
    elif normal is not None:
        paint_mesh_by_normals(mesh, perm=normal)
    return mesh


def camera_trajectory(traj, T_ci_c0):
    """Reconstruct a camera's c2w poses from the cam0 trajectory and rig extrinsic."""
    c2w_list = []
    for k in range(len(traj)):
        T_c2w0 = to_se3_matrix(traj[k])              # cam0 camera-to-world
        c2w_list.append(T_c2w0 @ np.linalg.inv(T_ci_c0))
    return c2w_list


def build_camera_geoms(c2w_list, color, stride, size):
    """Return a list of LineSets (frustums every `stride` + a trajectory polyline)."""
    geoms = []
    for k, c2w in enumerate(c2w_list):
        if k % stride == 0:
            geoms.append(create_frustum_lineset(c2w, color, size=size))
    poly = create_traj_polyline(c2w_list, color)
    if poly is not None:
        geoms.append(poly)
    return geoms


def print_legend(cameras):
    print("\n" + "=" * 60)
    print("Visualizer keys:")
    print("  A or 0  -> show ALL (fused mesh + every camera)")
    for i, (name, color, has_mesh) in enumerate(cameras, start=1):
        tag = "" if has_mesh else "  (no per-camera mesh, trajectory only)"
        print(f"  {i}       -> {name}  color={color}{tag}")
    print("  H       -> print this legend")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive MCGS-SLAM TSDF viewer with per-camera isolation.')
    parser.add_argument('--result', type=str, required=True,
                        help='Path to the MCGS-SLAM output folder (e.g. output/100613)')
    parser.add_argument('--calib', type=str, default=None,
                        help='Path to calib yaml. Default: calib/<seq>.yml.')
    parser.add_argument('--mesh', type=str, default=None,
                        help='Fused mesh PLY path (auto-detected if omitted)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Show every N-th camera frustum')
    parser.add_argument('--size', type=float, default=0.08, help='Frustum size')
    parser.add_argument('--gray', action='store_true',
                        help='Render meshes in uniform gray (drop vertex RGB colors)')
    parser.add_argument('--normal', nargs='?', const='xyz', default=None,
                        choices=NORMAL_PERMS,
                        help='Color meshes by surface normals using the given axis '
                             'permutation: xyz/xzy/yxz/yzx/zxy/zyx. Bare --normal = xyz.')
    args = parser.parse_args()

    seq = os.path.basename(os.path.normpath(args.result))
    calib_path = args.calib or os.path.join('calib', f'{seq}.yml')
    if not os.path.exists(calib_path):
        raise SystemExit(f"Calib yaml not found: {calib_path} (pass --calib)")
    T_cami_cam0 = load_calib(calib_path)

    # Locate fused mesh.
    mesh_path = args.mesh
    if mesh_path is None:
        cands = sorted(glob(f'{args.result}/tsdf_mesh_w*.ply'))
        if not cands:
            raise SystemExit(f"No fused tsdf_mesh_w*.ply under {args.result}. "
                             f"Run tsdf_integrate.py first.")
        mesh_path = cands[0]
    print(f"Fused mesh: {mesh_path}")
    global_mesh = load_mesh(mesh_path, gray=args.gray, normal=args.normal)

    # Trajectory + camera folders.
    traj_path = f'{args.result}/traj_mcgs.txt'
    if not os.path.exists(traj_path):
        raise SystemExit(f"Trajectory not found: {traj_path}")
    traj = np.loadtxt(traj_path)
    if traj.ndim == 1:
        traj = traj[None]

    cam_names = discover_cameras(args.result)
    if not cam_names:
        raise SystemExit(
            f"No renders/depth_after_opt/cam* folders under {args.result}")
    print(f"Found {len(cam_names)} cameras: {', '.join(cam_names)}")

    # Build per-camera geometry bundles.
    cameras = []          # list of (name, color, has_mesh)
    per_cam_geoms = []     # list of (mesh_or_None, [linesets,...])
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)

    for i, cam_name in enumerate(cam_names):
        cam_idx = int(cam_name[3:])
        color = CAM_COLORS[i % len(CAM_COLORS)]
        T_ci_c0 = T_cami_cam0[cam_idx] if cam_idx < len(T_cami_cam0) else np.eye(4)
        c2w_list = camera_trajectory(traj, T_ci_c0)
        geoms = build_camera_geoms(c2w_list, color, stride=args.stride, size=args.size)

        mesh = None
        mcands = sorted(glob(f'{args.result}/tsdf_mesh_{cam_name}_w*.ply'))
        if mcands:
            mesh = load_mesh(mcands[0], gray=args.gray, normal=args.normal)

        per_cam_geoms.append((mesh, geoms))
        cameras.append((cam_name, color, mesh is not None))
        tag = "with per-camera mesh" if mesh is not None else "no per-camera mesh"
        print(f"  [{i + 1}] {cam_name}: {len(geoms)} linesets, {tag}")

    print_legend(cameras)

    # Open3D doesn't show/hide individual geometries cleanly; instead we clear
    # and re-add the bundle that matches the active mode each time a key fires.
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Per-camera TSDF viewer", width=1920, height=1080)

    state = {'mode': 'all', 'first': True}

    def show_all(v):
        v.clear_geometries()
        v.add_geometry(global_mesh, reset_bounding_box=state['first'])
        for _, geoms in per_cam_geoms:
            for g in geoms:
                v.add_geometry(g, reset_bounding_box=False)
        v.add_geometry(coord, reset_bounding_box=False)
        state['first'] = False
        state['mode'] = 'all'
        print("[mode] ALL")
        return False

    def show_only(idx):
        def cb(v):
            if idx >= len(per_cam_geoms):
                print(f"[mode] no camera {idx + 1}")
                return False
            mesh, geoms = per_cam_geoms[idx]
            v.clear_geometries()
            if mesh is not None:
                v.add_geometry(mesh, reset_bounding_box=state['first'])
            for g in geoms:
                v.add_geometry(g, reset_bounding_box=(mesh is None and state['first']))
            v.add_geometry(coord, reset_bounding_box=False)
            state['first'] = False
            state['mode'] = f'cam{idx}'
            label = "(no per-camera mesh)" if mesh is None else ""
            print(f"[mode] camera {idx + 1} = {cameras[idx][0]} {label}")
            return False
        return cb

    def help_cb(_v):
        print_legend(cameras)
        return False

    vis.register_key_callback(ord('A'), show_all)
    vis.register_key_callback(ord('0'), show_all)
    vis.register_key_callback(ord('H'), help_cb)
    for i in range(len(per_cam_geoms)):
        vis.register_key_callback(ord(str(i + 1)), show_only(i))

    show_all(vis)  # initial scene
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
