#!/home/wei/miniconda3/envs/mcgs/bin/python
import os    # nopep8
import sys   # nopep8
_ROOT = os.path.dirname(os.path.abspath(__file__))   # nopep8
sys.path.append(os.path.join(_ROOT, 'mcgs_slam'))   # nopep8
# CUDA extensions are built in-place by `pixi run build` (see pixi.toml).
sys.path.append(os.path.join(_ROOT, 'thirdparty/lietorch'))   # nopep8
sys.path.append(os.path.join(_ROOT, 'thirdparty/simple-knn'))   # nopep8
sys.path.append(os.path.join(_ROOT, 'thirdparty/diff-gaussian-rasterization'))   # nopep8

import cv2
import time
import torch
import numpy as np

from mcgs import Mcgs
from tqdm import tqdm
from mcgs_slam.utils import save_utils
from mcgs_slam.streams import image_stream
from mcgs_slam.options import get_args, load_configs
from rerun_logger import RerunLogger
from utils.plot_depth_map import colorize_np


def show_image(image, disp_est):
    image = image.permute(1, 2, 0).numpy()
    depth_est = np.divide(1., disp_est.numpy(), out=np.zeros(disp_est.shape, dtype=float), where=disp_est != 0)
    depth_max = min(10, np.percentile(depth_est, 90))
    depth_est = colorize_np(depth_est, range=[0.5, depth_max], append_cbar=True)
    overlay = np.concatenate((image/255., depth_est), axis=1)
    cv2.imshow('image', overlay)
    cv2.waitKey(1)


if __name__ == '__main__':
    args = get_args()
    args = load_configs(args)
    os.makedirs(args.output, exist_ok=True)

    torch.multiprocessing.set_start_method('spawn')

    rr_logger = None
    if args.rrd or args.rerun_spawn:
        # The pipeline skips input dir 1 (stereo-right duplicate) when filling
        # the video buffers, so cameras are logged in video order [0, 2, 3, ...].
        stream_indices = [0] + list(range(2, args.multi)) if args.multi else [0]
        cam_names = [os.path.basename(os.path.normpath(args.imagedir[i])) for i in stream_indices]
        rr_logger = RerunLogger(cam_names, stream_indices, scale_factor=0.2, save_path=args.rrd,
                                spawn=args.rerun_spawn, splat_every=args.rr_splat_every)

    mcgs = Mcgs(args, rr_logger=rr_logger)
    tstamps = {}
    t0 = time.time()
    N = len(os.listdir(args.imagedir[0])[::args.stride])
    pbar = tqdm(image_stream(args.imagedir, args.calib, args), total=N)
    for (t, image, intrinsics, timestamp) in pbar:
        if timestamp < args.t0:
            continue
        tstamps[t] = timestamp

        if rr_logger is not None:
            rr_logger.set_frame(t, timestamp)
            if len(tstamps) == 1:
                T_rig_cams = args.T_cami_cam0.cpu() if args.multi else torch.tensor([[0., 0, 0, 0, 0, 0, 1]])
                rr_logger.log_calibration(T_rig_cams, intrinsics, args.wd, args.ht)
            rr_logger.log_images(image)

        mcgs.track(t, timestamp, image, intrinsics=intrinsics)

        if args.vis and t == mcgs.video.tstamp[mcgs.video.counter.value-1]:
            show_image(image[0], mcgs.video.disps_up[mcgs.video.counter.value-1])

        pbar.set_description(f"Processing keyframe {mcgs.video.total_counter} {timestamp}")
        if args.early_stop > 0 and mcgs.video.total_counter >= args.early_stop:
            break

    t1 = time.time()
    print(f'Elapsed time: {(t1-t0):.2f} s')

    mcgs.terminate()
    if rr_logger is not None:
        rr_logger.send_final_blueprint()
    mcgs.video.globuf.fill_global_data()

    mcgs.save_kf_poses(args, mcgs.video)
    save_utils.save_pc(args, mcgs.video, args.output)
    mcgs.video.globuf.dump_global_buffer()

    print("done!")
