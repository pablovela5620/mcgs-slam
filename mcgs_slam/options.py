import os
import cv2
import yaml
import argparse
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', default=0, type=int)

    parser.add_argument("--imagedir", type=str, help="path to color image directory", nargs='+')
    parser.add_argument("--posefile", type=str, help="path to pose file")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
    parser.add_argument("--stereo", default=True, type=bool, help="stereo")
    parser.add_argument("--output", default="output", type=str, help="output file")
    parser.add_argument("--early_stop", default=-1, type=int, help="stoping frame")

    parser.add_argument("--weights", default=os.path.dirname(__file__) + "/../pretrained_models/droid.pth")
    parser.add_argument("--config", default=os.path.dirname(__file__) + "/../config/config.yaml")
    parser.add_argument("--buffer", type=int, default=200)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--jdsa", action="store_true")
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--prgbd", action="store_true", help="pseudo rgbd")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--rrd", type=str, default=None, help="save a Rerun recording (.rrd) to this path")
    parser.add_argument("--rerun-spawn", action="store_true", help="spawn a live Rerun viewer")
    parser.add_argument("--rr-splat-every", type=int, default=10, help="log a Gaussian map snapshot every N keyframe updates")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=20.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    args.multi = len(args.imagedir) if len(args.imagedir) > 2 else False
    return args


def load_configs(args):
    params = yaml.load(open(args.calib, 'r'), Loader=yaml.FullLoader)
    args.calib = np.array(params['intrinsic'])
    args.camera = params['camera']
    print("Camera model:", args.camera)

    args.base = np.array(params['baseline'])
    if args.multi:
        args.T_cami_cam0 = torch.as_tensor(params['T_cami_cam0'], dtype=torch.float, device="cuda")

    args.timescale = float(params['timescale'])

    if 'ht' in params and 'wd' in params:
        args.ht, args.wd = params['ht'], params['wd']
    else:
        limit = 384 * 512
        h0, w0 = cv2.imread(os.path.join(args.imagedir[0], os.listdir(args.imagedir[0])[0])).shape[:2]
        ht = int(h0 * np.sqrt((limit) / (h0 * w0)))
        wd = int(w0 * np.sqrt((limit) / (h0 * w0)))
        args.ht, args.wd = ht-ht % 8, wd-wd % 8
    args.image_size = [args.ht, args.wd]
    print(f'Input image resolution {args.ht} x {args.wd}')

    return args
