import cv2
import torch
import lietorch
import torch.nn.functional as F

import geom.projective_ops as pops
from modules.corr import CorrBlock

from prior import MoGePrior, PriorPrediction


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video, thresh=2.5, args=None, device="cuda:0"):
        
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.args = args
        self.device = device

        self.count = 0
        
        self.prior = MoGePrior(args.prior_encoder, args.prior_level, device=self.device)

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        # operator at 1/8 resolution
        ht = video.ht // 8
        wd = video.wd // 8
        self.coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        x = self.cnet(image)
        net, inp = x.split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)
    
    def _append(
        self,
        t: int,
        tstamp: float,
        image_rgb: torch.Tensor,
        pose: torch.Tensor | None,
        disparity: float | None,
        measurement_depth: torch.Tensor | None,
        intrinsics: torch.Tensor,
        gmap: torch.Tensor,
        net: torch.Tensor,
        inp: torch.Tensor,
    ) -> None:
        """Append one accepted keyframe with the depth mode selected by the CLI."""
        pred: PriorPrediction = self.prior(image_rgb, intrinsics[:, 0])
        if self.args.rgbd:
            depth: torch.Tensor | None = measurement_depth
        elif self.args.prgbd:
            depth = pred.depth
        else:
            depth = None

        intrinsics[:, :4] /= 8.0
        self.video.append(
            t, tstamp, image_rgb, pose, disparity, depth, pred.normal,
            intrinsics, gmap, net, inp,
        )

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, t, tstamp, image, intrinsics, measurement_depth=None):
        """ main update operation - run on every frame in video """
        Id = lietorch.SE3.Identity(1,).data.squeeze()

        # normalize images
        image_rgb = image[:, [2, 1, 0]]
        inputs = image_rgb[None].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs)
            self.net, self.inp, self.fmap = net, inp, gmap
            self._append(t, tstamp, image_rgb, Id, 1.0, measurement_depth, intrinsics, gmap, net, inp)

        ### only add new frame if there is enough motion ###
        else:                
            # index correlation volume
            corr = CorrBlock(self.fmap[None,[0]], gmap[None,[0]])(self.coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None,[0]], self.inp[None,[0]], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh or (tstamp - self.video.kf_stamps[self.video.counter.value-1]) > 3:
                self.count = 0
                net, inp = self.__context_encoder(inputs)
                self.net, self.inp, self.fmap = net, inp, gmap
                self._append(t, tstamp, image_rgb, None, None, measurement_depth, intrinsics, gmap, net, inp)

            else:
                self.count += 1
