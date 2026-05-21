import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph


class DroidBackend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update

        # global optimization window
        self.t0 = 0
        self.t1 = 0

        self.beta = args.beta
        self.backend_thresh = args.backend_thresh
        self.backend_radius = args.backend_radius
        self.backend_nms = args.backend_nms
        self.need_normalize = not self.video.stereo and not args.posefile and not "depthdir" in args.__dict__
        
    @torch.no_grad()
    def __call__(self, steps=12):
        """ main update """

        t = self.video.counter.value
        if self.need_normalize:
             self.video.normalize()

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=5000)

        graph.add_proximity_factors(rad=self.backend_radius,
                                    nms=self.backend_nms,
                                    thresh=self.backend_thresh,
                                    beta=self.beta)

        graph.update_lowmem(steps=steps)
        graph.clear_edges()
        self.video.dirty[:t] = True

    @torch.no_grad()
    def pose_ba(self, iter, steps=12):
        """ main update """

        t = self.video.total_counter

        graph = FactorGraph(self.video, self.update_op, corr_impl="alt", max_factors=500)

        graph.add_proximity_factors_backend(t=t,
                                    rad=self.backend_radius,
                                    nms=self.backend_nms,
                                    thresh=self.backend_thresh,
                                    beta=self.beta,
                                    framedist=100,
                                    relset=self.video.globuf.rel_set)
        if graph.ii.shape[0] < 1:
            return

        graph.update_lowmem(t0=1, t1=t, steps=steps, pose_graph=True)

        self.video.globuf.add_rel_poses(graph.ii, graph.jj, graph.target, graph.weight, iter=iter)
