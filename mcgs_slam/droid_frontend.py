import torch

from factor_graph import FactorGraph
from collections import defaultdict


def disparity_scale(estimated: torch.Tensor, sensor: torch.Tensor) -> torch.Tensor:
    """Return the shared-support median disparity ratio."""
    valid = (
        torch.isfinite(estimated)
        & torch.isfinite(sensor)
        & (estimated > 0.0)
        & (sensor > 0.0)
    )
    estimated_median = estimated[valid].median()
    sensor_median = sensor[valid].median()
    return torch.where(
        torch.any(valid),
        estimated_median / sensor_median,
        estimated.new_tensor(1.0),
    )


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.corr_impl = "volume"
        self.graph = FactorGraph(video, net.update, corr_impl=self.corr_impl, max_factors=48)
        self.multi = args.multi
        self.graph_list = [self.graph] + [FactorGraph(video, net.update, corr_impl=self.corr_impl, max_factors=48, index=i) for i in range(1, self.multi)]

        # local optimization window
        self.t0 = 0
        self.t1 = 0
        self.delete_count = defaultdict(int)

        # frontent variables
        self.is_initialized = False

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius

    def release_buffer(self, window):
        self.t1 = window
        shift = self.video.counter.value - window
        self.graph.release_buffer(window, shift)
        for graph in self.graph_list[1:]:
            graph.release_buffer(window, shift)

    @torch.no_grad()
    def __graph_update(self, iters, mode, t0=None, use_scaling=False):
        if len(self.graph_list) > 1:
            for itr in range(iters):
                t0, target, weight, eta, ii, jj, mask, upmask = self.graph.update(t0=t0, tracking=('tracking' in mode), return_update=True)
                targets, weights, etas, iis, jjs, masks, upmasks = [target], [weight], [eta], [ii], [jj], [mask], [upmask]
                for graph in self.graph_list[1:]:
                    _, target2, weight2, eta2, ii2, jj2, mask2, upmask2  = graph.update(t0=t0, tracking=('tracking' in mode), return_update=True)
                    targets += [target2]
                    weights += [weight2]
                    etas += [eta2]
                    iis += [ii2]
                    jjs += [jj2]
                    masks += [mask2]
                    upmasks += [upmask2]
                if use_scaling:
                    self.video.multi_cam_ba(t0, targets, weights, etas, iis, jjs, use_scaling=itr>2)
                else:
                    self.video.multi_cam_ba(t0, targets, weights, etas, iis, jjs, use_scaling=False)
                
                # TODO update disps_up
                self.video.upsample_list(torch.unique(self.graph.ii[masks[0]]), upmasks[0], index=self.graph.index)
                for graph in self.graph_list[1:]:
                    self.video.upsample_list(torch.unique(graph.ii[masks[graph.index]]), upmasks[graph.index], index=graph.index)      
        else:
            for i in range(iters):
                if use_scaling:
                    self.graph.update(None, None, tracking=('tracking' in mode), use_scaling=i>2)
                else:
                    self.graph.update(None, None, tracking=('tracking' in mode), use_scaling=False)

    def __update(self):
        """ add edges, perform update """

        self.t1 += 1

        if self.graph.corr is not None or self.corr_impl == "alt":
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(self.t1-5, max(self.t1-self.frontend_window, 0),
            rad=self.frontend_radius, nms=self.frontend_nms, thresh=self.frontend_thresh, beta=self.beta, remove=True)
        for graph in self.graph_list[1:]:
            graph.update_factors(self.graph.ii, self.graph.jj)

        # TODO JDSA
        self.video.dscales[self.t1-1] = disparity_scale(
            self.video.disps[self.t1-1], self.video.disps_sens[self.t1-1]
        )
        
        for idx in range(1, self.multi):
            self.video.dscales_list[idx][self.t1-1] = disparity_scale(
                self.video.disps_list[idx][self.t1-1],
                self.video.disps_sens_list[idx][self.t1-1],
            )

        self.__graph_update(self.iters1, mode="vision_only_tracking", use_scaling=True)

        # set initial pose for next frame
        d = self.video.distance([self.t1-3], [self.t1-2], beta=self.beta, bidirectional=True)

        # print(self.t1-2, self.video.counter.value, self.video.total_counter)
        if d.item() < self.keyframe_thresh and self.delete_count[self.video.total_counter-2] == 0:
            self.graph.rm_keyframe(self.t1 - 2)
            for graph in self.graph_list[1:]:
                graph.rm_keyframe(self.t1 - 2)
            self.delete_count[self.video.total_counter-2] += 1

            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1
            
            update_idx = []
            
        else:
            self.__graph_update(self.iters2, mode="vision_only")
            
            update_idx = torch.arange(self.graph.ii.min(), self.t1-1, device='cuda')

        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1-1]
        self.video.disps[self.t1] = self.video.disps[self.t1-1].mean()
        for i in range(1, self.multi):
            self.video.disps_list[i][self.t1] = self.video.disps_list[i][self.t1-1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min():self.t1] = True
        self.video.num_factors.value = self.graph.ii.shape[0]
        self.video.ii[:self.graph.ii.shape[0], 0] = self.graph.ii
        self.video.jj[:self.graph.ii.shape[0], 0] = self.graph.jj
        
        return update_idx

    def __initialize(self):
        """ initialize the SLAM system """
        print("Initializing...")
        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)
        for graph in self.graph_list[1:]:
            graph.add_factors(self.graph.ii, self.graph.jj)

        self.__graph_update(8, mode="vision_only", t0=1)

        self.graph.add_proximity_factors(0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False)
        for graph in self.graph_list[1:]:
            graph.add_factors(self.graph.ii, self.graph.jj)

        # TODO JDSA
        for i in range(self.t1):
            self.video.dscales[i] = disparity_scale(
                self.video.disps[i], self.video.disps_sens[i]
            )
        
        for idx in range(1, self.multi):
            for i in range(self.t1):
                self.video.dscales_list[idx][i] = disparity_scale(
                    self.video.disps_list[idx][i], self.video.disps_sens_list[idx][i]
                )

        self.__graph_update(8, mode="vision_only", t0=1, use_scaling=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1-1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1-4:self.t1].mean()
        for i in range(1, self.multi):
            self.video.disps_list[i][self.t1] = self.video.disps_list[i][self.t1-4:self.t1].mean()

        # initialization complete
        self.is_initialized = True

        with self.video.get_lock():
            self.video.dirty[:self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup-4, store=True)
        
        return torch.arange(self.t1-1, device='cuda')

    def __del__(self):
        if self.graph.ii.shape[0] > 0:
            self.graph.rm_factors(self.graph.ii > 0, store=True)
        for graph in self.graph_list[1:]:
            if graph.ii.shape[0] > 0:
                graph.rm_factors(graph.ii > 0, store=True)

    def __call__(self):
        """ main update """

        self.to_update = []

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.to_update = self.__initialize()
            
        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.to_update = self.__update()
            
        return self.to_update
