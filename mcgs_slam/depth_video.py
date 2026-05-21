import numpy as np
import torch
import droid_backends
import geom.projective_ops as pops

from torch.multiprocessing import Value

from lietorch import SE3
from droid_net import cvx_upsample
from global_buffer import GlobalBuffer
from utils.jacobi import global_relative_pose_constraints

from geom.ba import JDSA

global_iter = 0

class DepthVideo:
    def __init__(self, args, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):

        self.vis = args.vis
        buffer += 15
        self.buffer = buffer
        
        self.use_jdsa = args.jdsa

        self.stereo = stereo
        c = 1 if not self.stereo else 2
        self.multi = args.multi if args.multi > 2 else False
        c = self.multi if self.multi else c  # front left/right, right, left

        # current keyframe count
        self.counter = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.base = torch.as_tensor(args.base, dtype=torch.float, device="cuda")
        self.kf_stamps = {}

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht//8, wd//8, device="cpu", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_list = [self.disps]
        self.intrinsics = torch.zeros(buffer, c, 8, device="cuda", dtype=torch.float).share_memory_()
        
        # TODO
        self.images_up = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.normals = torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.disps_prior_up = torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_()
        self.dscales = torch.ones(buffer, 2, 2, device='cuda', dtype=torch.float).share_memory_()
        
        self.ii = torch.zeros(int(1e4), 1, dtype=torch.long, device='cuda').share_memory_()
        self.jj = torch.zeros(int(1e4), 1, dtype=torch.long, device='cuda').share_memory_()
        self.num_factors = Value('i', 0)

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, c-1, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, c-1, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        ### Mutil Cameras ###
        self.T_ci_c0 = [SE3(torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")[None,None])]
        if self.multi:
            self.images_list = [self.images] + [torch.zeros(buffer, 3, ht//8, wd//8, device="cpu", dtype=torch.uint8) for _ in range(self.multi-2)]
            self.disps_list = [self.disps] + [torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            self.T_ci_c0 = [SE3(T[None,None]) for T in args.T_cami_cam0]
            
            # TODO
            self.disps_up_list = [self.disps_up] + [torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            self.images_up_list = [self.images_up] + [torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.uint8).share_memory_() for _ in range(self.multi-2)]
            self.normals_list = [self.normals] + [torch.zeros(buffer, 3, ht, wd, device="cpu", dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            self.disps_prior_up_list = [self.disps_prior_up] + [torch.zeros(buffer, ht, wd, device="cpu", dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            self.disps_sens_list = [self.disps_sens] + [torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            self.dscales_list = [self.dscales] + [torch.ones(buffer, 2, 2, device='cuda', dtype=torch.float).share_memory_() for _ in range(self.multi-2)]
            
        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")

        # global buffer to store history state
        self.globuf = GlobalBuffer(self, args, c)

    @property
    def total_counter(self):
        return self.counter.value + self.globuf.offset.value

    def release_buffer(self, window):
        with self.get_lock():
            i = self.counter.value - window
            self.globuf.fill_global_data(i)
            self.dirty[:] = False

            for ii in range(window+1):
                self.tstamp[ii] = self.tstamp[i+ii]
                self.poses[ii] = self.poses[i+ii]
                self.images[ii] = self.images[i+ii]
                self.disps[ii] = self.disps[i+ii]
                self.intrinsics[ii] = self.intrinsics[i+ii]
                self.fmaps[ii] = self.fmaps[i+ii]
                self.nets[ii] = self.nets[i+ii]
                self.inps[ii] = self.inps[i+ii]
                for ic in range(1, self.multi-1):
                    self.images_list[ic][ii] = self.images_list[ic][i+ii]
                    self.disps_list[ic][ii] = self.disps_list[ic][i+ii]
                    
                    # TODO release
                    self.disps_up_list[ic][ii] = self.disps_up_list[ic][i+ii]
                    self.normals_list[ic][ii] = self.normals_list[ic][i+ii]   
                    self.images_up_list[ic][ii] = self.images_up_list[ic][i+ii]
                    self.disps_prior_up_list[ic][ii] = self.disps_prior_up_list[ic][i+ii]
                    self.disps_sens_list[ic][ii] = self.disps_sens_list[ic][i+ii]
                    self.dscales_list[ic][ii] = self.dscales_list[ic][i+ii]
                    
                self.images_up[ii] = self.images_up[i+ii]
                self.disps_up[ii] = self.disps_up[i+ii]
                self.normals[ii] = self.normals[i+ii]
                self.disps_prior_up[ii] = self.disps_prior_up[i+ii]
                self.disps_sens[ii] = self.disps_sens[i+ii]
                self.dscales[ii] = self.dscales[i+ii]
                
                if ii < window:
                    self.kf_stamps[ii] = self.kf_stamps[i+ii]

    def rm_keyframe(self, ix):
        """ drop edges from factor graph """
        with self.get_lock():
            self.tstamp[ix] = self.tstamp[ix+1]
            self.images[ix] = self.images[ix+1]
            self.poses[ix] = self.poses[ix+1]
            self.disps[ix] = self.disps[ix+1]
            for ic in range(1, self.multi-1):
                self.images_list[ic][ix] = self.images_list[ic][ix+1]
                self.disps_list[ic][ix] = self.disps_list[ic][ix+1]
                
                # TODO
                self.disps_up_list[ic][ix] = self.disps_up_list[ic][ix+1]
                self.normals_list[ic][ix] = self.normals_list[ic][ix+1]
                self.images_up_list[ic][ix] = self.images_up_list[ic][ix+1]
                self.disps_prior_up_list[ic][ix] = self.disps_prior_up_list[ic][ix+1]
                self.disps_sens_list[ic][ix] = self.disps_sens_list[ic][ix+1]
                self.dscales_list[ic][ix] = self.dscales_list[ic][ix+1]

            self.images_up[ix] = self.images_up[ix+1]
            self.disps_up[ix] = self.disps_up[ix+1]
            self.normals[ix] = self.normals[ix+1]
            self.disps_prior_up[ix] = self.disps_prior_up[ix+1]
            self.disps_sens[ix] = self.disps_sens[ix+1]
            self.dscales[ix] = self.dscales[ix+1]
            
            self.intrinsics[ix] = self.intrinsics[ix+1]

            self.nets[ix] = self.nets[ix+1]
            self.inps[ix] = self.inps[ix+1]
            self.fmaps[ix] = self.fmaps[ix+1]

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        if item[1] is not None:
            self.kf_stamps[index] = item[1]

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        if item[2] is not None:
            self.images[index] = item[2][0][:,3::8,3::8].cpu()
            self.images_up[index] = item[2][0].cpu()
            for ic in range(1, self.multi-1):
                self.images_list[ic][index] = item[2][ic][:,3::8,3::8].cpu()
                self.images_up_list[ic][index] = item[2][ic].cpu()

        if item[3] is not None:
            self.poses[index] = item[3]

        if item[4] is not None:
            self.disps[index] = item[4]
            
        if item[5] is not None:
            depth = item[5][0]
            self.disps_prior_up[index] = torch.where(depth>0, 1.0/depth, 0).cpu()
            depth = item[5][0][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, 0).cuda()
            for ic in range(1, self.multi-1):
                depth = item[5][ic]
                self.disps_prior_up_list[ic][index] = torch.where(depth>0, 1.0/depth, 0).cpu()
                depth = item[5][ic][3::8,3::8]
                self.disps_sens_list[ic][index] = torch.where(depth>0, 1.0/depth, 0).cuda()
        
        if item[6] is not None:
            self.normals[index] = item[6][0]
            for ic in range(1, self.multi-1):
                self.normals_list[ic][index] = item[6][ic]

        if item[7] is not None:
            self.intrinsics[index] = item[7]
        else:
            self.intrinsics[index] = self.intrinsics[0].clone()

        if len(item) > 8:
            self.fmaps[index] = item[8]

        if len(item) > 9:
            self.nets[index] = item[9]

        if len(item) > 10:
            self.inps[index] = item[10]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask, index):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps_list[index][ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze().cpu()
    
    def upsample_list(self, ix, mask, index):
        """ upsample disparity """
        
        disps_up = cvx_upsample(self.disps_list[index][ix].unsqueeze(-1), mask)
        self.disps_up_list[index][ix] = disps_up.squeeze().cpu()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value, :3] *= s
            self.dirty[:self.counter.value] = True

    def reproject(self, ii, jj, index=0):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = SE3(self.poses[None])  # T_c0_w

        if index > 0:
            coords, valid_mask = \
                pops.projective_transform(Gs, self.disps_list[index][None], self.intrinsics[None,:,[index+1]], self.base, ii, jj, Tcb=self.T_ci_c0[index])
        else:
            coords, valid_mask = \
                pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], self.base, ii, jj)

        return coords, valid_mask

    def compute_flow_distance(self, ii, jj):
        """ compute flow magnitude between all pairs of frames """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = SE3(self.poses[None])  # T_c0_w

        N = ii.shape[0]
        MAX_FLOW = 100.0
        dists = torch.zeros(N, device='cuda')

        s = 2048
        for i in range(0, ii.shape[0], s):
            flow1, val1 = pops.induced_flow(Gs, self.disps[None], self.intrinsics[None], self.base, ii[i:i+s], jj[i:i+s])
            flow2, val2 = pops.induced_flow(Gs, self.disps[None], self.intrinsics[None], self.base, jj[i:i+s], ii[i:i+s])

            flow = torch.stack([flow1, flow2], dim=2)
            val = torch.stack([val1, val2], dim=2)

            mag = flow.norm(dim=-1).clamp(max=MAX_FLOW)
            mag = mag.view(mag.shape[1], -1)
            val = val.view(val.shape[1], -1)

            mag = (mag * val).mean(-1) / val.mean(-1)
            mag[val.mean(-1) < 0.7] = np.inf

            dists[i:i+s] = mag

        return dists

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing='ij')

        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0,0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0,0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0,0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def py_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            from geom.ba import BA
            poses = SE3(self.poses[:t1][None])
            disps = self.disps[:t1][None]
            for _ in range(itrs):
                poses, disps = BA(target, weight, eta, poses, disps, self.intrinsics[None], self.base, ii, jj, fixedp=t0)
            self.poses[:t1] = poses.data[0]
            self.disps[:t1] = disps[0]
            self.disps.clamp_(min=0.001)

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, use_scaling=False):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            for _ in range(itrs):
                droid_backends.ba(self.poses, self.disps, self.intrinsics[0,:2], self.base, self.disps_sens,
                                  target, weight, eta, ii, jj, t0, t1, 1, lm, ep)
            
            if self.use_jdsa and use_scaling:
                print("=====================JDSA=====================")
                poses = SE3(self.poses[:t1][None])
                disps = self.disps[:t1][None]
                dscales = self.dscales[:t1]
                disps, dscales, _ = JDSA(target, weight, eta, poses, disps, self.intrinsics[None, :, [0], :].contiguous().squeeze(2),
                                         self.disps_sens, dscales, ii, jj, 0.001)
                self.disps[:t1] = disps[0]
                self.dscales[:t1] = dscales

            self.disps.clamp_(min=0.001, max=10)

    def multi_cam_ba(self, t0, targets, weights, etas, iis, jjs, lm=1e-5, ep=1e-2, use_scaling=False):
        """ multi cam dense bundle adjustment (DBA) """
        verbose = False
        with self.get_lock():

            t1 = max(iis[0].max().item(), jjs[0].max().item()) + 1

            if verbose:
                print("="*50, "Multi CAM BA from frame '{}' to '{}' size {}".format(t0, t1, iis[0].shape[0]))

            for _ in range(2):
                poses_cw = SE3(self.poses[:t1][None])
                Gijs, Gicjs = [], []
                for ii, jj, T_ci_c0 in zip(iis, jjs, self.T_ci_c0):
                    Gicj = T_ci_c0 * poses_cw[:,jj] * poses_cw[:,ii].inv()
                    Gij = Gicj * T_ci_c0.inv()
                    Gij.data[:, ii==jj] = self.base
                    Gijs.append(Gij.data[0])
                    Gicjs.append(Gicj.data[0])

                Ks = [self.intrinsics[0,:2]] + [self.intrinsics[0,[ic]] for ic in range(2, self.multi)]
                T_ci_c0 = [T.data[0,0] for T in self.T_ci_c0]
                droid_backends.multi_cam_ba(self.poses, Ks, self.disps_list, self.disps_sens_list, Gijs, Gicjs, T_ci_c0,
                                            targets, weights, etas, iis, jjs, t0, t1, 6, lm, ep)
            
            if self.use_jdsa and use_scaling:
                print("=====================JDSA=====================")      
                for i, disps in enumerate(self.disps_list):
                    poses = self.T_ci_c0[i] * SE3(self.poses[:t1][None])
                    disps = self.disps_list[i][:t1][None]
                    dscales = self.dscales_list[i][:t1]
                    
                    idx = i if i == 0 else i + 1
                    intrinsics_i = self.intrinsics[None, :, [idx], :].contiguous().squeeze(2)
                    
                    disps, dscales, _ = JDSA(targets[i], weights[i], etas[i], poses, disps, intrinsics_i, 
                                                self.disps_sens_list[i], dscales, ii, jj, 0.001)
                        
                    self.disps_list[i][:t1] = disps[0]
                    self.dscales_list[i][:t1] = dscales
                    
            for disps in self.disps_list:
                disps.clamp_(min=0.001, max=10)

    def global_pose_ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1):
        """ dense bundle adjustment (DBA) """
        verbose = True

        with self.get_lock():

            if verbose:
                print("="*100)
                print("Global pose graph BA from frame '{}' to '{}' size {} and {} rel poses".format(t0, t1, ii.shape[0], self.globuf.rel_N))

            poses_cw = SE3(self.poses[:t1][None])
            for _ in range(itrs):
                # reprojection chi2 error
                coords, valid = pops.projective_transform(poses_cw, self.disps[None], self.intrinsics[None], self.base, ii, jj)
                r = (target[None] - coords.permute(0,1,4,2,3).contiguous())
                r = r.view(1, ii.shape[0], -1, 1)
                rw = .001 * (valid.permute(0,1,4,2,3) * weight[None]).view(1, ii.shape[0], -1, 1)
                rchi2 = torch.sum((rw * r).transpose(2,3) @ r)

                # rel pose constraints
                iip, jjp = self.globuf.rel_ii[:self.globuf.rel_N].cuda(), self.globuf.rel_jj[:self.globuf.rel_N].cuda()
                mask = (iip != jjp)
                iip, jjp = iip[mask], jjp[mask]
                rel_poses = self.globuf.rel_poses[:self.globuf.rel_N][mask.cpu()].cuda()[None]
                if self.multi:
                    rel_cam_index = self.globuf.rel_cam_index[:self.globuf.rel_N][mask.cpu()]
                    for index in rel_cam_index.unique():
                        if index == 0:
                            continue
                        cami = rel_cam_index.cuda() == index
                        rel_poses[:,cami] = (self.T_ci_c0[index].inv() * SE3(rel_poses[:,cami]) * self.T_ci_c0[index]).data
                infos = 1 / self.globuf.rel_covs[:self.globuf.rel_N][mask.cpu()].cuda()
                infos = infos.unsqueeze(2).expand(*infos.size(), 6) * torch.eye(6, device='cuda')[None]
                infos[torch.isnan(infos) | torch.isinf(infos)] = 0.
                Hsp, vsp, pchi2, pchi2_scaled = global_relative_pose_constraints(iip, jjp, poses_cw.data, rel_poses, infos, pw=1e-3)
                Hsp, vsp = Hsp.squeeze(1), vsp.squeeze(1)
                B, N = Hsp.shape[:2]
                Hsp = torch.cat((Hsp, torch.zeros((4,N,9,6), device='cuda')), dim=2)
                Hsp = torch.cat((Hsp, torch.zeros((4,N,15,9), device='cuda')), dim=3)
                vsp = torch.cat((vsp, torch.zeros((2,N,9), device='cuda')), dim=2)

                if verbose:
                    print("- Chi2 error reproj: {:.5f} relpose: {:.5f} {:.5f}".format(rchi2.item(), pchi2.item(), pchi2_scaled.item()))

                Gibj = self.T_ci_c0[0] * poses_cw[:,jj] * poses_cw[:,ii].inv()
                Gij = Gibj * self.T_ci_c0[0].inv()
                Gij.data[:, ii==jj] = self.base
                droid_backends.global_pose_ba(poses_cw.data[0], self.disps, self.disps_sens, self.intrinsics[0,:2],
                                              Gij.data[0], Gibj.data[0], self.T_ci_c0[0].data[0,0],
                                              Hsp, vsp, target, weight, eta, ii, jj, iip, jjp, t0, t1, 1, lm, ep)

            self.poses[:t1] = poses_cw.data[0]
            self.disps.clamp_(min=0.001, max=10)
