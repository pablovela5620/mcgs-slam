import os
import torch
import geom.projective_ops as pops
from torch.multiprocessing import Value
from lietorch import SE3

class GlobalBuffer:
    def __init__(self, video, args, c):
        self.video = video
        self.multi = args.multi if args.multi > 2 else False
        self.vis = args.vis
        ht, wd = args.ht, args.wd
        self.offset = Value('i', 0)

        self.kf_stamps_all = {}
        self.poses_all = torch.zeros(0, 7, device="cpu", dtype=torch.float)
        self.images_all = torch.zeros(0, 3, ht//8, wd//8, device="cpu", dtype=torch.uint8)
        self.disps_all = torch.zeros(0, ht//8, wd//8, device="cpu", dtype=torch.float)
        self.fmaps_all = torch.zeros(0, c, 128, ht//8, wd//8, dtype=torch.half, device="cpu")
        self.nets_all = torch.zeros(0, c-1, 128, ht//8, wd//8, dtype=torch.half, device="cpu")
        self.inps_all = torch.zeros(0, c-1, 128, ht//8, wd//8, dtype=torch.half, device="cpu")
        if self.vis:
            self.disps_up_all = torch.zeros(0, ht, wd, device="cpu", dtype=torch.float)
        if self.multi:
            self.images_all_list = [self.images_all] + [torch.zeros(0, 3, ht//8, wd//8, device="cpu", dtype=torch.uint8) for _ in range(self.multi-2)]
            self.disps_all_list = [self.disps_all] + [torch.zeros(0, ht//8, wd//8, device="cpu", dtype=torch.float) for _ in range(self.multi-2)]

        # pose graph buffer
        self.rel_N = 0
        self.rel_set = set()
        max_rel = 2e5
        self.rel_ii = torch.zeros(int(max_rel), dtype=torch.long, device='cpu')
        self.rel_jj = torch.zeros(int(max_rel), dtype=torch.long, device='cpu')
        self.rel_poses = torch.zeros(int(max_rel), 7, device="cpu", dtype=torch.float)
        self.rel_covs = torch.zeros(int(max_rel), 6, device="cpu", dtype=torch.float)
        self.rel_valid_percent = torch.zeros(int(max_rel), device="cpu", dtype=torch.float)
        self.rel_cam_index = torch.zeros(int(max_rel), device="cpu", dtype=torch.uint8)
        self.output = args.output

    def fill_global_data(self, N=None):
        N = self.video.counter.value if N is None else N

        print("- - revert pose data", self.offset.value, self.video.counter.value, N)
        self.poses_all = torch.cat((self.poses_all, torch.zeros(N, 7, dtype=torch.float)), dim=0)
        self.images_all = torch.cat((self.images_all, torch.zeros((N,)+self.images_all.shape[1:], dtype=torch.uint8)), dim=0)
        self.disps_all = torch.cat((self.disps_all, torch.zeros((N,)+self.disps_all.shape[1:], dtype=torch.float)), dim=0)
        self.fmaps_all = torch.cat((self.fmaps_all, torch.zeros((N,)+self.fmaps_all.shape[1:], dtype=torch.half)), dim=0)
        self.nets_all = torch.cat((self.nets_all, torch.zeros((N,)+self.nets_all.shape[1:], dtype=torch.half)), dim=0)
        self.inps_all = torch.cat((self.inps_all, torch.zeros((N,)+self.inps_all.shape[1:], dtype=torch.half)), dim=0)
        if self.vis:
            self.disps_up_all = torch.cat((self.disps_up_all, torch.zeros((N,)+self.disps_up_all.shape[1:], dtype=torch.float)), dim=0)
        if self.multi:
            for ic in range(1, self.multi-1):
                self.images_all_list[ic] = torch.cat((self.images_all_list[ic], torch.zeros((N,)+self.images_all.shape[1:], dtype=torch.uint8)), dim=0)
                self.disps_all_list[ic] = torch.cat((self.disps_all_list[ic], torch.zeros((N,)+self.disps_all.shape[1:], dtype=torch.float)), dim=0)
        for ii in range(N):
            self.kf_stamps_all[self.offset.value+ii] = self.video.kf_stamps[ii]
            self.poses_all[self.offset.value+ii] = self.video.poses[ii].clone().cpu()
            self.images_all[self.offset.value+ii] = self.video.images[ii].clone().cpu()
            self.disps_all[self.offset.value+ii] = self.video.disps[ii].clone().cpu()
            self.fmaps_all[self.offset.value+ii] = self.video.fmaps[ii].clone().cpu()
            self.nets_all[self.offset.value+ii] = self.video.nets[ii].clone().cpu()
            self.inps_all[self.offset.value+ii] = self.video.inps[ii].clone().cpu()
            if self.vis:
                self.disps_up_all[self.offset.value+ii] = self.video.disps_up[ii].clone().cpu()
            if self.multi:
                for ic in range(1, self.multi-1):
                    self.images_all_list[ic][self.offset.value+ii] = self.video.images_list[ic][ii].clone().cpu()
                    self.disps_all_list[ic][self.offset.value+ii] = self.video.disps_list[ic][ii].clone().cpu()

        self.offset.value += N
        self.video.counter.value -= N

    def dump_global_buffer(self):
        print("Dump {} rel pose factors".format(self.rel_N))
        output = os.path.join(self.output, 'rel_factors.pt')
        datatodump = {
            'rel_N': self.rel_N, 'rel_ii': self.rel_ii[: self.rel_N].clone(),
            'rel_jj': self.rel_jj[: self.rel_N].clone(),
            'rel_poses': self.rel_poses[: self.rel_N].clone(),
            'rel_covs': self.rel_covs[: self.rel_N].clone(),
            'rel_valid_percent': self.rel_valid_percent[: self.rel_N].clone(),
            'rel_cam_index': self.rel_cam_index[: self.rel_N].clone(),
            'poses': self.poses_all[: self.video.total_counter].clone(),
            'disps': self.disps_all[: self.video.total_counter].clone(),
            'images': self.images_all[: self.video.total_counter].clone(),
            'fmaps': self.fmaps_all[: self.video.total_counter].clone(),
            'nets': self.nets_all[: self.video.total_counter].clone(),
            'inps': self.inps_all[: self.video.total_counter].clone(),
            'kf_stamps': self.kf_stamps_all, 'intrinsics': self.video.intrinsics[0].expand(
                self.video.total_counter, self.video.intrinsics.shape[1],
                self.video.intrinsics.shape[2]).clone(), }
        if self.vis:
            datatodump['disps_up'] = self.disps_up_all[: self.video.total_counter].clone()
        if self.multi:
            for ic in range(1, self.multi-1):
                datatodump[f'images{ic+1}'] = self.images_all_list[ic][:self.video.total_counter].clone()
                datatodump[f'disps{ic+1}'] = self.disps_all_list[ic][:self.video.total_counter].clone()
        torch.save(datatodump, output)

    @torch.cuda.amp.autocast(enabled=False)
    def add_rel_poses(self, ii, jj, target, weight, index=0, iter=None):
        mask = ii != jj
        ii, jj, target, weight = ii[mask], jj[mask], target[:, mask], weight[:, mask]

        if iter is not None:
            import shutil
            import numpy as np
            import cv2
            from utils.plot_depth_map import colorize_np
            outpath = f'{self.output}/idebugs/iter{iter}'
            if os.path.exists(outpath):
                shutil.rmtree(outpath)
            os.makedirs(outpath, exist_ok=True)
            debugs = {}
            sets = set()
            for n in range(0, ii.shape[0]):
                i, j, w, d = ii[n].item(), jj[n].item(), weight[0, n], self.video.ds[n]
                valid = (w > 0.1).float().mean([-3,-2,-1])
                id = (i,j) if i < j else (j,i)
                if valid > 1e-2 or id in sets:
                    sets.add(id)
                    weight_x = colorize_np(w[:, :, 0].cpu().numpy(), range=[0, 1])*256
                    weight_y = colorize_np(w[:, :, 1].cpu().numpy(), range=[0, 1])*256
                    wegs = np.concatenate((weight_x, weight_y), axis=1)
                    imgs = np.concatenate((self.video.images[i].permute(1,2,0).cpu().numpy(),
                                        self.video.images[i].permute(1,2,0).cpu().numpy()), axis=1)
                    vis = np.concatenate((imgs, wegs), axis=0)
                    if id in debugs:
                        debugs[id] = [np.concatenate((debugs[id][0], vis), axis=1), valid, d]
                    else:
                        debugs[id] = [vis, valid, d]
            for k, v in debugs.items():
                i, j = k
                vis, valid, d = v
                cv2.imwrite(outpath+"/{:.1f}_{:.1f}_{:06d}-{:06d}.png".format(valid*100, d, i, j), vis)

        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        masks = valid > 1e-2
        # print("- add rel poses {}/{}".format(torch.sum(masks).item(), ii.shape[0]))
        ii, jj, target, weight = ii[masks], jj[masks], target[:, masks], weight[:, masks]

        N = ii.shape[0]
        if N < 1:
            return

        t1 = max(ii.max(), jj.max())+1
        poses = SE3(self.video.poses[:t1][None])
        rel_poses = poses[:, jj] * poses[:, ii].inv()
        if index > 0:
            rel_poses = self.video.T_ci_c0[index] * rel_poses * self.video.T_ci_c0[index].inv()
        rel_poses.data[:, ii == jj] = self.video.base

        for _ in range(4):
            if index > 0:
                coords, valid, (_, Jj, _) = pops.projective_transform(
                    None, self.video.disps_list[index][None], self.video.intrinsics[None, :, [index+1]], self.video.base, ii, jj, jacobian=True, Gij=rel_poses)
            else:
                coords, valid, (_, Jj, _) = pops.projective_transform(
                    None, self.video.disps[None], self.video.intrinsics[None], self.video.base, ii, jj, jacobian=True, Gij=rel_poses)
            r = (target - coords).view(1, N, -1, 1)
            w = .001 * (valid * weight).view(1, N, -1, 1)
            Jj = Jj.reshape(1, N, -1, 6)
            wJjT = (w * Jj).transpose(2, 3)
            Hjj = torch.matmul(wJjT, Jj) + 1e-4*torch.eye(6, device='cuda')[None, None]
            vj = torch.matmul(wJjT, r)

            Hinv = torch.linalg.inv(Hjj)
            dx = Hinv @ vj
            rel_poses = rel_poses.retr(dx.squeeze(-1))
            # print(torch.sum(torch.norm(dx, dim=2).squeeze()))

        V = Jj @ dx - r
        sig2 = (w * V).transpose(2, 3) @ V
        cov = sig2 * Hinv
        cov = torch.diagonal(cov, dim1=2, dim2=3)

        valid = (weight[0] > 0.1).float().mean([-3, -2, -1])
        self.rel_ii[self.rel_N:self.rel_N+N] = self.offset.value + ii
        self.rel_jj[self.rel_N:self.rel_N+N] = self.offset.value + jj
        self.rel_poses[self.rel_N:self.rel_N+N] = rel_poses.data[0]
        self.rel_covs[self.rel_N:self.rel_N+N] = cov[0]
        self.rel_set.update(
            [(self.offset.value+i.item(), self.offset.value+j.item()) for i, j in zip(ii, jj)])
        self.rel_valid_percent[self.rel_N:self.rel_N+N] = valid
        self.rel_cam_index[self.rel_N:self.rel_N+N] = index
        self.rel_N += N
