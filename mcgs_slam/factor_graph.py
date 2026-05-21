import torch
import lietorch
import numpy as np

import matplotlib.pyplot as plt
from lietorch import SE3
from modules.corr import CorrBlock, AltCorrBlock
import geom.projective_ops as pops


class FactorGraph:
    def __init__(self, video, update_op, device="cuda:0", corr_impl="volume", max_factors=-1, index=0):
        self.video = video
        self.update_op = update_op
        self.device = device
        self.max_factors = max_factors
        self.corr_impl = corr_impl
        self.index = index

        # operator at 1/8 resolution
        self.ht = ht = video.ht // 8
        self.wd = wd = video.wd // 8

        self.coords0 = pops.coords_grid(ht, wd, device=device)
        self.ii = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=device)
        self.age = torch.as_tensor([], dtype=torch.long, device=device)

        self.corr, self.net, self.inp = None, None, None
        self.damping = 1e-6 * torch.ones_like(self.video.disps)

        self.target = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

        # inactive factors
        self.ii_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.jj_inac = torch.as_tensor([], dtype=torch.long, device=device)
        self.eset = set()

        self.target_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)
        self.weight_inac = torch.zeros([1, 0, ht, wd, 2], device=device, dtype=torch.float)

    def __filter_repeated_edges(self, ii, jj):
        """ remove duplicate edges """

        keep = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)

        for k, (i, j) in enumerate(zip(ii, jj)):
            keep[k] = (i.item(), j.item()) not in self.eset

        return ii[keep], jj[keep]

    def print_edges(self):
        ii = self.ii.cpu().numpy()
        jj = self.jj.cpu().numpy()

        ix = np.argsort(ii)
        ii = ii[ix]
        jj = jj[ix]

        w = torch.mean(self.weight, dim=[0,2,3,4]).cpu().numpy()
        w = w[ix]
        for e in zip(ii, jj, w):
            print(e)
        print()

    def clear_edges(self):
        self.rm_factors(self.ii >= 0)
        self.net = None
        self.inp = None

    @torch.cuda.amp.autocast(enabled=True)
    def add_factors(self, ii, jj, remove=False):
        """ add edges to factor graph """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii, dtype=torch.long, device=self.device)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj, dtype=torch.long, device=self.device)

        # remove duplicate edges
        ii, jj = self.__filter_repeated_edges(ii, jj)


        if ii.shape[0] == 0:
            return

        # place limit on number of factors
        if self.max_factors > 0 and self.ii.shape[0] + ii.shape[0] > self.max_factors \
                and self.corr is not None and remove:
            
            ix = torch.arange(len(self.age))[torch.argsort(self.age).cpu()]
            self.rm_factors(ix >= self.max_factors - ii.shape[0], store=True)

        net = self.video.nets[ii.cpu(), self.index].to(self.device).unsqueeze(0)

        # correlation volume for new edges
        if self.corr_impl == "volume":
            if self.index > 0:
                fmap1 = self.video.fmaps[ii,self.index+1].to(self.device).unsqueeze(0)
                fmap2 = self.video.fmaps[jj,self.index+1].to(self.device).unsqueeze(0)
            else:
                c = (ii == jj).long()
                fmap1 = self.video.fmaps[ii,0].to(self.device).unsqueeze(0)
                fmap2 = self.video.fmaps[jj,c].to(self.device).unsqueeze(0)
            corr = CorrBlock(fmap1, fmap2)
            self.corr = corr if self.corr is None else self.corr.cat(corr)

            inp = self.video.inps[ii,self.index].to(self.device).unsqueeze(0)
            self.inp = inp if self.inp is None else torch.cat([self.inp, inp], 1)

        with torch.cuda.amp.autocast(enabled=False):
            target, _ = self.video.reproject(ii, jj, self.index)
            weight = torch.zeros_like(target)

        self.ii = torch.cat([self.ii, ii], 0)
        self.jj = torch.cat([self.jj, jj], 0)
        self.age = torch.cat([self.age, torch.zeros_like(ii)], 0)
        self.eset.update([(i.item(), j.item()) for i, j in zip(ii, jj)])

        # reprojection factors
        self.net = net if self.net is None else torch.cat([self.net, net], 1)

        self.target = torch.cat([self.target, target], 1)
        self.weight = torch.cat([self.weight, weight], 1)

    @torch.cuda.amp.autocast(enabled=True)
    def update_factors(self, ii, jj):
        rm = torch.zeros(self.ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set([(i.item(), j.item()) for i, j in zip(ii, jj)])
        for k, (i, j) in enumerate(zip(self.ii, self.jj)):
            rm[k] = (i.item(), j.item()) not in eset

        self.rm_factors(rm, store=True)

        add = torch.zeros(ii.shape[0], dtype=torch.bool, device=ii.device)
        eset = set([(i.item(), j.item()) for i, j in zip(self.ii, self.jj)])
        for k, (i, j) in enumerate(zip(ii, jj)):
            add[k] = (i.item(), j.item()) not in eset

        self.add_factors(ii[add], jj[add])
        
    @torch.cuda.amp.autocast(enabled=True)
    def rm_factors(self, mask, store=False):
        """ drop edges from factor graph """

        # store estimated factors
        if store:
            self.video.globuf.add_rel_poses(self.ii[mask], self.jj[mask], self.target[:,mask], self.weight[:,mask], self.index)
            self.ii_inac = torch.cat([self.ii_inac, self.ii[mask]], 0)
            self.jj_inac = torch.cat([self.jj_inac, self.jj[mask]], 0)
            self.target_inac = torch.cat([self.target_inac, self.target[:,mask]], 1)
            self.weight_inac = torch.cat([self.weight_inac, self.weight[:,mask]], 1)

        self.ii = self.ii[~mask]
        self.jj = self.jj[~mask]
        self.age = self.age[~mask]
        
        if self.corr_impl == "volume":
            self.corr = self.corr[~mask]

        if self.net is not None:
            self.net = self.net[:,~mask]

        if self.inp is not None:
            self.inp = self.inp[:,~mask]

        self.target = self.target[:,~mask]
        self.weight = self.weight[:,~mask]


    def release_buffer(self, window, shift):
        self.ii -= shift
        self.jj -= shift
        self.ii_inac -= shift
        self.jj_inac -= shift
        mask = (self.ii_inac < 0) | (self.jj_inac < 0)
        self.ii_inac = self.ii_inac[~mask]
        self.jj_inac = self.jj_inac[~mask]
        self.target_inac = self.target_inac[:,~mask]
        self.weight_inac = self.weight_inac[:,~mask]
        for ii in range(window):
            self.damping[ii] = self.damping[shift+ii]
        self.eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

    @torch.cuda.amp.autocast(enabled=True)
    def rm_keyframe(self, ix):
        """ drop edges from factor graph """

        if self.index == 0:
            self.video.rm_keyframe(ix)

        m = (self.ii_inac == ix) | (self.jj_inac == ix)
        self.ii_inac[self.ii_inac >= ix] -= 1
        self.jj_inac[self.jj_inac >= ix] -= 1

        if torch.any(m):
            self.ii_inac = self.ii_inac[~m]
            self.jj_inac = self.jj_inac[~m]
            self.target_inac = self.target_inac[:,~m]
            self.weight_inac = self.weight_inac[:,~m]

        m = (self.ii == ix) | (self.jj == ix)

        self.ii[self.ii >= ix] -= 1
        self.jj[self.jj >= ix] -= 1
        self.rm_factors(m, store=False)
        self.eset = set(
            [(i.item(), j.item()) for i, j in zip(self.ii, self.jj)] +
            [(i.item(), j.item()) for i, j in zip(self.ii_inac, self.jj_inac)])

    @torch.cuda.amp.autocast(enabled=True)
    def get_network_update(self, t0=None, EP=1e-7, tracking=False):
        """ run update operator on factor graph """

        if tracking:
            mask = self.ii == self.ii.max()
            t0 = self.ii.max().item()
            corr = self.corr.view(mask)
        else:
            mask = self.ii >= 0
            corr = self.corr

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.video.reproject(self.ii[mask], self.jj[mask], index=self.index)
            motn = torch.cat([coords1 - self.coords0, self.target[:,mask] - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        # correlation features
        corr = corr(coords1)
        self.net[:,mask], delta, weight, damping, _ = \
            self.update_op(self.net[:,mask], self.inp[:,mask], corr, motn, self.ii[mask], self.jj[mask])

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target[:,mask] = coords1 + delta.to(dtype=torch.float)
            self.weight[:,mask] = weight.to(dtype=torch.float)
            self.damping[torch.unique(self.ii[mask])] = damping
            self.age += 1

            if not tracking:
                m = (self.ii_inac >= t0 - 5) & (self.jj_inac >= t0 - 5)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)
                damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
                return t0, target, weight, damping[None], ii, jj
            else:
                damping = .2 * self.damping[torch.unique(self.ii[mask])].contiguous() + EP
                return t0, self.target[:,mask], self.weight[:,mask], damping[None], self.ii[mask], self.jj[mask]

    @torch.cuda.amp.autocast(enabled=True)
    def update(self, t0=None, t1=None, itrs=2, tracking=False, return_update=False, EP=1e-7):
        """ run update operator on factor graph """

        if tracking:
            mask = self.ii == self.ii.max()
            t0 = self.ii.max().item()
            corr = self.corr.view(mask)
        else:
            mask = self.ii >= 0
            corr = self.corr

        # motion features
        with torch.cuda.amp.autocast(enabled=False):
            coords1, _ = self.video.reproject(self.ii[mask], self.jj[mask], self.index)
            motn = torch.cat([coords1 - self.coords0, self.target[:,mask] - coords1], dim=-1)
            motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

        # correlation features
        corr = corr(coords1)
        self.net[:,mask], delta, weight, damping, upmask = \
            self.update_op(self.net[:,mask], self.inp[:,mask], corr, motn, self.ii[mask], self.jj[mask])

        if t0 is None:
            t0 = max(1, self.ii.min().item()+1)

        with torch.cuda.amp.autocast(enabled=False):
            self.target[:,mask] = coords1 + delta.to(dtype=torch.float)
            self.weight[:,mask] = weight.to(dtype=torch.float)
            self.damping[torch.unique(self.ii[mask])] = damping
            ht, wd = self.coords0.shape[0:2]

            if tracking:
                ii, jj, target, weight = self.ii[mask], self.jj[mask], self.target[:,mask], self.weight[:,mask]
            else:
                m = (self.ii_inac >= t0 - 5) & (self.jj_inac >= t0 - 5)
                ii = torch.cat([self.ii_inac[m], self.ii], 0)
                jj = torch.cat([self.jj_inac[m], self.jj], 0)
                target = torch.cat([self.target_inac[:,m], self.target], 1)
                weight = torch.cat([self.weight_inac[:,m], self.weight], 1)

            damping = .2 * self.damping[torch.unique(ii)].contiguous() + EP
            target = target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            self.age += 1

            # TODO: move to after BA operation
            if not tracking and self.video.vis and self.index == 0:
                self.video.upsample(torch.unique(self.ii[mask]), upmask, index=self.index)

            # dense bundle adjustment
            if not return_update:
                self.video.ba(target, weight, damping, ii, jj, t0, t1, itrs=itrs, lm=1e-5, ep=0.01)
                self.video.upsample(torch.unique(self.ii[mask]), upmask, index=self.index)
            else:
                return t0, target, weight, damping, ii, jj, mask, upmask

    @torch.cuda.amp.autocast(enabled=False)
    def update_lowmem(self, t0=None, t1=None, itrs=2, pose_graph=False, EP=1e-7, steps=8):
        """ run update operator on factor graph - reduced memory implementation """

        indices = torch.cat([self.ii, self.jj])
        iu, iu_exp = torch.unique(indices, return_inverse=True)
        iu_exp, ju_exp = torch.split(iu_exp, [self.ii.shape[0], self.jj.shape[0]])

        # alternate corr implementation
        num, rig, ch, ht, wd = self.video.fmaps[:,:2].shape
        num = iu.shape[0]
        corr_op = AltCorrBlock(self.video.fmaps[iu.cpu(),:2].reshape(1, num*rig, ch, ht, wd).cuda())

        for step in range(steps):
            print("Global BA Iteration #{}".format(step+1))
            with torch.cuda.amp.autocast(enabled=False):
                coords1, mask = self.video.reproject(self.ii, self.jj)
                motn = torch.cat([coords1 - self.coords0, self.target - coords1], dim=-1)
                motn = motn.permute(0,1,4,2,3).clamp(-64.0, 64.0)

            s = 1
            masks = torch.ones(self.ii.shape, device='cuda', dtype=torch.bool)
            for i in range(0, self.jj.max()+1, s):
                v = (self.ii >= i) & (self.ii < i + s)
                if not torch.any(v):
                    continue
                iis = self.ii[v]
                jjs = self.jj[v]

                ht, wd = self.coords0.shape[0:2]
                corr1 = corr_op(coords1[:,v], rig * iu_exp[v], rig * ju_exp[v] + (iis == jjs).long())

                with torch.cuda.amp.autocast(enabled=True):
                 
                    net, delta, weight, damping, upmask = \
                        self.update_op(self.net[:,v], self.video.inps[None,iis.cpu(),0].cuda(), corr1, motn[:,v], iis, jjs)

                self.net[:,v] = net
                self.target[:,v] = coords1[:,v] + delta.float()
                self.weight[:,v] = weight.float()
                self.damping[torch.unique(iis)] = damping
                valid_weight_percent = 1e-2
                masks[v] = (weight > 0.1).float().mean([-3,-2,-1]) > valid_weight_percent

            target = self.target.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            weight = self.weight.view(-1, ht, wd, 2).permute(0,3,1,2).contiguous()
            target, weight, ii, jj = target[masks], weight[masks], self.ii[masks], self.jj[masks]
            damping = .2 * self.damping[torch.unique(torch.cat((torch.arange(t0, t1, device='cuda'), ii)))].contiguous() + EP
            print("- filtered {}/{}".format(ii.shape[0], self.ii.shape[0]))
            if ii.shape[0] < 1:
                return

            if pose_graph:
                self.video.global_pose_ba(target, weight, damping, ii, jj, 1, t1,
                    itrs=itrs, lm=1e-7, ep=1e-4)
            else:
                # dense bundle adjustment
                self.video.ba(target, weight, damping, ii, jj, 1, t1,
                    itrs=itrs, lm=1e-5, ep=1e-2)

            self.video.num_factors.value = ii.shape[0]
            self.video.ii[:self.video.num_factors.value, 0] = ii
            self.video.jj[:self.video.num_factors.value, 0] = jj
            self.video.dirty[:t1] = True

    def add_neighborhood_factors(self, t0, t1, r=3):
        """ add edges between neighboring frames within radius r """

        ii, jj = torch.meshgrid(torch.arange(t0,t1), torch.arange(t0,t1), indexing='ij')
        ii = ii.reshape(-1).to(dtype=torch.long, device=self.device)
        jj = jj.reshape(-1).to(dtype=torch.long, device=self.device)

        c = 1 if self.video.stereo else 0

        keep = ((ii - jj).abs() > c) & ((ii - jj).abs() <= r)
        self.add_factors(ii[keep], jj[keep])

    
    def add_proximity_factors(self, t0=0, t1=0, rad=2, nms=2, beta=0.25, thresh=16.0, remove=False):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value
        ix = torch.arange(t0, t)
        jx = torch.arange(t1, t)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf

        ii1 = torch.cat([self.ii, self.ii_inac], 0)
        jj1 = torch.cat([self.jj, self.jj_inac], 0)
        mask = (ii1 > min(t0, t1)) & (jj1 > min(t0, t1))
        ii1, jj1 = ii1[mask], jj1[mask]
        for i, j in zip(ii1.cpu().numpy(), jj1.cpu().numpy()):
            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf


        es = []
        for i in range(t0, t):
            if self.video.stereo and self.index == 0:
                es.append((i, i))
                d[(i-t0)*(t-t1) + (i-t1)] = np.inf

            for j in range(max(i-rad-1,0), i):
                es.append((i,j))
                es.append((j,i))
                d[(i-t0)*(t-t1) + (j-t1)] = np.inf

        ix = torch.argsort(d)
        for k in ix:
            if d[k].item() > thresh:
                continue

            if len(es) > self.max_factors:
                break

            i = ii[k]
            j = jj[k]
            
            # bidirectional
            es.append((i, j))
            es.append((j, i))

            for di in range(-nms, nms+1):
                for dj in range(-nms, nms+1):
                    if abs(di) + abs(dj) <= max(min(abs(i-j)-2, nms), 0):
                        i1 = i + di
                        j1 = j + dj

                        if (t0 <= i1 < t) and (t1 <= j1 < t):
                            d[(i1-t0)*(t-t1) + (j1-t1)] = np.inf

        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        self.add_factors(ii, jj, remove)

    def add_proximity_factors_backend(self, t, rad, nms, beta, thresh, framedist, relset):
        """ add edges to the factor graph based on distance """

        t = self.video.counter.value if t is None else t
        ix = torch.arange(0, t)
        jx = torch.arange(0, t)

        ii, jj = torch.meshgrid(ix, jx, indexing='ij')
        ii = ii.reshape(-1)
        jj = jj.reshape(-1)

        d = self.video.distance(ii, jj, beta=beta)
        d[ii - rad < jj] = np.inf
        d[d > 100] = np.inf
        d_matrix = d.reshape(t, t)
        for (i, j) in relset:
            d_matrix[max(0,i-nms):min(t,i+nms), max(0,j-nms):min(t,j+nms)] = np.inf
            d_matrix[max(0,j-nms):min(t,j+nms), max(0,i-nms):min(t,i+nms)] = np.inf
        d = d_matrix.reshape(-1)

        es = []
        ds = []
        ix = torch.argsort(d)
        for k in ix:
            dd = d_matrix[int(k/t), k%t].item()
            if dd > thresh:
                continue
            if len(es) >= self.max_factors:
                break
            i = ii[k].item()
            j = jj[k].item()
            if abs(i - j) < framedist:
                continue
            # bidirectional
            es.append((i, j))
            ds.append(dd)
            es.append((j, i))
            ds.append(dd)
            d_matrix[max(0,i-nms):min(t,i+nms), max(0,j-nms):min(t,j+nms)] = np.inf
            d_matrix[max(0,j-nms):min(t,j+nms), max(0,i-nms):min(t,i+nms)] = np.inf

        if len(es) == 0:
            return

        # print("- compute distance - 3")
        self.video.ds = torch.as_tensor(ds, device=self.device)
        ii, jj = torch.as_tensor(es, device=self.device).unbind(dim=-1)
        # add stereo factor to constraint depth maps
        iu = torch.unique(ii)
        ii = torch.cat([ii, iu])
        jj = torch.cat([jj, iu])
        self.add_factors(ii, jj)
