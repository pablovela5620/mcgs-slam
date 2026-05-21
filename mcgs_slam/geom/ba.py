import math
import torch
import droid_backends

from .chol import block_solve, schur_solve, solve_dR, schur_solve_prior
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))

def construct_joint_BA(targets, weights, etas, iis, jjs, poses, disps_list, intrs, base, T_ci_c0, t0, D=6):
    Es, Cs, ws = [], [], []
    for i in range(len(targets)):
        if i == 0:
            H, E, C, v, w, chi2, chi2R = BA_prepare(targets[i], weights[i], etas[i], poses, disps_list[i], intrs[:,:,[0,1]], base, iis[i], jjs[i], T_ci_c0[i], fixedp=t0, D=D)
        else:
            E, C, w, chi2, chi2R = BA_prepare(targets[i], weights[i], etas[i], poses, disps_list[i], intrs[:,:,[i+1]], base, iis[i], jjs[i], T_ci_c0[i], H, v, fixedp=t0, D=D)
        Es.append(E)
        Cs.append(C)
        ws.append(w)
    # print("- - Chi2 error re-proj raw/robust: {:.4f} {:.4f}".format((chi2+chi22+chi23).item(), (chi2R+chi2R2+chi2R3).item()))
    E = torch.cat(Es, dim=2)
    C = torch.cat(Cs, dim=1)
    w = torch.cat(ws, dim=1)
    return H, E, C, v, w

def BA_prepare(target, weight, eta, poses, disps, intrinsics, base, ii, jj, T_ci_c0=None,
               H=None, v=None, fixedp=1, D=6):
    """ Construct linear system for Full Bundle Adjustment """

    disps = disps[:poses.shape[1]][None]
    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, base, ii, jj, jacobian=True, Tcb=T_ci_c0)

    r = (target - coords).view(B, N, -1, 1)
    rw = .001 * (valid * weight).view(B, N, -1, 1)
    chi2 = (rw * r).transpose(2,3) @ r

    # counts = []
    # for k in kx:
    #     rk = r[:,ii==k]
    #     num = rk.shape[1]
    #     rkmin = torch.min(torch.abs(rk), dim=1, keepdim=True)[0]
    #     counts.append(rk.shape[1])
    #     thresh = (1 * rkmin).repeat(1,num,1,1)
    #     thresh[thresh < 1] = 1
    #     mask = torch.abs(rk) > thresh
    #     x = torch.sqrt(2*thresh*torch.abs(rk) - thresh*thresh)  # huber
    #     rk[mask] *= x[mask] / torch.abs(rk[mask])
    #     r[:,ii==k] = rk

    chi2R = (rw * r).transpose(2,3) @ r

    ### 2: construct linear system ###
    if D != 6:
        Jnull = torch.cat([torch.zeros_like(Ji), torch.zeros_like(Ji)], dim=-1)[...,:D-6]
        Ji = torch.cat([Ji, Jnull], dim=-1).reshape(B, N, -1, D)
        Jj = torch.cat([Jj, Jnull], dim=-1).reshape(B, N, -1, D)
    else:
        Ji = Ji.reshape(B, N, -1, D)
        Jj = Jj.reshape(B, N, -1, D)
    wJiT = (rw * Ji).transpose(2,3)
    wJjT = (rw * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    rw = rw.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(rw*r*Jz, dim=-1)
    Ck = torch.sum(rw*Jz*Jz, dim=-1)

    # only optimize keyframe poses
    P = P - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)
    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)
    C += eta.view(*C.shape) + 1e-7
    E = E.view(B, P, M, D, ht*wd)

    if H is None:
        H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
            safe_scatter_add_mat(Hij, ii, jj, P, P) + \
            safe_scatter_add_mat(Hji, jj, ii, P, P) + \
            safe_scatter_add_mat(Hjj, jj, jj, P, P)
        v = safe_scatter_add_vec(vi, ii, P) + \
            safe_scatter_add_vec(vj, jj, P)
        return H, E, C, v, w, torch.sum(chi2), torch.sum(chi2R)
    else:
        H += safe_scatter_add_mat(Hii, ii, ii, P, P) + \
             safe_scatter_add_mat(Hij, ii, jj, P, P) + \
             safe_scatter_add_mat(Hji, jj, ii, P, P) + \
             safe_scatter_add_mat(Hjj, jj, jj, P, P)
        v += safe_scatter_add_vec(vi, ii, P) + \
             safe_scatter_add_vec(vj, jj, P)
        return E, C, w, torch.sum(chi2), torch.sum(chi2R)

def BA_solve(poses, disps, disps2, disps3, ii, jj, H, E, C, v, w, fixedp=1):
    B, P, ht, wd = disps.shape
    D = poses.manifold_dim
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    P = P - fixedp
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz[:,:M].view(B,-1,ht,wd), kx)
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    if disps2 is not None:
        disps2 = disp_retr(disps2, dz[:,M:2*M].view(B,-1,ht,wd), kx)
        disps2 = torch.where(disps2 > 10, torch.zeros_like(disps2), disps2)
        disps2 = disps2.clamp(min=0.0)
        disps3 = disp_retr(disps3, dz[:,2*M:3*M].view(B,-1,ht,wd), kx)
        disps3 = torch.where(disps3 > 10, torch.zeros_like(disps3), disps3)
        disps3 = disps3.clamp(min=0.0)
        return poses, disps, disps2, disps3
    return poses, disps

def BA(target, weight, eta, poses, disps, intrinsics, base, ii, jj, fixedp=1):
    """ Full Bundle Adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, base, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)
    # print("- - residual1", torch.sum(ii==jj), torch.sum(ii!=jj))
    # print("- - - residual1", torch.sum(w[:,ii==jj]), torch.sum(torch.abs(r)[:,ii==jj])/torch.sum(ii==jj), torch.sum(torch.abs(w*r)[:,ii==jj]))
    # print("- - - residual2", torch.sum(w[:,ii!=jj]), torch.sum(torch.abs(r)[:,ii!=jj])/torch.sum(ii!=jj), torch.sum(torch.abs(w*r)[:,ii!=jj]))

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Jz = Jz.reshape(B, N, ht*wd, -1)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    wk = torch.sum(w*r*Jz, dim=-1)
    Ck = torch.sum(w*Jz*Jz, dim=-1)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # only optimize keyframe poses
    P = P - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    C = C + eta.view(*C.shape) + 1e-7

    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: solve the system ###
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, base, ii, jj, fixedp=1, rig=1):
    """ Motion only bundle adjustment """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: commpute jacobians and residuals ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, base, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: construct linear system ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    P = P - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: solve the system ###
    dx = block_solve(H, v)

    ### 4: apply retraction ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses

def get_prior_depth_aligned(depth_prior, scales):
    M, ht, wd = depth_prior.shape
    hs, ws = scales.shape[-2:]
    meshx, meshy = torch.meshgrid(torch.linspace(0, hs-1-1e-6, ht), torch.linspace(0, ws-1-1e-6, wd), indexing='ij')
    grid = torch.stack((meshy, meshx), -1).cuda()
    grid = grid.unsqueeze(0).expand(M, -1, -1, -1).contiguous()
    mscales_bi, Jbi = droid_backends.bi_inter(scales, grid)
    depth_prior_aligned = depth_prior * mscales_bi
    return depth_prior_aligned, Jbi

def JDSA(target, weight, eta, poses, disps, intrinsics, disps_prior, dscales, ii, jj, alpha):
    
    # TODO
    intrinsics = intrinsics[:, :4]

    B, P, ht, wd = disps.shape
    N = ii.shape[0]

    ### 1: commpute jacobians and residuals ###
    C, w = droid_backends.proj_trans(poses.data.squeeze(), disps[0], intrinsics[0], target, weight, ii, jj)

    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    disps_prior = disps_prior[kx]
    m = (disps_prior > 0).to(torch.float).view(-1, ht*wd)

    hs, ws = dscales.shape[-2:]
    disps_bi, Jbi = get_prior_depth_aligned(disps_prior, dscales[kx])

    rd = (disps[0,kx] - disps_bi).view(-1, ht*wd)
    Jd = torch.ones_like(rd).view(1, -1, 1, ht*wd)
    # Jd = (-1. / (disps[0,kx] ** 2)).view(1, -1, 1, ht*wd)
    Jso = -m.unsqueeze(-1) * disps_prior.view(-1, ht*wd).unsqueeze(-1) * Jbi.view(M, ht*wd, -1)[None]

    alpha = torch.ones(M,ht*wd,1).float().cuda() * alpha

    D = hs*ws
    fixedp = kx[0]
    kx = kx - fixedp
    wJsoT = (alpha * Jso).transpose(2,3)
    Hs = safe_scatter_add_mat(wJsoT @ Jso, kx, kx, M, M).view(B, M, M, D, D)
    Es = safe_scatter_add_mat(wJsoT * Jd, kx, kx, M, M).view(B, M, M, D, ht*wd)
    vs = safe_scatter_add_vec(-wJsoT @ rd[None].unsqueeze(-1), kx, M)
    kx += fixedp

    alpha = alpha.squeeze()
    C = C[None] + m * alpha * (Jd * Jd).squeeze() + (1-m) * eta.view(*C.shape)
    w = w[None] - m * alpha * rd * Jd.squeeze()

    ### 3: solve the system ###
    dso, dz, dzcov = schur_solve_prior(C, w, Hs, Es, vs, dzcov=True)

    ### 4: apply retraction ###
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)
    dscales[kx] += dso.view(-1, hs, ws)

    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.001)

    return disps, dscales, dzcov
