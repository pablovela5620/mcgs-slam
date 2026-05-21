import os
import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = float(os.getenv('MIN_DEPTH', "0.2"))

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float(), indexing='ij')

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics[...,:4])

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(), indexing='ij')

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics[...,:4])
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        Ja = torch.stack([
            d,  o,  o,  o,  Z, -Y,
            o,  d,  o, -Z,  o,  X,
            o,  o,  d,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(B, N, H, W, 4, 6)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, base, ii, jj, jacobian=False, return_depth=False, Tcb=None, Gij=None):
    """ map points from ii->jj """

    # inverse project
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii,0], jacobian=jacobian)

    # transform
    if Gij is None:
        if Tcb is not None:
            Gibj = Tcb * poses[:,jj] * poses[:,ii].inv()
            Gij = Gibj * Tcb.inv()
        else:
            Gij = poses[:,jj] * poses[:,ii].inv()
        Gij.data[:, ii==jj] = base

    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project
    x1, Jp = proj(X1, intrinsics[:,jj,0], jacobian=jacobian, return_depth=return_depth)

    # exclude points too far to camera
    basesize = 1 / torch.clamp(torch.norm(Gij.data[:, :, :3], dim=2)*40, min=5., max=100.)
    valid = (X0[...,3] > basesize[...,None, None]).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        # Ja[:, ii==jj] = 0
        Jj = torch.matmul(Jp, Ja)
        if Tcb is not None:
            Ji = -Gibj[:,:,None,None,None].adjT(Jj)
            Jj = Tcb[:,:,None,None,None].adjT(Jj)
        else:
            Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, base, ii, jj):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float(), indexing='ij')

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, base, ii, jj, False, True)

    valid *= (coords1[..., 2] > 0.2).unsqueeze(-1)
    return coords1[...,:2] - coords0, valid



            # eps = 1e-8
            # def diff(x1, x2):
            #     return (x1 - x2) / (2*eps)
            # def num_jacobi(func, X0, Gi, Gj=None, first=True):
            #     X0 = X0.type(torch.float64)
            #     Gi.data = Gi.data.type(torch.float64)
            #     Gj.data = Gj.data.type(torch.float64)
            #     if Tcb is not None:
            #         Tcb.data = Tcb.data.type(torch.float64)
            #     batch, N = Gi.shape[:2]
            #     J_num = []
            #     for i in range(6):
            #         delta = torch.zeros((batch,N,6), device='cuda').type(torch.float64)
            #         delta[:, :, i] = eps
            #         if first:
            #             J_num.append(diff(func(X0, SE3.exp(delta)*Gi, Gj), func(X0, SE3.exp(-delta)*Gi, Gj)))
            #         else:
            #             J_num.append(diff(func(X0, Gi, SE3.exp(delta)*Gj), func(X0, Gi, SE3.exp(-delta)*Gj)))
            #     return torch.stack(J_num, dim=-1)
            # def func(X0, Gii, Gjj):
            #     if Tcb is not None:
            #         Gij = Tcb * Gjj * Gii.inv() * Tcb.inv()
            #     else:
            #         Gij = Gjj * Gii.inv()
            #     return Gij[:,:,None,None] * X0
            # print("- Num X1", func(X0, poses[:,ii], poses[:,jj]).shape)
            # Ji = num_jacobi(func, X0, poses[:,ii], poses[:,jj], first=True).type(torch.float)
            # Jj = num_jacobi(func, X0, poses[:,ii], poses[:,jj], first=False).type(torch.float)
            # # print("- i-j", ii[0], jj[0])
            # # print("- X0", X0[0,0,0,0])
            # # print("- X1", X1[0,0,0,0])
            # # print("Ja", Ja[0,0,0,0])
            # # print("- Ji", Ji[0,0,0,0])
            # # print("Jj", Jj[0,0,0,0,:3,3:])
            # if Tcb is not None:
            #     Tcb.data = Tcb.data.type(torch.float)
            # # print("- Ji", Ji[0,0,0,0])
            # Ji = torch.matmul(Jp, Ji)
            # Jj = torch.matmul(Jp, Jj)
