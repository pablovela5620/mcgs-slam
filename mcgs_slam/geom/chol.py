import torch
import torch.nn.functional as F
import geom.projective_ops as pops

EP=1e-1
LM=1e-4

class CholeskySolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)

        return xs

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

class CholeskySolver2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b):
        # don't crash training if cholesky decomp fails
        try:
            U = torch.linalg.cholesky(H)
            xs = torch.cholesky_solve(b, U)
            ctx.save_for_backward(U, xs)
            ctx.failed = False
        except Exception as e:
            print(e)
            ctx.failed = True
            xs = torch.zeros_like(b)
            U = torch.zeros_like(H)

        return xs, U

    @staticmethod
    def backward(ctx, grad_x):
        if ctx.failed:
            return None, None

        U, xs = ctx.saved_tensors
        dz = torch.cholesky_solve(grad_x, U)
        dH = -torch.matmul(xs, dz.transpose(-1,-2))

        return dH, dz

def block_solve(H, b, Hgdir=None, vgdir=None, Hi_gdir=None, Hj_gdir=None, H_gba=None, v_gba=None, ep=EP, lm=LM, fix_front=0):
    """ solve normal equations """
    B, N, _, D, _ = H.shape
    H = H.permute(0,1,3,2,4)
    H = H.reshape(B, N*D, N*D)
    b = b.reshape(B, N*D, 1)

    if Hgdir is not None:
        _, G, _ = Hgdir.shape
        G = 3
        H = torch.cat((H, torch.zeros((B,G,N*D), device='cuda')), dim=1)
        H = torch.cat((H, torch.zeros((B,N*D+G,G), device='cuda')), dim=2)
        H[:,-G:,-G:] = Hgdir[:,:G,:G]
        Hgdir_i = Hi_gdir.transpose(2,3)
        Hgdir_j = Hj_gdir.transpose(2,3)
        for i in range(N-1):
            # print(i, i*D,i*D+D, (i+1)*D,(i+1)*D+D)
            H[:, -G:, i*D:i*D+D] += Hgdir_i[:,i,:G]
            H[:, i*D:i*D+D, -G:] += Hi_gdir[:,i,:,:G]
            H[:, -G:, (i+1)*D:(i+1)*D+D] += Hgdir_j[:,i,:G]
            H[:, (i+1)*D:(i+1)*D+D, -G:] += Hj_gdir[:,i,:,:G]

        b = torch.cat((b, torch.zeros((B,G,1), device='cuda')), dim=1)
        b[:,-G:,:] = vgdir[:,:G,:]
        I = torch.eye(N*D+G).to(H.device)
    else:
        I = torch.eye(N*D).to(H.device)

    if H_gba is not None:
        H[:, N*D:N*D+3, N*D:N*D+3] = H_gba[:, :3, :3]
        b[:, N*D:N*D+3, :] += v_gba[:, :3, :]
        H[:, 6:9, 6:9] += H_gba[:, 3:, 3:]
        b[:, 6:9, :] += v_gba[:, 3:, :]

    H = H + (ep + lm*H) * I
    H = H[:,fix_front:,fix_front:]
    b = b[:,fix_front:,:]
    x = CholeskySolver.apply(H,b)
    x = torch.cat((torch.zeros_like(x)[:,:fix_front,:], x), dim=1)

    if Hgdir is not None:
        return x[:,:-G].reshape(B, N, D), x[:,-G:]
    return x.reshape(B, N, D)


def schur_solve(H, E, C, v, w, Hgdir=None, vgdir=None, Hi_gdir=None, Hj_gdir=None, ep=EP, lm=LM, sless=False, fix_front=0):
    """ solve using shur complement """
    
    B, P, M, D, HW = E.shape
    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    v = v.reshape(B, P*D, 1)
    w = w.reshape(B, M*HW, 1)
    Q = (1.0 / C).view(B, M*HW, 1)

    if Hgdir is not None:
        G = 1
        H = torch.cat((H, torch.zeros((B,G,P*D), device='cuda')), dim=1)
        H = torch.cat((H, torch.zeros((B,P*D+G,G), device='cuda')), dim=2)
        H[:,-G:,-G:] = Hgdir[:,-G:,-G:]
        Hgdir_i = Hi_gdir.transpose(2,3)
        Hgdir_j = Hj_gdir.transpose(2,3)
        for i in range(P-1):
            # print(i, i*D,i*D+D, (i+1)*D,(i+1)*D+D)
            H[:, -G:, i*D:i*D+D] += Hgdir_i[:,i,-G:]
            H[:, i*D:i*D+D, -G:] += Hi_gdir[:,i,:,-G:]
            H[:, -G:, (i+1)*D:(i+1)*D+D] += Hgdir_j[:,i,-G:]
            H[:, (i+1)*D:(i+1)*D+D, -G:] += Hj_gdir[:,i,:,-G:]

        v = torch.cat((v, torch.zeros((B,G,1), device='cuda')), dim=1)
        v[:,-G:,:] = vgdir[:,-G,:]

        E = torch.cat((E, torch.zeros((B,G,M*HW), device='cuda')), dim=1)
        I = torch.eye(P*D+G).to(H.device)
    else:
        I = torch.eye(P*D).to(H.device)

    # damping
    H = H + (ep + lm*H) * I

    H = H[:,fix_front:,fix_front:]
    E = E[:,fix_front:,:]
    v = v[:,fix_front:,:]

    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    dx = CholeskySolver.apply(S, v)
    if sless:
        return dx.reshape(B, P, D)

    dz = Q * (w - Et @ dx)    
    dx = torch.cat((torch.zeros_like(dx)[:,:fix_front,:], dx), dim=1)
    dz = dz.reshape(B, M, HW)

    if Hgdir is not None:
        return dx[:,:-G].reshape(B, P, D), dz, dx[:,-G:]

    dx = dx.reshape(B, P, D)
    return dx, dz

def schur_solve_multi(H, E, C, v, w, ep=EP, lm=LM):
    """ solve using shur complement """

    print("!!!!!!!!!!!!!!!!!!1 schur_solve_multi")
    B, P, M, D, HW = E.shape
    
    E = E.permute(0,1,3,2,4).reshape(B, P*D, M*HW)
    Et = E.transpose(1,2)
    Q = (1.0 / C).view(B, M*HW, 1)
    w = w.reshape(B, M*HW, 1)

    M = int(M/3)
    EE = E.split(M*HW, 2)
    EtEt = Et.split(M*HW, 1)
    QQ = Q.split(M*HW, 1)
    ww = w.split(M*HW, 1)

    H = H.permute(0,1,3,2,4).reshape(B, P*D, P*D)
    v = v.reshape(B, P*D, 1)

    # damping
    I = torch.eye(P*D).to(H.device)
    H = H + (ep + lm*H) * I

    S = H
    for i in range(3):
        E, Q = EE[i], QQ[i]
        S = S - torch.matmul(E, Q*EtEt[i])
        v = v - torch.matmul(E, Q*ww[i])
    dx = CholeskySolver.apply(S, v)
    dz = torch.cat([QQ[i] * (ww[i] - EtEt[i] @ dx) for i in range(3)])

    dz = dz.reshape(B, 3*M, HW)
    dx = dx.reshape(B, P, D)
    return dx, dz


def solve_dR(H, b, ep=EP, lm=LM):
    B, D, _ = H.shape
    I = torch.eye(D).to(H.device)
    H = H + (ep + lm*H) * I

    x = CholeskySolver.apply(H,b)
    return x

def schur_solve_prior(C, w, Hs, Es, vs, ep=0.1, lm=0.0001, dzcov=False):
    """ solve using shur complement """
    D = Hs.shape[-1]
    B, M, HW = C.shape
    Q = (1.0 / C).view(B, M*HW, 1)
    w = w.reshape(B, M*HW, 1)

    H = Hs.permute(0,1,3,2,4).reshape(B, M*D, M*D)
    E = Es.permute(0,1,3,2,4).reshape(B, M*D, M*HW)
    v = vs.reshape(B, M*D, 1)

    # damping
    I = torch.eye(M*D).to(H.device)
    H = H + (ep + lm*H) * I
    
    Et = E.transpose(1,2)
    S = H - torch.matmul(E, Q*Et)
    v = v - torch.matmul(E, Q*w)

    dso, L = CholeskySolver2.apply(S, v)
    dz = Q * (w - Et @ dso)
    dz = dz.reshape(B, M, HW)
    dso = dso.reshape(B, M, D)

    F = torch.linalg.inv(L) @ (E * Q[...,0])
    dzcov = torch.sum(torch.square(F), dim=1) + Q[...,0]
    # dzcov = Q[...,0]
    dzcov = dzcov.reshape(M, HW)
    return dso, dz, dzcov