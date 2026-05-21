import torch
from lietorch import SE3
eps = 1e-8


def diff(x1, x2):
    return (x1 - x2) / 2. / eps


def num_jacobi(func, Gi, Gj=None, first=True):
    batch = Gi.shape[0]
    J_num = []
    for i in range(6):
        delta = torch.zeros(6, device='cuda').expand(batch, 1, 6).type(torch.float64)
        delta[:, :, i] = eps
        if first:
            J_num.append(diff(func(SE3.exp(delta)*Gi, Gj), func(SE3.exp(-delta)*Gi, Gj)))
        else:
            J_num.append(diff(func(Gi, SE3.exp(delta)*Gj), func(Gi, SE3.exp(-delta)*Gj)))
    return torch.stack(J_num, dim=-1)


def global_relative_pose_constraints(ii, jj, poses, rel_poses, infos, Tcb=None, pw=1e-5, verbose=False):
    torch.set_printoptions(precision=8, sci_mode=False, linewidth=120)

    Gii = SE3(poses[:, ii].data.type(torch.float64))
    Gjj = SE3(poses[:, jj].data.type(torch.float64))

    # relative constraint
    Gij = SE3(rel_poses.data.type(torch.float64))
    if verbose:
        print("Estimated", (Gii * Gjj.inv()).data)
        print("Reference", Gij.data)

    def func(Gii, Gjj, Tcb=Tcb):
        if Tcb is not None:
            Tcb = Tcb.double('cuda')
            e = Gij * Tcb * Gii * Gjj.inv() * Tcb.inv()
        else:
            # e = Gii * Gjj.inv() * Gij
            e = Gij * Gii * Gjj.inv()
        return e.log()
        return e.data[:,:,:6]

    # numerical jacobi
    Ji = num_jacobi(func, Gii, Gjj, first=True).type(torch.float32)
    Jj = num_jacobi(func, Gii, Gjj, first=False).type(torch.float32)

    # Analytical jacobi
    # Jlinv = SE3.exp(func(Gii, Gjj)).Jinv_mat().reshape(-1,6,6)
    # Jrinv = SE3.exp(-func(Gii, Gjj)).Jinv_mat().reshape(-1,6,6)
    # Ji = (Gii * Gjj.inv()).inv()[:,:,None,:].adjT(Jrinv[None]).type(torch.float32)
    # Jj = -Jrinv[None].type(torch.float32)

    # print("- Ji num", Ji.shape, "\n", Ji[0,5])
    # print("- Ji ana", Jia.shape, "\n", Jia[0,5])
    # print("- Ji diff\n", Jia[0,0] - Ji[0,0])    
    # print("- Jj num", Jj.shape, "\n", Jj[0,5])
    # print("- Jj num", Jja.shape, "\n", Jja[0,5])
    # print("- Jj diff\n", Jja[0,0] - Jj[0,0])
    # return

    r = func(Gii, Gjj).unsqueeze(-1).type(torch.float32)
    chi2 = torch.sum(r.transpose(2, 3) @ r)
    chi2_scaled = torch.sum(r.transpose(2, 3) @ infos @ r)
    # print("- Sum of chi2 relative pose errors", chi2.item(), chi2_scaled.item())

    wJiT = ((pw * Ji.double()).transpose(2, 3) @ infos.double()).float()
    wJjT = ((pw * Jj.double()).transpose(2, 3) @ infos.double()).float()
    Hsp = torch.stack([torch.matmul(wJiT, Ji), torch.matmul(wJiT, Jj), torch.matmul(wJjT, Ji), torch.matmul(wJjT, Jj)])    # 4x7x6x6
    vsp = -torch.stack([torch.matmul(wJiT, r), torch.matmul(wJjT, r)]).squeeze(-1)  # 2x7x6
    return Hsp, vsp, chi2, chi2_scaled


def relative_pose_constraints(t0, t1, poses, pose_priors, pw=1e1, verbose=False):
    if verbose:
        torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)

    jjp = torch.arange(t0, t1, device='cuda')
    iip = jjp - 1
    if verbose:
        print(iip, jjp)
    Gs = SE3(poses)
    Gs_prior = SE3(pose_priors)

    # relative constraint
    Gii = SE3(Gs[iip].data.type(torch.float64))
    Gjj = SE3(Gs[jjp].data.type(torch.float64))
    Gij_prior = Gs_prior[iip] * Gs_prior[jjp].inv()  # G_ij* = (G_iw*) * (G_jw*)^-1
    Gij_prior = SE3(Gij_prior.data.type(torch.float64))
    if verbose:
        print("Estimated", (Gii * Gjj.inv()).data)
        print("Reference", Gij_prior.data)

    def func(Gii, Gjj):
        return Gij_prior.inv() * Gii * Gjj.inv()   # dx = log( (G_ij*)^-1 * G_i * G_j^-1)

    # numerical jacobi
    Ji = num_jacobi(func, Gii, Gjj, first=True).type(torch.float32)
    Jj = num_jacobi(func, Gii, Gjj, first=False).type(torch.float32)
    r = func(Gii, Gjj).log().unsqueeze(-1).type(torch.float32)  # 7x6x1
    if verbose:
        print("Residual", func(Gii, Gjj).log())
    # print("Mean relpose residual", r.abs().sum())

    wJiT = (pw * Ji).transpose(1, 2)
    wJjT = (pw * Jj).transpose(1, 2)
    Hsp = torch.stack([torch.matmul(wJiT, Ji), torch.matmul(wJiT, Jj), torch.matmul(wJjT, Ji), torch.matmul(wJjT, Jj)])    # 4x7x6x6
    vsp = torch.stack([-torch.matmul(wJiT, r), -torch.matmul(wJjT, r)]).squeeze(-1)  # 2x7x6

    return Hsp, vsp
