#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> projective_transform_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);



torch::Tensor depth_filter_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh);


torch::Tensor frame_distance_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj,
  const float beta);

std::vector<torch::Tensor> projmap_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics,
  torch::Tensor ii,
  torch::Tensor jj);

torch::Tensor iproj_cuda(
  torch::Tensor poses,
  torch::Tensor disps,
  torch::Tensor intrinsics);

torch::Tensor proj_cuda(
  torch::Tensor points,
  torch::Tensor intrinsics);

std::vector<torch::Tensor> bi_inter_cuda(
  torch::Tensor scales,
  torch::Tensor grids);

std::vector<torch::Tensor> proj_trans_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj);

std::vector<torch::Tensor> ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor base,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep);

torch::Tensor multi_cam_ba_cuda(
    torch::Tensor poses,
    std::vector<torch::Tensor> intrinsics,
    std::vector<torch::Tensor> disps_list,
    std::vector<torch::Tensor> disps_sens_list,
    std::vector<torch::Tensor> Tij_list,
    std::vector<torch::Tensor> Ticj_list,
    std::vector<torch::Tensor> Tcic0_list,
    std::vector<torch::Tensor> targets,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> etas,
    std::vector<torch::Tensor> iis,
    std::vector<torch::Tensor> jjs,
    const int t0,
    const int t1,
    const int D,
    const float lm,
    const float ep);

torch::Tensor global_pose_ba_cuda(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor disps_sens,
    torch::Tensor intrinsics,
    torch::Tensor Tij,
    torch::Tensor Tibj,
    torch::Tensor Tcb,
    torch::Tensor Hsp,
    torch::Tensor vsp,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor iip,
    torch::Tensor jjp,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep);

std::vector<torch::Tensor> corr_index_cuda_forward(
  torch::Tensor volume,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> corr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_forward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  int radius);

std::vector<torch::Tensor> altcorr_cuda_backward(
  torch::Tensor fmap1,
  torch::Tensor fmap2,
  torch::Tensor coords,
  torch::Tensor corr_grad,
  int radius);


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> ba(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor base,
    torch::Tensor disps_sens,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(disps_sens);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return ba_cuda(poses, disps, intrinsics, base, disps_sens, targets, weights,
                 eta, ii, jj, t0, t1, iterations, lm, ep);

}

torch::Tensor multi_cam_ba(
    torch::Tensor poses,
    std::vector<torch::Tensor> intrinsics,
    std::vector<torch::Tensor> disps_list,
    std::vector<torch::Tensor> disps_sens_list,
    std::vector<torch::Tensor> Tij_list,
    std::vector<torch::Tensor> Ticj_list,
    std::vector<torch::Tensor> Tcic0_list,
    std::vector<torch::Tensor> targets,
    std::vector<torch::Tensor> weights,
    std::vector<torch::Tensor> etas,
    std::vector<torch::Tensor> iis,
    std::vector<torch::Tensor> jjs,
    const int t0,
    const int t1,
    const int D,
    const float lm,
    const float ep) {

  CHECK_INPUT(poses);
  CHECK_INPUT(intrinsics[0]);
  CHECK_INPUT(disps_list[0]);
  CHECK_INPUT(targets[0]);
  CHECK_INPUT(weights[0]);
  CHECK_INPUT(etas[0]);
  CHECK_INPUT(iis[0]);
  CHECK_INPUT(jjs[0]);

  return multi_cam_ba_cuda(poses, intrinsics, disps_list, disps_sens_list, Tij_list, Ticj_list, Tcic0_list, targets, weights, etas, iis, jjs, t0, t1, D, lm, ep);
}

torch::Tensor global_pose_ba(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor disps_sens,
    torch::Tensor intrinsics,
    torch::Tensor Tij,
    torch::Tensor Tibj,
    torch::Tensor Tcb,
    torch::Tensor Hsp,
    torch::Tensor vsp,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor eta,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor iip,
    torch::Tensor jjp,
    const int t0,
    const int t1,
    const int iterations,
    const float lm,
    const float ep) {
  return global_pose_ba_cuda(poses, disps, disps_sens, intrinsics,
                 Tij, Tibj, Tcb,
                 Hsp, vsp, targets, weights, eta, ii, jj, iip, jjp, t0, t1, iterations, lm, ep);
}

torch::Tensor frame_distance(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    const float beta) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return frame_distance_cuda(poses, disps, intrinsics, ii, jj, beta);

}


std::vector<torch::Tensor> projmap(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return projmap_cuda(poses, disps, intrinsics, ii, jj);

}


torch::Tensor iproj(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics) {
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);

  return iproj_cuda(poses, disps, intrinsics);
}

torch::Tensor proj(
    torch::Tensor points,
    torch::Tensor intrinsics) {
  CHECK_INPUT(points);
  CHECK_INPUT(intrinsics);

  return proj_cuda(points, intrinsics);
}

std::vector<torch::Tensor> proj_trans(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor targets,
    torch::Tensor weights,
    torch::Tensor ii,
    torch::Tensor jj) {

  CHECK_INPUT(targets);
  CHECK_INPUT(weights);
  CHECK_INPUT(poses);
  CHECK_INPUT(disps);
  CHECK_INPUT(intrinsics);
  CHECK_INPUT(ii);
  CHECK_INPUT(jj);

  return proj_trans_cuda(poses, disps, intrinsics, targets, weights, ii, jj);
}

std::vector<torch::Tensor> bi_inter(
    torch::Tensor scales,
    torch::Tensor grids) {
  CHECK_INPUT(scales);
  CHECK_INPUT(grids);

  return bi_inter_cuda(scales, grids);
}

// c++ python binding
std::vector<torch::Tensor> corr_index_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);

  return corr_index_cuda_forward(volume, coords, radius);
}

std::vector<torch::Tensor> corr_index_backward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(volume);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  auto volume_grad = corr_index_cuda_backward(volume, coords, corr_grad, radius);
  return {volume_grad};
}

std::vector<torch::Tensor> altcorr_forward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);

  return altcorr_cuda_forward(fmap1, fmap2, coords, radius);
}

std::vector<torch::Tensor> altcorr_backward(
    torch::Tensor fmap1,
    torch::Tensor fmap2,
    torch::Tensor coords,
    torch::Tensor corr_grad,
    int radius) {
  CHECK_INPUT(fmap1);
  CHECK_INPUT(fmap2);
  CHECK_INPUT(coords);
  CHECK_INPUT(corr_grad);

  return altcorr_cuda_backward(fmap1, fmap2, coords, corr_grad, radius);
}


torch::Tensor depth_filter(
    torch::Tensor poses,
    torch::Tensor disps,
    torch::Tensor intrinsics,
    torch::Tensor ix,
    torch::Tensor thresh) {

    CHECK_INPUT(poses);
    CHECK_INPUT(disps);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(ix);
    CHECK_INPUT(thresh);

    return depth_filter_cuda(poses, disps, intrinsics, ix, thresh);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // bundle adjustment kernels
  m.def("ba", &ba, "bundle adjustment");
  m.def("multi_cam_ba", &multi_cam_ba, "multi-cam bundle adjustment");
  m.def("global_pose_ba", &global_pose_ba, "global pose bundle adjustment");
  m.def("frame_distance", &frame_distance, "frame_distance");
  m.def("projmap", &projmap, "projmap");
  m.def("depth_filter", &depth_filter, "depth_filter");
  m.def("iproj", &iproj, "back projection");
  m.def("proj", &proj, "projection");
  m.def("bi_inter", &bi_inter, "bilinear interpolation");
  m.def("proj_trans", &proj_trans, "projective transform");

  // correlation volume kernels
  m.def("altcorr_forward", &altcorr_forward, "ALTCORR forward");
  m.def("altcorr_backward", &altcorr_backward, "ALTCORR backward");
  m.def("corr_index_forward", &corr_index_forward, "INDEX forward");
  m.def("corr_index_backward", &corr_index_backward, "INDEX backward");
}