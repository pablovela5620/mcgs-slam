# Sky handling for MCGS-SLAM: diagnosis and implementation plan

Status: researched, not implemented. Written 2026-08-28 from a deep-research pass
over the 2024–2026 sky-handling literature and this repo's code. The intended
reader is an implementing agent with no other context.

## Problem

Outdoor 3DGS maps from this pipeline (Waymo driving sequences) contain giant
soft sky/background gaussians. Image-space renders are fine (PSNR ~27.4 on
seq 100613) but the 3D map viewed from outside is a blob halo. The current
mitigation is display-only: `rerun_logger.py` culls splats whose largest axis
exceeds `max_splat_scale = 8.0` scaled units before logging.

Baseline to beat (seq 100613, `pixi run demo` then `pixi run ate` on a DGX
Spark): PSNR 27.43 / SSIM 0.840 / LPIPS 0.208, ATE RMSE 0.426 m, 159,871
gaussians, ~12 min wall. Any change here must hold or improve these numbers.

## Root-cause diagnosis (verified against this tree)

1. **Sky gaussians are not created from Metric3D directly.** Gaussians are
   built from BA disparities: `mcgs.py` `call_gs` passes
   `depths = scale_factor / video.disps_up[...]`. Metric3D enters only as a
   BA prior (`disps_sens`, weight `mono_depth_alpha: 0.001`). In textureless
   sky BA has no photometric signal, so `disps_up` collapses onto the prior.
   Metric3D ViT-small internally clamps at 200 m canonical (~197 m at Waymo
   focal), so sky lands at ~40 scaled units — inside the 100-unit
   `depth_trunc` in `gaussian_model.py`, so nothing filters it.
2. **`distCUDA2` inflates them.** New splat scale = kNN spacing × point_size
   (`gaussian_model.py`, `extend_from_pcd_seq` path). Far sky points are
   sparse, so their spacing is meters — that is the blob generator.
3. **Size pruning is disabled exactly where blobs are born.**
   `gs_backend.py` `initialize_map` calls
   `densify_and_prune(..., self.init_gaussian_extent, None)` —
   `max_screen_size=None` disables the size prune for the 1050 init
   iterations. During `map()` the prune threshold is `0.1 * gaussian_extent
   = 0.6` scaled units, 13× tighter than the viz cull.
4. **`loss_mapping += 10 * isotropic_loss`** (`gs_backend.py`, in `map()`)
   forces the artifacts into round balls.
5. **Three useful signals are computed and discarded:**
   - Rendered accumulated alpha: `cuda_rasterizer/forward.cu` writes
     `out_alpha`, `_C.rasterize_gaussians` returns it, but
     `diff_gaussian_rasterization/__init__.py` drops it
     (`return color, radii, depth, n_touched`).
   - Metric3D confidence: `motion_filter.py` does
     `pred_depth, _, output_dict = model.inference(...)` — the `_` is a
     per-pixel confidence map. Caveat: it is supervised only where LiDAR GT
     exists, so its sky values are uncalibrated; gate on per-frame
     percentiles, not absolute cutoffs, and verify empirically first.
   - Metric3D normal kappa (vMF concentration): channel 3 of
     `prediction_normal`, sliced away at `motion_filter.py` (`[:, :3]`).
6. **The needed multi-view filter already exists in this repo.**
   `mcgs_slam/visualization.py` (old Open3D viewer, dead code path in the
   pixi port) uses `droid_backends.depth_filter` with an absolute depth
   threshold plus a disparity gate. Depth error grows as z², so an absolute
   consistency threshold automatically rejects the far field — no sky
   semantics needed. GlORIE-SLAM / Splat-SLAM use exactly this before
   Gaussian insertion.

## What the field does (summary)

- **Driving-scene consensus recipe** (Street Gaussians, OmniRe/DriveStudio,
  PVG, StreetCrafter, DeSiRe-GS): semantic sky mask (SegFormer-B5
  cityscapes, class 10) + separate optimizable sky cubemap composited as
  `rgb + sky(viewdir) * (1 - alpha)` + alpha loss at λ=0.05. Street
  Gaussians ablation: +1.5 dB PSNR on sky-heavy scenes.
- **Luma-style dome** (confirmed from Luma's capture API): cubemap on a
  sphere at ~4.5× scene radius, drawn behind the splats. Postshot ships the
  same (`--create-sky-model`). MVSAnywhere's `regsplatfacto` implements it
  as 10k frozen Fibonacci-sphere gaussians (means `requires_grad=False`,
  SH0-only color, excluded from densify/prune) rendered as a second
  rasterization pass.
- **LichtFeld-Studio has no sky model** (wishlist issue #907); its HDRI is
  viewer-side compositing only. Useful pieces: a bounded one-sided mask
  penalty `L = mean(alpha * (1-mask)^2)` and far-field machinery that
  *protects* far gaussians (decay at 1/4 rate beyond 2× orbit radius) and
  sizes far seeds by angular footprint: `log_s = log(t * 4 / fx)`.
- **Loss-form lesson (StreetCrafter):** on sky pixels use
  `-log(1 - alpha)`; on non-sky pixels use binary entropy
  `-(a·log a + (1-a)·log(1-a))`, NOT `-log(alpha)` — the two-sided log
  barrier destroys thin structures (cables) caught inside sky-mask errors.
- **Convergent design rule:** exempt far/dome gaussians from big-splat
  pruning instead of tightening it (Street Gaussians, PVG, MTGS, INRIA
  hierarchical-3dgs, LichtFeld all do this independently).
- **Negative result:** no SLAM+3DGS system (S3PO-GS, OpenGS-SLAM — same
  MonoGS lineage as this repo) handles sky at all. Solutions come from the
  offline driving-reconstruction lineage.

## Implementation plan, in order

Each step is independently landable and A/B-testable with
`pixi run demo && pixi run ate` plus the visual checks at the bottom.

### 1. Gate GS depths with the in-repo multi-view filter (biggest win, ~½ day)

In `mcgs.py`, before depths enter the GS packets (`call_gs` and
`call_global_gs`), port the GlORIE/Splat-SLAM pattern using the filter
already linked in this repo:

```python
depths = 1.0 / disps_up                                   # metric, unscaled
thresh = 0.01 * depths.mean(dim=[1, 2])                   # scene-relative eta
count  = droid_backends.depth_filter(poses, disps_up, intrinsics, index, thresh)
depths[count < 2] = torch.nan
median = depths.view(depths.shape[0], -1).nanmedian(dim=1).values
mask   = depths < 3 * median[:, None, None]               # NaN < x is False
depths = torch.where(mask, depths * scale_factor, torch.zeros(()))
```

Zeroed depths are dropped for free by
`create_pcd_from_image_and_depth(..., project_valid_depth_only=True)` —
verify that flag is set on this call path. See `visualization.py` for the
exact `depth_filter` call signature (poses/disps/intrinsics device and
layout). Do NOT use a relative MVS-style check (`Δd/d < 0.01`) — it is
scale-free and does not discriminate against the far field.

### 2. Distance-based viz cull (~1 h)

`rerun_logger.py` `log_gaussians`: replace/augment the
`scales.max(axis=1) < max_splat_scale` test with distance from the
trajectory (`self._traj_centers` already exists):

```python
d = np.linalg.norm(centers[:, None, :] - traj[None, :, :], axis=-1).min(axis=1)
sane &= d < 20.0          # 20 scaled units = 100 m metric; make it a CLI arg
```

Drop `max_splat_scale` to ~1.0 at the same time. Optionally log culled
splats to a separate toggleable entity (`world/splats_far`) instead of
dropping them.

### 3. Fix the init-side amplifiers (~½ day)

- `gs_backend.py` `initialize_map`: pass `self.size_threshold` instead of
  `None` to `densify_and_prune` so size pruning runs during the 1050 init
  iterations.
- Far-point initial scale: in `gaussian_model.py`, for points beyond some
  depth, replace the `distCUDA2`-based scale with angular footprint
  `log_s = log(max(z * 4.0 / fx, 1e-6))` (LichtFeld `mrnf.cpp` formula).
- Consider replacing `10 * isotropic_loss` in `map()` with MVSAnywhere's
  pair: max/median scale-ratio hinge at 2.0, weight 0.1, every 10 steps,
  plus `flat_loss = exp(scales).amin(-1).mean()` weight 1.0. This changes
  optimization behavior — A/B PSNR/ATE before keeping.

### 4. Return alpha from the rasterizer (free, ~1 h)

`thirdparty/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
already receives `alpha` from `_C.rasterize_gaussians` and discards it.
Add it to the return tuple and to `render()`'s dict in
`mcgs_slam/gaussian/renderer/__init__.py`. Forward-only: no CUDA change,
no gradient yet. Unlocks diagnostics and everything below.

### 5. Sky mask → zero depth before unprojection (~½ day, first new dep)

Belt-and-braces on top of step 1. Options, all aarch64-safe:

| Option | Dep | Notes |
|---|---|---|
| MoGe-2 | `Ruicheng/moge-2-vitb-normal-onnx` or PyTorch ckpt | Sky = literal `inf` + validity mask; what LichtFeld & StreetCrafter use. Could replace/augment Metric3D as the prior. |
| SegFormer-B5 cityscapes | `transformers` (noarch) | `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`, `sky = (pred == 10)`. What the driving repos use. B0 suffices for binary sky. |
| MVSAnywhere sky head | torch.hub `nianticlabs/mvsanywhere` | Returns `sky_mask` directly, but it is an MVS model needing source views. |

Apply the mask by zeroing depth at sky pixels before
`create_pcd_from_image_and_depth`, and store it in `DepthVideo` (widen a
buffer or add one) so step 8 can reuse it.

### 6. Frozen sky dome (~2 days, correct sky pixels without CUDA changes)

Copy MVSAnywhere `regsplatfacto` (`src/regsplatfacto/regsplatfacto_model.py`):

- 10k points, Fibonacci sphere, radius ≈ 4× scene radius (Luma uses 4.5×,
  INRIA hierarchical 10×). Compute scene radius from keyframe camera bounds.
- Means frozen (`requires_grad=False`); SH0 color only (init 0.7); opacity
  logit init at 0.5; scales from kNN mean distance on the sphere.
- Keep dome params in a separate ParameterDict so densify/prune/big-point
  culls never touch them (this is the step everyone gets wrong).
- Render as a second rasterization pass whose output becomes the background:
  `background = dome_rgb + (1 - dome_alpha) * bg_color`, then
  `rgb = rgb_raw + (1 - alpha) * background`.
- Add regsplatfacto's regularizers: dome color uniformity
  `0.01 * mean|f_dc - detach(mean f_dc)|`, and (with a sky mask) the linear
  alpha pair `mean(alpha[sky]) + (1 - mean(alpha[non_sky]))` plus
  `0.002 * mean|bg(non_sky) - detach(mean bg(sky))|`.
- Exclude the dome entity from the Rerun trajectory-distance cull by
  logging it to its own path (`world/sky_dome`).

### 7. Alpha gradient in backward.cu (~1 day incl. gradcheck)

`cuda_rasterizer/backward.cu` already carries an internal `dL_dalpha` and
propagates it to opacity (`dL_dopa += (1 - accum_alpha_rec) * dL_dalpha`),
currently seeded only from the depth term. Thread a `dL_dout_alpha` input
through `rasterize_points.cu` / `rasterizer_impl.cu` / `backward.cu` and
seed `dL_dalpha += dL_dout_alpha[pix_id]` inside the `if (inside)` block.
Verify with finite differences.

Interim alternative without CUDA: rasterize a second pass with
`colors_precomp = torch.ones(N, 3)`; the resulting "color" image equals
accumulated alpha and is differentiable through the existing color path
(costs one extra rasterization per view).

### 8. Sky alpha barrier (~½ day, needs 5 + 7)

In `gs_backend.map()`:

```python
acc = alpha.clamp(1e-6, 1 - 1e-6)
sky_loss = torch.where(
    sky_mask,
    -torch.log(1 - acc),                                   # sky: push alpha -> 0
    -(acc * acc.log() + (1 - acc) * (1 - acc).log()),      # elsewhere: entropy only
).mean()
loss_mapping += 0.05 * sky_loss
```

λ = 0.05 is near-universal. Expect to need per-camera weights (Street
Gaussians ships `[1, 1, 0]` because side-camera masks are worse).

### Optional endgame: trainable env map (~3 days, needs 7)

Skip nvdiffrast (no aarch64 conda package; builds CUDA/GL extensions).
Use AD-GS's pure-PyTorch equirect map: a `1024x2048x3` parameter grid,
`rgb = sigmoid(grid_sample(grid, dir_to_equirect(viewdirs)))`, composited
like the dome background. Only worth it after 1–6 are in and measured.

## Traps (all observed in the wild)

- Do not tighten global big-splat pruning to reach the sky — it eats
  legitimate far road/building geometry. Exempt far splats instead.
- Do not use two-sided `-log(alpha)` on non-sky pixels (kills cables/thin
  structures inside mask errors).
- Do not use a relative depth-consistency threshold (scale-free ⇒ no far
  suppression). Absolute threshold + median cap.
- Do not expect gsplat-MCMC to fix this: it has no scale-based pruning.
- Do not trust Metric3D confidence in sky without checking: it is
  unsupervised there (no LiDAR GT above the horizon).
- nvdiffrast is the only genuinely risky aarch64 dependency in this space.

## Validation protocol

For every step: full `pixi run demo` on seq 100613, then:

1. `pixi run ate` — ATE RMSE must stay ≤ ~0.43 m (baseline band
   0.37–0.43 across seeds).
2. Keyframe render eval (printed at end of demo) — PSNR/SSIM/LPIPS must
   stay ≥ baseline − run-to-run noise (PSNR 27.3–27.5 observed).
3. Gaussian count and .rrd size (baseline 159,871 / ~125 MB).
4. Visual: headless rerun viewer screenshot at frame 197 (see
   `docs/`/report tooling from the port session; the recording's final
   blueprint already frames the corridor). Success = street corridor
   readable from orbit without the display-side cull doing the work —
   i.e. also log one snapshot with culling disabled and compare.
5. Render-vs-GT rows in the recording: sky region must not regress
   (step 6+ should make sky in renders *better*, not black).

## References

- MVSAnywhere (sky head + regsplatfacto dome + losses):
  https://github.com/nianticlabs/mvsanywhere —
  `src/regsplatfacto/regsplatfacto/regsplatfacto_model.py`,
  `src/mvsanywhere/experiment_modules/sr_depth_model.py`, `hubconf.py`
- LichtFeld-Studio (mask modes, far-field decay, MoGe-2 preprocessing):
  https://github.com/MrNeRF/LichtFeld-Studio —
  `src/training/kernels/mask_preprocess.cu`,
  `src/training/strategies/mrnf.*`, issues #907, #1377, PRs #1314, #1839
- Street Gaussians (cubemap + ablation, per-cam sky weights):
  https://github.com/zju3dv/street_gaussians
- DriveStudio/OmniRe (EnvLight cubemap + BCE):
  https://github.com/ziqipang/DriveStudio (`models/modules.py`,
  `models/trainers/base.py`)
- StreetCrafter (entropy-form sky loss; the `-log(O)` regression note is in
  its git history at `train.py`): https://github.com/zju3dv/street_crafter
- GlORIE-SLAM / Splat-SLAM (`update_valid_depth_mask`, 3×median cap);
  DROID-Splat (BA-weight confidence, `conf_th = 0.1`, `src/depth_video.py`)
- MoGe-2: https://github.com/microsoft/MoGe (sky = inf + validity mask)
- Luma skybox evidence: capture API `.../captures/<uuid>/public` returns
  `skybox` (6×384² cubemap strip) + `skybox_meta`
  (`{"type": "sphere", "distance": 1000.0, ...}`)
- Sky ablation number: Street Gaussians suppl. Table 6 — with cubemap
  32.63/0.928/0.083 vs without 31.12/0.921/0.100 (≈ +1.5 dB)
