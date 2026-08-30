"""Monocular geometry prior (metric depth + surface normals + validity) from MoGe-2.

MoGe-2 is pulled from the rerun examples monorepo (pinned in pixi.toml), which
vendors the model as pure PyTorch (DINOv2 backbone, SDPA attention) and pins the
Hugging Face checkpoints by revision.
"""
from dataclasses import dataclass
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from jaxtyping import Bool, Float, UInt8
from monopriors.third_party.moge.model.v2 import MoGeModel
from torch import Tensor

Encoder = Literal["vits", "vitb", "vitl"]

CHECKPOINTS: dict[Encoder, tuple[str, str]] = {
    "vits": ("Ruicheng/moge-2-vits-normal", "679230677b4d282c6f304189a93e98e14f085902"),
    "vitb": ("Ruicheng/moge-2-vitb-normal", "54ad3a693e61907ea4633d13dec6ee682fa09419"),
    "vitl": ("Ruicheng/moge-2-vitl-normal", "b135031bae30b5ac2ae141a0e68717795ce38340"),
}
"""Depth+normal checkpoints per encoder size, pinned to a Hugging Face revision."""


@dataclass
class PriorPrediction:
    """Per-image metric prior on the CPU. Invalid pixels (sky, undefined geometry) are zeroed."""

    depth: Float[Tensor, "n h w"]
    """Metric depth in meters; 0 where invalid."""
    normal: Float[Tensor, "n 3 h w"]
    """Unit camera-frame normals (same convention as Metric3D: x right, y down, z forward); 0 where invalid."""

    @property
    def valid(self) -> Bool[Tensor, "n h w"]:
        """Validity mask derived from the zero-depth invalid-pixel contract."""
        return self.depth > 0


class MoGePrior:
    """MoGe-2 depth + normal prior for a batch of images with known focal lengths."""

    def __init__(self, encoder: Encoder = "vits", resolution_level: int = 3, device: str = "cuda") -> None:
        """
        Args:
            encoder: DINOv2 backbone size. ``vits`` runs 3 cameras at 360x536 in ~200 ms on a GB10.
            resolution_level: MoGe token budget, 0 (fastest, ~35 ms/img) to 9 (most detail, ~120 ms/img).
                Level 3 deviates <2% in depth and a few degrees in normals from level 9.
            device: torch device for inference.
        """
        repo, rev = CHECKPOINTS[encoder]
        self.model: MoGeModel = MoGeModel.from_pretrained(repo, revision=rev).to(device).eval()
        self.resolution_level = resolution_level
        self.device = device

    @staticmethod
    def fetch(encoder: Encoder = "vits") -> str:
        """Download the pinned checkpoint into the Hugging Face cache; returns the local path."""
        repo, rev = CHECKPOINTS[encoder]
        return snapshot_download(repo, revision=rev)

    @torch.no_grad()
    def __call__(self, images_rgb: UInt8[Tensor, "n 3 h w"], fx: Float[Tensor, "n"]) -> PriorPrediction:
        """
        Args:
            images_rgb: uint8 RGB images (any device).
            fx: focal length in pixels of each image, at the image's resolution; sets MoGe's field of view
                instead of letting it estimate one.
        """
        n, _, h, w = images_rgb.shape
        x: Float[Tensor, "n 3 h w"] = images_rgb.to(self.device, non_blocking=True).float().div(255.0)
        fov_x: Float[Tensor, "n"] = torch.rad2deg(2.0 * torch.atan(w / (2.0 * fx.to(self.device).float())))
        out = self.model.infer(x, fov_x=fov_x, resolution_level=self.resolution_level, apply_mask=True)
        depth: Float[Tensor, "n h w"] = out["depth"].float()
        normal: Float[Tensor, "n h w 3"] = out["normal"].float()
        valid: Bool[Tensor, "n h w"] = out["mask"] & torch.isfinite(depth) & (depth > 0)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        normal = torch.where(valid[..., None], normal, torch.zeros_like(normal)).permute(0, 3, 1, 2).contiguous()
        return PriorPrediction(depth=depth.cpu(), normal=normal.cpu())
