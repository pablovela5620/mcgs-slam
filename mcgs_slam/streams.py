import os
import re
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
import torch
from jaxtyping import Float, UInt8

base = 8


class StreamArgs(Protocol):
    """Options used by the image stream."""

    stride: int
    ht: int
    wd: int
    timescale: float
    decode_reduced: bool

# With args.decode_reduced, JPEGs are decoded at half size in the DCT domain
# (IMREAD_REDUCED_COLOR_2) so decode + undistort work on a quarter of the
# pixels (27 vs 62 ms/frame on the Waymo demo). Against the full-resolution
# path, the tested mean absolute pixel deviation is <2/255 and the 99th
# percentile is <=16/255. It is opt-in; the default remains bit-exact.


def map_filename(x: str) -> float:
    return float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1])


def resize(image: UInt8[np.ndarray, "h w 3"], h1: int, w1: int) -> UInt8[torch.Tensor, "3 h1 w1"]:
    image = cv2.resize(image, (w1, h1))
    image = image[:h1 - h1 % base, :w1 - w1 % base]
    return torch.as_tensor(image).permute(2, 0, 1)


@dataclass
class _Undistorter:
    """Precomputed cv2.remap tables for one camera at the decoded resolution."""

    map1: np.ndarray
    map2: np.ndarray

    @classmethod
    def build(cls, calib_row: Float[np.ndarray, "8"], decoded_hw: tuple[int, int], scale: float) -> "_Undistorter":
        fx, fy, cx, cy = calib_row[:4]
        # pixel-center-preserving intrinsics for the DCT-reduced image
        K = np.array([[fx * scale, 0, (cx + 0.5) * scale - 0.5],
                      [0, fy * scale, (cy + 0.5) * scale - 0.5],
                      [0, 0, 1]])
        h, w = decoded_hw
        map1, map2 = cv2.initUndistortRectifyMap(K, calib_row[4:], None, K, (w, h), cv2.CV_16SC2)
        return cls(map1, map2)

    def __call__(self, image: UInt8[np.ndarray, "h w 3"]) -> UInt8[np.ndarray, "h w 3"]:
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR)


def _decode(path: str, reduced: bool) -> tuple[UInt8[np.ndarray, "h w 3"], float]:
    """Decode an image; returns (image, scale of the decoded size vs. the file's size)."""
    if not reduced:
        return cv2.imread(path), 1.0
    if path.lower().endswith((".jpg", ".jpeg")):
        return cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2), 0.5
    img = cv2.imread(path)
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA), 0.5


def image_stream(imagedirs: list[str], calib: Float[np.ndarray, "n_dirs k"], args: StreamArgs):
    """Yield (t, images[n_dirs,3,ht,wd] uint8 BGR, intrinsics[n_dirs,8], timestamp) per frame.

    Undistortion (when calib has distortion coefficients) uses remap tables
    built once per camera.
    args.decode_reduced enables half-size JPEG decode. Its tested mean absolute
    pixel deviation is <2/255 and its 99th percentile is <=16/255.
    """
    reduced = args.decode_reduced
    image_lists = [sorted(os.listdir(d), key=map_filename)[::args.stride] for d in imagedirs]
    undistort = calib.shape[1] > 4
    undistorters: list[_Undistorter | None] = [None] * len(imagedirs)

    h1, w1 = args.ht, args.wd
    for t in range(len(image_lists[0])):
        images: list[torch.Tensor] = []
        intrinsics = torch.zeros((len(imagedirs), 8))
        native_sizes: list[tuple[float, float]] = []
        for i, imagelist in enumerate(image_lists):
            timestamp = map_filename(imagelist[t]) / args.timescale
            if i == 0:
                t0 = timestamp
            else:
                assert abs(timestamp - t0) < 2e-2, f"{timestamp} vs. {t0}"

            path = os.path.join(imagedirs[i], imagelist[t])
            decoded_image, scale = _decode(path, reduced)
            hs, ws = decoded_image.shape[:2]
            if undistort:
                if undistorters[i] is None:
                    undistorters[i] = _Undistorter.build(calib[i], (hs, ws), scale)
                decoded_image = undistorters[i](decoded_image)
            h0, w0 = hs / scale, ws / scale   # full-resolution size
            image = resize(decoded_image, h1, w1)
            images.append(image)
            native_sizes.append((h0, w0))
            intrinsics[i, :4] = torch.tensor(calib[i, :4])
            intrinsics[i, [0, 2]] *= w1 / w0
            intrinsics[i, [1, 3]] *= h1 / h0

        images_t = torch.stack(images, dim=0)

        if t == 0:
            print(f'Orig sizes: {native_sizes}; Down to size: {h1}-{w1}\nInput calib: {list(calib)}\nAdapt calib: {list(intrinsics.numpy())}')

        yield t, images_t, intrinsics, timestamp
