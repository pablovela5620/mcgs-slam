"""GlobalBuffer stores every camera's keyframes, camera 0 included."""

from types import SimpleNamespace

import torch
from torch.multiprocessing import Value

from global_buffer import GlobalBuffer


def _fake_video(n_cams: int, n_kf: int, ht: int = 16, wd: int = 24) -> SimpleNamespace:
    h8, w8 = ht // 8, wd // 8
    return SimpleNamespace(
        counter=Value("i", n_kf),
        total_counter=n_kf,
        kf_stamps={i: float(i) for i in range(n_kf)},
        poses=torch.zeros(n_kf, 7),
        images=torch.full((n_kf, 3, h8, w8), 1, dtype=torch.uint8),
        disps=torch.ones(n_kf, h8, w8),
        fmaps=torch.zeros(n_kf, n_cams, 128, h8, w8, dtype=torch.half),
        nets=torch.zeros(n_kf, n_cams, 128, h8, w8, dtype=torch.half),
        inps=torch.zeros(n_kf, n_cams, 128, h8, w8, dtype=torch.half),
        images_list=[torch.full((n_kf, 3, h8, w8), c + 1, dtype=torch.uint8) for c in range(n_cams)],
        disps_list=[torch.full((n_kf, h8, w8), float(c + 1)) for c in range(n_cams)],
    )


def test_fill_global_data_keeps_every_camera_including_camera_zero() -> None:
    n_cams, n_kf = 3, 4
    video = _fake_video(n_cams, n_kf)
    args = SimpleNamespace(multi=n_cams, vis=False, ht=16, wd=24, output="/tmp")
    buf = GlobalBuffer(video, args, n_cams)

    buf.fill_global_data()

    assert buf.offset.value == n_kf and video.counter.value == 0
    for cam in range(n_cams):
        assert buf.images_all_list[cam].shape[0] == n_kf
        assert int(buf.images_all_list[cam][0, 0, 0, 0]) == cam + 1
        assert float(buf.disps_all_list[cam][0, 0, 0]) == cam + 1
    # camera-0 aliases follow the grown tensors, not the initial empty ones
    assert buf.images_all is buf.images_all_list[0]
    assert buf.disps_all is buf.disps_all_list[0]
