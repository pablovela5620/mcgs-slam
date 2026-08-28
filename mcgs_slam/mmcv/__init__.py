"""Minimal mmcv stub.

Metric3D (loaded via torch.hub in motion_filter.py) imports a handful of
mmcv.utils helpers that all exist in mmengine. Vendoring this stub avoids
installing mmcv itself, whose PyPI wheel would drag in a second copy of
OpenCV next to the conda-forge one. A real mmcv in site-packages takes
precedence over this stub because mcgs_slam is appended to sys.path.
"""

from mmcv import utils  # noqa: F401
