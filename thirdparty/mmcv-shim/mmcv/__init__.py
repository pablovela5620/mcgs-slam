"""Minimal mmcv stub.

Metric3D (loaded via torch.hub in motion_filter.py) has exactly one unguarded
mmcv import: `from mmcv.utils import collect_env` in mono/utils/comm.py (its
other mmcv uses fall back to mmengine). Vendoring this one-symbol shim avoids
installing mmcv itself, whose PyPI wheel would drag in a second copy of OpenCV
next to the conda-forge one.

This directory (thirdparty/mmcv-shim) exists solely to be put on sys.path, so
the stub cannot shadow a real mmcv for code that does not opt in.
"""

from . import utils  # noqa: F401
