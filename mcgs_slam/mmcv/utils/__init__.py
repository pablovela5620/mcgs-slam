"""mmcv.utils compatibility shims backed by mmengine (see mcgs_slam/mmcv/__init__.py)."""

from mmengine import Config, DictAction  # noqa: F401
from mmengine.utils import get_git_hash  # noqa: F401
from mmengine.utils.dl_utils import collect_env  # noqa: F401
