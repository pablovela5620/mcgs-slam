"""Shared import-path setup for the repository's flat module layout."""

import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[1]

for path in (
    ROOT,
    ROOT / "mcgs_slam",
    ROOT / "thirdparty" / "lietorch",
    ROOT / "thirdparty" / "simple-knn",
    ROOT / "thirdparty" / "diff-gaussian-rasterization",
):
    sys.path.insert(0, str(path))
