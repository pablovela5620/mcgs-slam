"""Repository import paths needed by the flat legacy module layout."""

import sys
from pathlib import Path


REPOSITORY_ROOT: Path = Path(__file__).resolve().parents[1]


def configure_import_paths() -> None:
    """Prepend the repository's source and in-place extension directories."""
    import_paths: tuple[Path, ...] = (
        REPOSITORY_ROOT,
        REPOSITORY_ROOT / "mcgs_slam",
        REPOSITORY_ROOT / "thirdparty" / "lietorch",
        REPOSITORY_ROOT / "thirdparty" / "simple-knn",
        REPOSITORY_ROOT / "thirdparty" / "diff-gaussian-rasterization",
    )
    for import_path in reversed(import_paths):
        import_path_string: str = str(import_path)
        if import_path_string not in sys.path:
            sys.path.insert(0, import_path_string)
