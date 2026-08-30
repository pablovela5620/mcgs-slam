"""Shared import-path setup for the repository's flat module layout."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcgs_slam._paths import configure_import_paths


configure_import_paths()
