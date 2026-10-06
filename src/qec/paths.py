"""Single source of truth for every path in the project.

Nothing else in the repo builds a path by hand. A new artifact path is
added here first, then recorded in the roadmap doc and in project memory.
"""
import os
from pathlib import Path

# --- Repo-side paths: code, configs, small committed results ---
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CANONICAL_NUMBERS = RESULTS_DIR / "canonical_numbers.txt"
ARTIFACTS_SMALL = REPO_ROOT / "artifacts" / "small"   # < ~50 MB, plain git
ARTIFACTS_LFS = REPO_ROOT / "artifacts" / "lfs"       # > ~50 MB, Git LFS

# --- Heavy-data root: never committed ---
COLAB_DRIVE = Path("/content/drive/MyDrive/ml-qec")
LOCAL_DEFAULT = Path.home() / "ml-qec-data"


def _resolve_drive_root() -> Path:
    """QEC_ROOT env var > mounted Colab Drive > local default."""
    env = os.environ.get("QEC_ROOT")
    if env:
        return Path(env).expanduser()
    if COLAB_DRIVE.parent.is_dir():  # /content/drive/MyDrive exists => mounted
        return COLAB_DRIVE
    return LOCAL_DEFAULT


DRIVE_ROOT = _resolve_drive_root()
DATA_DIR = DRIVE_ROOT / "data"
DEMS_DIR = DRIVE_ROOT / "dems"
BASELINES_DIR = DRIVE_ROOT / "baselines"
CHECKPOINTS_DIR = DRIVE_ROOT / "checkpoints"
RUNS_DIR = DRIVE_ROOT / "runs"
BENCHMARKS_DIR = DRIVE_ROOT / "benchmarks"


def ensure_dirs() -> None:
    """Create the top-level skeleton under DRIVE_ROOT (idempotent)."""
    for d in (DATA_DIR, DEMS_DIR, BASELINES_DIR,
              CHECKPOINTS_DIR, RUNS_DIR, BENCHMARKS_DIR):
        d.mkdir(parents=True, exist_ok=True)