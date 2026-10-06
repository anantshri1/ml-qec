"""Single source of truth for every path in the project.

Nothing else in the repo builds a path by hand. A new artifact path is
added here first.
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

# --- Naming helpers: the ONLY place these patterns are written ---
NOISES = ("si1000", "corr")


def _check_noise(noise: str) -> None:
    if noise not in NOISES:
        raise ValueError(f"noise must be one of {NOISES}, got {noise!r}")


def p_tag(p: float) -> str:
    """0.001 -> 'p0.001'. One formatter, so data and DEM paths never diverge."""
    return f"p{p:g}"


def point_tag(d: int, rounds: int, p: float) -> str:
    """The 'd5_r10_p0.001' fragment shared by data and DEM paths."""
    return f"d{d}_r{rounds}_{p_tag(p)}"


# --- Data-side paths (pure: return a Path, never touch the disk) ---
def shard_path(noise: str, d: int, rounds: int, p: float, k: int) -> Path:
    _check_noise(noise)
    return DATA_DIR / noise / point_tag(d, rounds, p) / f"shard_{k:04d}.npz"


def dem_path(noise: str, d: int, rounds: int, p: float) -> Path:
    _check_noise(noise)
    return DEMS_DIR / noise / f"{point_tag(d, rounds, p)}.dem"


def baseline_csv(decoder: str, noise: str) -> Path:
    _check_noise(noise)
    return BASELINES_DIR / decoder / f"{noise}.csv"