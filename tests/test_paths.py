import os
import subprocess
import sys
from datetime import date

import pytest

from qec import paths as P


def rel(path):
    """Path relative to DRIVE_ROOT as a posix string, so tests ignore the machine."""
    return path.relative_to(P.DRIVE_ROOT).as_posix()


def run_py(code, root):
    """Run code in a fresh interpreter with QEC_ROOT=root; return stdout."""
    env = {**os.environ, "QEC_ROOT": str(root)}
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_p_tag():
    assert P.p_tag(0.001) == "p0.001"
    assert P.p_tag(1e-5) == "p1e-05"


def test_data_side_patterns():
    assert rel(P.shard_path("si1000", 5, 10, 0.001, 3)) == "data/si1000/d5_r10_p0.001/shard_0003.npz"
    assert rel(P.dem_path("corr", 5, 10, 0.001)) == "dems/corr/d5_r10_p0.001.dem"
    assert rel(P.baseline_csv("pymatching", "si1000")) == "baselines/pymatching/si1000.csv"


def test_run_side_patterns():
    n = P.make_run_name("gnn", "si1000", 3, date(2026, 10, 6))
    assert n == "gnn_si1000_d3_20261006"
    assert rel(P.checkpoint_path(n, 7)) == f"checkpoints/{n}/epoch_007.pt"
    assert rel(P.last_checkpoint(n)) == f"checkpoints/{n}/last.pt"
    assert rel(P.run_config_copy(n)) == f"runs/{n}/config.yaml"
    assert rel(P.metrics_csv(n)) == f"runs/{n}/metrics.csv"
    assert rel(P.ler_csv(n)) == f"benchmarks/{n}/ler.csv"
    assert rel(P.latency_csv(n)) == f"benchmarks/{n}/latency.csv"
    assert P.config_path("gnn_d3") == P.REPO_ROOT / "configs" / "gnn_d3.yaml"


def test_bad_noise_raises():
    with pytest.raises(ValueError):
        P.shard_path("si1000x", 5, 10, 0.001, 0)
    with pytest.raises(ValueError):
        P.dem_path("si1000x", 5, 10, 0.001)
    with pytest.raises(ValueError):
        P.baseline_csv("pymatching", "si1000x")
    with pytest.raises(ValueError):
        P.make_run_name("gnn", "si1000x", 3)


def test_repo_root_is_the_repo():
    assert (P.REPO_ROOT / "pyproject.toml").is_file()
    assert P.CANONICAL_NUMBERS.parent == P.REPO_ROOT / "results"


def test_env_override(tmp_path):
    out = run_py("from qec import paths; print(paths.DRIVE_ROOT)", tmp_path)
    assert out == str(tmp_path)


def test_path_functions_are_pure(tmp_path):
    code = ("from qec import paths as P; "
            "P.shard_path('si1000',5,10,0.001,0); P.dem_path('corr',5,10,0.001); "
            "P.baseline_csv('pymatching','si1000'); P.checkpoint_path('r',1); "
            "P.last_checkpoint('r'); P.run_dir('r'); P.ler_csv('r'); P.latency_csv('r')")
    run_py(code, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_ensure_dirs_is_idempotent(tmp_path):
    run_py("from qec import paths as P; P.ensure_dirs(); P.ensure_dirs()", tmp_path)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["baselines", "benchmarks", "checkpoints", "data", "dems", "runs"]