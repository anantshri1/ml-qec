# Phase 0: Setup (lab notes)

Oct 2026. Spec lives in the roadmap doc; this note holds the reasoning, the bugs and what I understood. Items marked **TODO** need filling in from my own terminal.

## 1. Goal and plan

Goal: a repo and environment where every path has one definition, code runs identically on my Mac and in a fresh Colab runtime, and a test suite guards the path patterns.

Plan agreed before any code (blocks, one commit each):

| Block | What |
|---|---|
| 0.1 | Repo skeleton, `.gitignore`, README stub |
| 0.1b | Git LFS for large artifacts |
| 0.2 | `pyproject.toml`, `src/qec`, editable install |
| 0.3a/b/c | `paths.py`: roots, data-side paths, run-side paths |
| 0.4 | Colab bootstrap notebook |
| 0.5 | `tests/test_paths.py` and the gate |

**Gate:** `from qec.paths import DRIVE_ROOT` works locally and in a fresh Colab runtime; tests pass in both; everything pushed.

## 2. Concepts (with the code that used them)

- **One installable package beats copy-pasted notebook code.** A fix is made once; notebooks only `import qec`.
- **`src/` layout.** The package is importable only once installed, so a packaging mistake fails on my machine rather than later in a fresh runtime.
- **Editable install (`pip install -e`).** Installs a link to `src/qec` instead of a copy. Mechanically it drops an `__editable__.qec-0.1.0.pth` file in `site-packages` containing the path to `src/`.
- **`.pth` files and `sys.path`.** Python builds `sys.path` once, at interpreter startup, reading `.pth` files then. A new `.pth` is invisible to an already-running process.
- **Git LFS.** Git commits a small pointer file (hash and size) and the real content goes to LFS storage. LFS tracks by path pattern, not by file size.
- **Pure path functions.** Each returns a `Path` and never touches the disk. Only `ensure_dirs()` creates folders.
- **Tests as spec.** A test is a `test_*` function whose `assert`s must hold. Here they pin every path to the exact roadmap string.

## 3. Design decisions

- **Repo public from day one.** Portfolio piece. Rule: tokens and personal data never committed.
- **Compute split.** Local (VS Code) for everything except GNN training and GPU latency runs, which use Colab. Code flows one way: edit and commit locally, push, Colab clones or pulls. Public repo, so Colab needs no token and never pushes.
- **Shared venv** (the one for my other ML projects) instead of a dedicated one. Safeguards: `pip freeze` snapshot before installing, and `pip install --dry-run` to see changes first. Dry run showed only additions (PyMatching 2.4.0, sinter 1.16.0, stim 1.16.0, qec 0.1.0). Risk accepted: a shared venv can hide a missing dependency; the fresh-Colab gate catches that.
- **Dependencies.** Light ones in `pyproject.toml` (numpy, pyyaml, stim, pymatching, sinter, matplotlib). Torch stays as Colab ships it; `torch_geometric` is installed in the bootstrap. Versions unpinned until Phase 3.
- **`DRIVE_ROOT` name kept** even though locally it is not on Drive. Resolution order: `QEC_ROOT` env var, then mounted Colab Drive, then `~/ml-qec-data`.
- **Artifacts in git, with a size rule.** I chose not to gitignore `.npz` and `.pt`, for reproducibility. Compromise: `artifacts/small/` (under about 50 MB, plain git) and `artifacts/lfs/` (larger, tracked by `artifacts/lfs/**`). Folder-based because LFS cannot enforce a size threshold. Rule committed before any large file, to avoid `git lfs migrate` and a history rewrite.
- **LFS quota.** GitHub's docs disagree: older pages say 1 GiB storage and 1 GiB/month bandwidth, current billing docs say 10 GiB each on Free. **TODO:** check my own billing page. Per-file cap 2 GB on Free.
- **Colab clones with `GIT_LFS_SKIP_SMUDGE=1`,** then `git lfs pull --include=...` for only the files needed, because LFS downloads count against the owner's bandwidth.
- **`canonical_numbers.txt`** at `results/canonical_numbers.txt`, committed, exposed as `paths.CANONICAL_NUMBERS`. Format decided in Phase 3 when the first number arrives.
- **`p_tag` uses `f"p{p:g}"`,** one formatter for all paths (0.001 gives `p0.001`, 1e-5 gives `p1e-05`).
- **`make_run_name`** is generated once at launch and stored in the config; a resume must reuse the stored name.

## 4. Code map

`src/qec/paths.py`
- Repo-side constants: `REPO_ROOT` (from `__file__`, `parents[2]`), `CONFIGS_DIR`, `RESULTS_DIR`, `FIGURES_DIR`, `CANONICAL_NUMBERS`, `ARTIFACTS_SMALL`, `ARTIFACTS_LFS`.
- `_resolve_drive_root()` and `DRIVE_ROOT`, plus `DATA_DIR`, `DEMS_DIR`, `BASELINES_DIR`, `CHECKPOINTS_DIR`, `RUNS_DIR`, `BENCHMARKS_DIR`; `ensure_dirs()`.
- Helpers: `NOISES = ("si1000", "corr")`, `_check_noise`, `p_tag(p)`, `point_tag(d, rounds, p)`.
- Data side: `shard_path(noise, d, rounds, p, k)`, `dem_path(noise, d, rounds, p)`, `baseline_csv(decoder, noise)`.
- Run side: `make_run_name(model, noise, d, when=None)`, `config_path`, `run_dir`, `run_config_copy`, `metrics_csv`, `checkpoint_path(run_name, epoch)`, `last_checkpoint`, `benchmark_dir`, `ler_csv`, `latency_csv`.

`notebooks/00_bootstrap.ipynb`: cell 1 mounts Drive; cell 2 (`%%bash`) pulls or clones with LFS skipped and installs; cell 3 re-reads `.pth` files, prints versions, asserts `DRIVE_ROOT` is under `/content/drive`, then `ensure_dirs()`.

`tests/test_paths.py`: 8 tests (patterns, validation, repo root, env override, purity, `ensure_dirs` idempotence). Override and purity tests use subprocesses via `run_py`.

Gotchas to remember:
- Mount Drive before importing `qec.paths` on Colab, or `DRIVE_ROOT` silently falls back to a temporary local folder.
- Any path change is made in `paths.py` first, then the roadmap and the tests.
- Phase 5 may add a second noise parameter, which would extend `point_tag`.

## 5. Debugging log

### Bug 1: `ModuleNotFoundError: No module named 'qec'` on Colab (cell 3)
- **Symptom:** versions printed fine, then `from qec import paths` failed right after cell 2's editable install.
- **Hypotheses:** H1: the kernel started before the `.pth` file existed, so it never saw it. H2: the install silently failed.
- **Probe 1:** `pip show qec` and listing `dist-packages`. It showed `qec 0.1.0` with editable location `/content/ml-qec` and `__editable__.qec-0.1.0.pth` present, which refutes H2. It then crashed with `FileNotFoundError` because `site.getsitepackages()` returned a directory that does not exist on Colab. That crash was a bug in the probe, not in my setup.
- **Probe 2:** guarded with `os.path.isdir`, re-read `.pth` files with `site.addsitedir`. Result: `before: []`, `after: ['/content/ml-qec/src']`, import works. **H1 confirmed.**
- **Fix:** four lines at the top of cell 3 (`site.addsitedir` loop plus `importlib.invalidate_caches()`).
- **Why in cell 3, not cell 2:** cell 2 is `%%bash`, a separate process that cannot touch the kernel's `sys.path`. The fix has to be Python running in the kernel after the install.
- **Lesson:** a long-lived kernel does not re-read `.pth` files. On my Mac every `python` command is a fresh process, so the problem never appeared. Rejected shortcut: `sys.path.insert(...)`, which would bypass the install and hide real packaging bugs. Also, write probes defensively.

### Bug 2: `no tests ran in 0.01s` on Colab
- **Symptom:** pytest found `tests/` but collected nothing.
- **Hypothesis:** stale clone from before the push, since cell 2 only pulls when rerun.
- **Resolution:** after refreshing, `8 passed`. **TODO:** confirm that the stale clone really was the cause.

## 6. Q&A and explain-backs

Format: my words, then what was sharpened.

- **Why `.gitignore` excluded `.npz` and `.pt` / why repo and data root are separate.** I did not answer; I overrode the ignore rule instead, for reproducibility. This led to the LFS compromise above.
- **LFS rule before the large file.** *Mine:* a file committed first and tracked later stays as a full blob, and fixing it is a pain; git commits a small text pointer and the content goes to LFS. *Sharpened:* the pointer holds the SHA-256 and size; the rule must exist before `git add` because the swap to a pointer happens then.
- **`pip install -e` vs `pip install .`, and `src/`.** *Mine:* `-e` installs a link to `src/qec` instead of a copy, so edits take effect immediately; with `src/`, packaging mistakes fail on the machine instead of in a fresh runtime. *Sharpened:* in a flat layout `import qec` works from the repo root even with broken packaging; with `src/` it works only if the install worked. Also, `REPO_ROOT` via `parents[2]` relies on the editable install; a regular install would point into site-packages.
- **`REPO_ROOT` from `__file__`, and the Colab risk.** *Mine:* so it works without ambiguity wherever Python runs from; on Colab, if Drive is not mounted it falls back to a local folder and everything vanishes on disconnect. *Confirmed.*
- **Why `p_tag` exists.** *First try:* echoed the explanation back, not my own. *Second:* without it we would have two different files, shard and DEM paths would silently disagree, and things would seem to work but give wrong answers. *Sharpened:* the more common failure is the skip-if-exists logic not finding the file and silently re-sampling (wasted compute). Wrong answers need stale data at the wrongly spelled path that then gets loaded.
- **`make_run_name`.** *Mine:* call it once, store the name, never regenerate. *Missing, added:* a resume past midnight would get a new date, look for checkpoints in a folder that does not exist, and silently train from scratch while the real checkpoints sit under the old name.
- **`GIT_LFS_SKIP_SMUDGE=1`.** *Mine:* makes clone and pull fetch pointer files so we do not waste LFS bandwidth. *Confirmed.*
- **What the `DRIVE_ROOT` assert protects against.** *Mine:* it tells us if Drive is not mounted, and makes sure data does not vanish. *Sharpened:* it does not fix anything; it turns a silent failure into a loud one in the first minute rather than after hours of lost work.
- **Mac vs Colab and `.pth`.** *Mine:* on the Mac we install first and then run, so a fresh process reads the `.pth` on startup; on Colab the kernel started and built its path before the package existed, and does not re-read, so the `.pth` is not in memory. *Confirmed.*
- **Why override and purity tests use subprocesses.** *Mine:* `DRIVE_ROOT` is fixed at first import; `run_py` starts a fresh Python each time, re-importing `paths` and re-computing `DRIVE_ROOT`; real data is never touched. *Confirmed:* `QEC_ROOT` points at pytest's temporary folder, so even `ensure_dirs()` only creates folders there.

## 7. Gate checklist and environment

| Gate item | Status |
|---|---|
| `pytest` passes locally | 8 passed |
| `pytest` passes on Colab | 8 passed |
| `from qec.paths import DRIVE_ROOT` locally | works (`~/ml-qec-data`) |
| Same in fresh Colab runtime, no restart | works (`/content/drive/MyDrive/ml-qec`) |
| Guard test (skip the mount) | `AssertionError` raised as intended |
| Everything pushed | **TODO:** `git status` clean, `git log --oneline -8` shows 0.1 to 0.5 |

Commit hashes: **TODO** (paste `git log --oneline -8`).

Environment, local: macOS, Python 3.13.4, pytest 9.1.1, stim 1.16.0, pymatching 2.4.0, sinter 1.16.0 (shared venv).
Environment, Colab (CPU runtime, Python 3.13): numpy 2.1.3, torch 2.11.0+cpu, torch_geometric 2.8.0.post1, stim 1.16.0, pymatching 2.4.0, sinter 1.16.0.

## 8. Open items carried forward

- Update the roadmap doc: compute split, `artifacts/` and LFS rule, `canonical_numbers.txt`, bootstrap `.pth` fix, version table.
- Phase 3: decide the `canonical_numbers.txt` format and pin versions.
- Phase 4: decide how data reaches Colab (Drive for Desktop, zipped shards, or sampling on Colab); install CPU torch and `torch_geometric` locally for testing `graphs.py` and `models.py`; `train.py` must read the stored run name, never regenerate it.
- Phase 5: a correlated noise parameter may extend `point_tag`; change `paths.py` first, then the roadmap and tests.
- Phase 7: latency tables record the machine (CPU, GPU, batch size) next to each number.