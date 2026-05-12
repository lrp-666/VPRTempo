# VPRTempo — Agent Guide

> VPRTempo: A Fast Temporally Encoded Spiking Neural Network for Visual Place Recognition  
> Queensland University of Technology, Centre for Robotics  
> Version: 1.1.11 (setup.py) / 1.1.10 (pixi.toml)

---

## Project Overview

VPRTempo is a PyTorch-based spiking neural network (SNN) for Visual Place Recognition (VPR). It uses temporally encoded spikes and custom learning rules (STDP + intrinsic threshold plasticity) to match query images against a database of reference places. The architecture is derived from [BLiTNet](https://arxiv.org/pdf/2208.01204.pdf) and adapted to the [VPRSNN](https://github.com/QVPR/VPRSNN) framework.

The repository provides two variants:
- **VPRTempo** — Base fp32 model.
- **VPRTempoQuant** — Quantization-Aware Training (QAT) enabled int8 model.

Supported datasets out of the box: Nordland (spring/fall/summer/winter) and Oxford RobotCar (sun/rain/dusk). Custom datasets are supported by providing image folders and CSV annotation files.

---

## Technology Stack

- **Language:** Python >=3.6, <3.13
- **Deep Learning:** PyTorch >=2.4.0, torchvision >=0.19.0
- **Scientific:** numpy >=1.26.2 (<2), pandas >=2.2.2
- **Utilities:** tqdm, prettytable, matplotlib, requests, imageio
- **Dependency Manager:** [pixi](https://pixi.sh) (primary); conda/micromamba and pip are alternatives
- **License:** MIT (note: `vprtempo/src/metrics.py` is GPL v3, imported from Stefan Schubert)

---

## Repository Layout

```
├── main.py                      # CLI entry point; argument parser & orchestration
├── pixi.toml                    # Pixi manifest (dependencies, tasks, environments)
├── setup.py                     # setuptools config for PyPI distribution
├── requirements.txt             # Plain pip requirements (not recommended)
├── vprtempo/
│   ├── __init__.py              # Package init; exposes __version__
│   ├── VPRTempo.py              # Inference model (fp32)
│   ├── VPRTempoTrain.py         # Training model (fp32)
│   ├── VPRTempoQuant.py         # Inference model (int8 QAT)
│   ├── VPRTempoQuantTrain.py    # Training model (int8 QAT)
│   ├── src/
│   │   ├── blitnet.py           # Core SNNLayer (weights, STDP, threshold plasticity)
│   │   ├── dataset.py           # CustomImageDataset, ProcessImage, patch normalization
│   │   ├── loggers.py           # model_logger() and model_logger_quant()
│   │   ├── metrics.py           # VPR metrics: recallAtK, createPR, recallAt100precision
│   │   ├── demo.py              # Matplotlib animation demo
│   │   ├── download.py          # Dropbox downloader for pretrained models/datasets
│   │   ├── nordland.py          # Nordland dataset unzip/rename helper
│   │   ├── create_data_csv.py   # Generates CSV annotations from a folder of images
│   │   └── process_orc.py       # Oxford RobotCar preprocessing helper
│   ├── dataset/                 # CSV annotation files (nordland-*.csv, orc-*.csv)
│   └── models/                  # Pretrained model storage (.gitkeep + README.txt)
├── tutorials/                   # Jupyter notebooks (EN + CN): BasicDemo, Training, Modules
├── .github/workflows/main.yml   # PyPI publish on GitHub release
└── orc_list.txt                 # Oxford RobotCar run identifiers
```

---

## Build, Run & Development Commands

### Primary: pixi

Install pixi if missing: `curl -fsSL https://pixi.sh/install.sh | bash`

```bash
# Run the pretrained demo (downloads ~600 MB of models + images)
pixi run demo

# Train / evaluate the base fp32 model
pixi run train
pixi run eval

# Train / evaluate the quantized int8 model
pixi run train_quant
pixi run eval_quant

# Replicate conference results (Nordland)
pixi run nordland_train
pixi run nordland_eval

# Replicate conference results (Oxford RobotCar)
pixi run oxford_train
pixi run oxford_eval
```

Pixi environments:
- Default: CPU-only PyTorch
- `cuda`: `pixi run --environment cuda <task>` enables CUDA 12 support (Linux x86_64 only)

### Alternative: conda/micromamba

```bash
micromamba create -n vprtempo -c conda-forge vprtempo
micromamba activate vprtempo
```

### Direct Python execution

```bash
python main.py --train_new_model   # train
python main.py                     # evaluate
python main.py --quantize          # evaluate quantized
```

---

## CLI Arguments (main.py)

Key arguments exposed via `argparse`:
- `--dataset` — dataset name (`nordland`, `orc`, or custom)
- `--data_dir` — base dataset directory (default: `./vprtempo/dataset/`)
- `--database_places` / `--query_places` — number of reference/query images
- `--max_module` — max images per module (default 500); larger databases are split into multiple modules
- `--database_dirs` / `--query_dir` — comma-separated folder names
- `--dims` — image resize dimensions, e.g. `56,56`
- `--patches` — patch size for patch normalization (default 15)
- `--filter` — frame subsampling step (default 8)
- `--epoch` — training epochs (default 4)
- `--GT_tolerance` — ground-truth diagonal tolerance in pixels
- `--skip` — images to skip at start
- `--train_new_model` — train instead of infer
- `--quantize` — use QAT variant
- `--PR_curve` — output Precision-Recall curve
- `--sim_mat` — plot similarity matrix and ground truth
- `--run_demo` — run the animated demo

---

## Code Organization & Module Divisions

### Core Model Architecture

Each model (`VPRTempo`, `VPRTempoTrain`, `VPRTempoQuant`, `VPRTempoQuantTrain`) is a `torch.nn.Module` with two dynamically added layers:

1. **feature_layer** — `input -> feature` (feature = input * 2)
2. **output_layer** — `feature -> output` (output = places per module)

Layers are added via `add_layer(name, **kwargs)`, which instantiates `bn.SNNLayer` from `blitnet.py`.

### Training vs Inference

- Training classes (`VPRTempoTrain`, `VPRTempoQuantTrain`) contain:
  - STDP weight updates (`calc_stdp` in `blitnet.py`)
  - Intrinsic threshold plasticity (ITP)
  - Learning rate annealing (`_anneal_learning_rate`)
  - Spike forcing on the output layer to associate images with output neurons
- Inference classes (`VPRTempo`, `VPRTempoQuant`) strip learning machinery and run a simple `nn.Sequential` forward pass across modules.

### Multi-Module Support

When `database_places > max_module`, the codebase splits the database into multiple model instances (modules). Each module is trained independently, then at inference outputs are concatenated. Models are saved/loaded as combined state dicts keyed by `model_0`, `model_1`, etc.

### Image Pipeline (`dataset.py`)

`ProcessImage` applies:
1. RGB -> grayscale
2. Gamma correction
3. Resize to `dims`
4. Patch normalization (`PatchNormalisePad`) — local mean/std normalization over sliding patches
5. Scale to uint8 and encode as spikes (`SetImageAsSpikes`)

For quantized training, `SetImageAsSpikes` includes a `FakeQuantize` observer.

### Logging (`loggers.py`)

Loggers write to `./vprtempo/output/<DDMMYY-HH-MM-SS>/logfile.log` and echo to console. They print an ASCII banner and device info at startup.

---

## Code Style Guidelines

- **License header:** Every Python file starts with the MIT license blurb and a short `Imports` comment block.
- **Docstrings:** Google-style or reST-style param/type docs are used in public methods.
- **Progress bars:** All long-running loops use `tqdm`.
- **Results tables:** Evaluation prints a `PrettyTable` with Recall@1,5,10,15,20,25.
- **Device handling:** Explicit checks for `cuda:0`, `mps`, and `cpu`. Note: MPS forces `num_workers=0` in DataLoader.
- **Model naming convention:** `<database_dirs>_VPRTempo[_Quant]_IN<input>_FN<feature>_DB<places>.pth`
- **Git ignore:** `__pycache__/`, dataset image folders, `*.pth`, `vprtempo/output/`, `.vscode/`, `.DS_Store`

---

## Testing & Evaluation

There is **no formal unit-test suite** (no pytest, unittest, or CI test jobs). Correctness is assessed via:

- **Recall@K** (`recallAtK` in `metrics.py`) for K = 1, 5, 10, 15, 20, 25
- **Precision-Recall curves** (`createPR` in `metrics.py`)
- Ground-truth matrices are identity matrices with optional diagonal tolerance (`GT_tolerance`) and skip offsets

When adding new features, verify by running:
```bash
pixi run eval --dataset <dataset> --database_dirs <dirs> --query_dir <dir>
```
and confirming Recall@K values are sensible.

---

## Deployment & Distribution

- **PyPI:** Published automatically by `.github/workflows/main.yml` on GitHub release creation using `setup.py sdist bdist_wheel` + `twine upload`.
- **Conda:** Available on `conda-forge`.
- **Pixi:** Lockfile (`pixi.lock`) is committed for reproducible solves.

---

## Security Considerations

- The demo auto-downloads pretrained models and images from Dropbox URLs hardcoded in `vprtempo/src/download.py`. If modifying these URLs, ensure they are trustworthy.
- `torch.load` is called with `weights_only=True` in inference paths.
- No secrets, API keys, or credentials are stored in the repository.

---

## Custom Dataset Quick Reference

1. Place images in a folder under `./vprtempo/dataset/<name>/`.
2. Generate a CSV with `vprtempo/src/create_data_csv.py` (update the `dataset_name` variable).
3. Train: `pixi run train --dataset <name> --database_dirs <name>`
4. Evaluate: `pixi run eval --dataset <name> --database_dirs <name> --query_dir <name>`

---

## Useful Links

- Project page: https://vprtempo.github.io
- Paper (IEEE ICRA 2024): https://ieeexplore.ieee.org/document/10610918
- Repository: https://github.com/QVPR/VPRTempo
