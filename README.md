# UNF-Net

Python research project scaffold for 3D brain tumor segmentation. The model,
training, evaluation, and inference implementations have not been added yet.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Update `configs/base_config.yaml` with your experiment settings before running
future experiments. Raw NIfTI files belong in `data/`; that directory is
intentionally excluded from version control.

## Project layout

- `data/` — raw NIfTI files (not tracked by Git)
- `datasets/` — data loading code
- `models/encoder/` — encoder components
- `models/decoder/` — segmentation decoder
- `models/anfis/` — differentiable ANFIS head
- `models/biomarker/` — biomarker extraction components
- `losses/` — compound loss functions
- `utils/` — shared utilities
- `configs/` — experiment configurations
- `train.py`, `evaluate.py`, `infer.py` — future workflow entry points

## BraTS 2021 loader

Place each subject in its own directory under `data/`, with files named
`t1.nii.gz`, `t1ce.nii.gz`, `t2.nii.gz`, `flair.nii.gz`, and `seg.nii.gz`.
After installing the requirements, inspect a sample with:

```bash
python -m datasets.test_loader --data-dir data
```
