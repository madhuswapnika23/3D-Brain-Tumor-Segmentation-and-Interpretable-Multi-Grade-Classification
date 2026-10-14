# unf-net

Python research project scaffold for unf-net. Model implementation has not been
added yet.

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

Edit `configs/default.yaml` with the paths to your data and the experiment
settings before running future experiments.

## Project layout

- `data/` — local datasets (not tracked by Git)
- `src/` — Python package source
- `configs/` — experiment configuration
- `experiments/` — experiment scripts and outputs
- `notebooks/` — research notebooks
- `tests/` — automated tests
