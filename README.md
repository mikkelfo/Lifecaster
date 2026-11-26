# Lifecaster project

Use `conda env create -f env.yml` to install a conda environment called `lifecaster`

It installs Python 3.13 and runs pip install -e ., which pulls the dependencies from pyproject.toml (torch, lightning, polars, lmdb, ...) and puts the lifecaster package on the path. Flash-attention is left out on purpose; the tutorial uses the no-flash model.

# Layout

### checkpoints/
Where ModelCheckpoint writes weights, under checkpoints/{experiment_name}/{run_id}/, keeping best.ckpt (best val/loss) and last.ckpt. Finetuning loads best.ckpt from here. Empty until you train.

### configs/
finetune.yaml holds the settings for the script version of finetuning (finetune.py): experiment name, source files, outcome, prediction windows, model size, optimizer. We use hydra inheritance to organize things, such as providing defaults under `data`, `hardware`, and `model`. See `finetune.yaml` and `pretrain.yaml` for final versions

### logs/
Where CSVLogger writes, under logs/{experiment_name}/{run_id}/, with metrics.csv (per-step/epoch losses and metrics) and hparams.yaml. The version number matches the checkpoint folder so the two line up. Empty until you train.

### scripts/
Where scripts (pretrain and finetune) is placed

### src/
The actual logic to run the Lifecaster. It's organized into
- datamodules: Where all data logic is called from
- dataset: Fetches people from the LMDB dataset
- lightningmodules: Where all training logic is called from
- metrics: Custom metrics that isn't supported by torchmetrics
- model: The Lifecaster model code
- misc: Files that doesn't fit neatly into single categories (might be changed going forward)