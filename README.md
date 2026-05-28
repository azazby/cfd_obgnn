This repository builds on the original OB-GNN implementation from
`SolarisAdams/ML4CFD-Offset-based-Graph-Convolution`. It extends the codebase
to train and evaluate OB-GNN CFD surrogate models on AirfRANS plus bumped
airfoil variants, so experiments can measure how node-level predictions change
under localized geometric deviations.

To get started:

1. Configure the Python environment and data paths.
2. Train an OB-GNN variant with `run_obgnn.sbatch`.
3. Evaluate a saved checkpoint with `eval_preds_obgnn.sbatch`, optionally saving
   per-simulation predictions and appending node-level MSE metrics to a CSV.

## Repository layout

- `src/main.py`: training entry point. Builds/caches graph tensors, trains
  `CoordGNN`, saves epoch checkpoints, and runs standard AirfRANS test/OOD
  evaluation at the end of training.
- `src/evaluate_predictions.py`: evaluation/inference entry point. Loads a
  checkpoint, runs inference on a processed split, writes summary MSE metrics,
  and can save per-simulation prediction tensors.
- `src/bump_airfoil_dataset.py`: wraps bumped `.npz` samples and split metadata
  so they look like AirfRANS datasets.
- `src/path_config.py`: shared path configuration and CLI path overrides.
- `config/paths.ini`: default dataset/cache/output paths.
- `run_obgnn.sbatch`: SLURM training script.
- `eval_preds_obgnn.sbatch`: SLURM evaluation and prediction-saving script.
- `val_obgnn.sbatch`: example SLURM validation script that evaluates all checkpoints.
- `model_checkpoints/`: example checkpoint directories.
- `evaluation/`: summary MSE CSV outputs.
- `predictions/`: saved prediction tensors.

## Environment setup

This project was tested with Python 3.10.19.

On MIT ORCD Engaging, load conda support first:

```bash
module load miniforge
```

Create and activate the environment:

```bash
conda create -n ob_gnn python=3.10
conda activate ob_gnn
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The sbatch scripts assume this environment is named `ob_gnn` and also load
`cuda`. If you use a different environment name, update the `mamba activate`
line in the scripts.

For an interactive ORCD session:

```bash
salloc -N 1 -c 4 --mem=32G -p mit_normal_cpu --time=1:00:00
salloc -N 1 -G 1 -c 4 -p mit_normal_gpu --time=1:00:00
```

## Data and path configuration

All training and evaluation scripts read paths from `config/paths.ini` by
default. Relative paths are resolved from `base_dir`; absolute paths can point
to ORCD scratch.

Default keys:

- `base_dir`: repo root or another base used for relative paths.
- `dataset_dir`: AirfRANS `Dataset` directory.
- `processed_data_dir`: cache directory for processed DGL graph tensors.
- `bumped_dir`: directory containing bumped airfoil `.npz` files.
- `bumped_split_csv`: CSV assigning bumped samples to train/val/test splits.
- `bench_config_path`: AirfRANS/LIPS config, usually `src/confAirfoil.ini`.
- `predictions_dir`: root for saved prediction tensors.
- `evaluation_dir`: root for summary MSE CSVs.

You can edit `config/paths.ini` or override any path from the command line:

```bash
python src/main.py \
  --processed_data_dir /orcd/home/002/$USER/orcd/scratch/airfrans_data/processed \
  --bumped_dir /orcd/home/002/$USER/orcd/scratch/airfrans_data/bumped_dataset
```

To download the original AirfRANS data into `airfrans_data/Dataset`:

```bash
python download_airfrans.py
```

Bumped data is expected as `.npz` files under `bumped_dir`, with split membership
defined by `airfrans_data/bumped_dataset_split.csv`. The code uses columns such
as `train_A`, `train_B`, `train_C`, `train_D`, `train_S`, `val`, and `split`.

## Training with `run_obgnn.sbatch`

Edit the bottom of `run_obgnn.sbatch` to choose exactly one training command.
Also update `/orcd/home/002/$USER/cfd_obgnn` to the path of your checkout if it
differs. Then submit:

```bash
sbatch run_obgnn.sbatch
```

Example bumped-airfoil training:

```bash
srun python3 -u /orcd/home/002/$USER/cfd_obgnn/src/main.py \
  --epochs 20 \
  --batch_size 128 \
  --lr 1.5e-4 \
  --bump_train_version train_A \
  --checkpoint_dir ./model_checkpoints/checkpoints_A
```

Example baseline AirfRANS-only training:

```bash
srun python3 -u /orcd/home/002/$USEAR/cfd_obgnn/src/main.py \
  --epochs 12 \
  --batch_size 128 \
  --lr 1.5e-4 \
  --checkpoint_dir ./model_checkpoints/checkpoints_O
```

Important training arguments:

- `--epochs`: number of epochs per bagging round. Checkpoints are saved after
  every epoch as `ckpt_bagXX_epYYYY.pt`.
- `--batch_size`: node sampler batch size for the custom balanced training
  dataset. Larger values need more GPU memory.
- `--lr`: Adam learning rate.
- `--weight_decay`: Adam weight decay.
- `--hidden1`, `--hidden2`, `--hidden3`: hidden sizes for the model's three
  output branches. Defaults are `256`, `256`, and `128`.
- `--steps`: maximum dataloader iteration index per epoch. The loop stops once
  `it > steps`.
- `--bump_train_version`: bumped split to mix into training. Use values from the
  split CSV, for example `train_A`, `train_B`, `train_C`, `train_D`, `train_S`.
  Leave unset to train on the original AirfRANS train split only. `all_bumps`
  currently maps to the `train_D` bumped subset without the original AirfRANS
  training data.
- `--checkpoint_dir`: directory for checkpoint save/resume.
- `--bagging_k`: number of outer bagging rounds.
- `--k`: number of inner models trained per bagging round and averaged.
- `--w1` through `--w6`: weights for loss components:
  x-velocity, y-velocity, pressure, turbulent viscosity, surface pressure, and
  surface velocity error.
- `--noise1`, `--noise2`, `--noise3`: training-time feature noise injected into
  the model branches.
- `--lr_decay` and `--lr_decay_start_epoch`: StepLR decay factor and epoch where
  scheduler stepping begins.
- `--gpu`: CUDA device index.

Training automatically creates processed graph cache files in
`processed_data_dir`. For example, `--bump_train_version train_A` uses
`processed_train_A.pt`; baseline training uses `processed_train.pt`. Existing
cache files are reused, so delete the relevant processed `.pt` file if the raw
data or preprocessing logic changes.

Checkpoint resume is automatic. If `--checkpoint_dir` already contains valid
`ckpt_bagXX_epYYYY.pt` files, training resumes from the latest checkpoint.

Logs from the training sbatch job go to `logs/ob_gnn_<jobid>.out` and
`logs/ob_gnn_<jobid>.err`.

## Evaluation and prediction saving

Edit the command at the bottom of `eval_preds_obgnn.sbatch`. Also update
`/orcd/home/002/$USER/cfd_obgnn` to the path of your checkout if it differs.
Then submit:

```bash
sbatch eval_preds_obgnn.sbatch
```

Example:

```bash
srun python3 -u /orcd/home/002/$USER/cfd_obgnn/src/evaluate_predictions.py \
  --model_version A_ep0012 \
  --split test_bump \
  --checkpoints_dir model_checkpoints/checkpoints_A \
  --checkpoint_epoch 0012 \
  --mse_csv A_summary_mse.csv \
  --save_predictions True
```

Important evaluation arguments:

- `--model_version`: label written to the MSE CSV and used in the prediction
  output path.
- `--split`: split to evaluate. Choices are `test`, `test_ood`, `test_bump`,
  `test_all`, and `val`. `test_all` runs `test`, `test_ood`, and `test_bump`.
- `--checkpoints_dir`: directory containing `ckpt_bagXX_epYYYY.pt` files.
- `--checkpoint_epoch`: specific epoch to load, zero-padded to four digits
  such as `0012`. This loads `ckpt_bag00_ep0012.pt`.
- `--all_checkpoints True`: evaluate every checkpoint in `--checkpoints_dir`.
  Useful for validation sweeps; see `val_obgnn.sbatch`.
- `--mse_csv`: output CSV filename. Relative filenames are written under
  `evaluation_dir`; absolute paths are used as-is. If the CSV already exists,
  new rows are appended.
- `--save_predictions True`: save one prediction tensor per simulation.
- `--processed_dir`: compatibility alias for `--processed_data_dir`.
- Path overrides from `src/path_config.py` are also available, including
  `--processed_data_dir`, `--predictions_dir`, and `--evaluation_dir`.

Note: `--all_checkpoints` and `--save_predictions` are parsed as Python bools
in the current script. Use `--save_predictions True` or `--all_checkpoints True`
to enable them, and omit the flag when you want the default `False` behavior.

The evaluation script reports node-level metrics for:

- `x-velocity`
- `y-velocity`
- `pressure`
- `turbulent_viscosity`

There are two CSV formats used in the evaluation outputs:

1. Per-simulation error CSVs, named like `<model_version>_<split>.csv` or
   `<model_version>_<split>_mse.csv` depending on the run, contain one row per
   simulation. This is the format written by `src/mse_playground.ipynb`. Examples in this repo include `B_test_bump_mse.csv` and
   `O_test_bump_mse.csv`. Their columns are:

```text
index, sim_name,
x-vel_mse, x-vel_mse_norm,
y-vel_mse, y-vel_mse_norm,
pressure_mse, pressure_mse_norm,
turbulent_viscosity_mse, turbulent_viscosity_mse_norm
```

These files are useful for finding which individual bumped airfoils or
AirfRANS simulations have large errors. `sim_name` is the simulation identifier,
and each MSE value is computed over all nodes in that simulation.

2. Summary MSE CSVs, named with `--mse_csv` such as `A_summary_mse.csv`, contain
   one row per `(model_version, split)` evaluation run. This is the format
   currently written by `src/evaluate_predictions.py`. Its columns are:

```text
model_version, split,
x-vel_mse, y-vel_mse, pressure_mse, turbulent_viscosity_mse,
x-vel_mse_norm, y-vel_mse_norm, pressure_mse_norm, turbulent_viscosity_mse_norm
```

For the summary CSV, `evaluate_predictions.py` first computes the per-simulation
MSE for each field, then averages those per-simulation MSEs across the split.
This gives each simulation equal weight regardless of mesh size. The non-`_norm`
columns are raw physical-unit MSEs. The `_norm` columns divide each raw field
MSE by that field's dataset-wide variance for the evaluated split. If
`--split test_all` is used, the script appends three rows with the same
`model_version`: one each for `test`, `test_ood`, and `test_bump`. If
`--all_checkpoints True` is used, `evaluate_predictions.py` appends the
checkpoint epoch to the version label, for example `D_ep0016`.

When `--save_predictions True` is set, prediction tensors are written to:

```text
<predictions_dir>/<model_version>/<split>/pred_<simulation_name>.pt
```

Each tensor has shape `(num_nodes, 4)` in physical units, with columns ordered as:

```text
x-velocity, y-velocity, pressure, turbulent_viscosity
```

Evaluation logs go to `pred_logs/ob_gnn_<jobid>.out` and
`pred_logs/ob_gnn_<jobid>.err`.

## Other Notes

- Both `main.py` and `evaluate_predictions.py` require CUDA for their current
  sbatch workflows. Evaluation uses DGL UVA and explicitly errors if CUDA is not
  available.
- Use distinct `checkpoint_dir` values for distinct experiments. Otherwise
  training may resume from an older run in that directory.
- Use distinct `model_version` labels during evaluation so saved predictions do
  not overwrite previous runs.
- If preprocessing crashes, stale `*.tmp` cache files are removed by training on
  startup for known train/test caches.
