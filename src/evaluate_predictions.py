import os
import numpy as np
import torch
import dgl
import pandas as pd

from main import CoordGNN, EnsembleModel
from path_config import add_path_config_args, load_path_config, resolve_path

from lips import get_root_path
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from bump_airfoil_dataset import BumpComboAirfoilDataset


PREDICT_NORMALIZE = {
    "means": [62, 4.5, 12.5, 62, 5, 42, 10, -475, 0.0008],
    "stds":  [20, 5.5, 4.2, 20, 6.5, 30, 31, 2800, 0.003],
    "maxs":  [0.025, 0.065],
}


def load_model(checkpoint_dir, device, ckpt_file=None, hidden1=256, hidden2=256, hidden3=128, k=1):
    # if no specified checkpoint file, get latest checkpoint
    if ckpt_file is None:
        ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

        latest = os.path.join(checkpoint_dir, ckpt_files[-1])
    else:
        latest = os.path.join(checkpoint_dir, ckpt_file)
        
    print(f"Loading checkpoint: {latest}")

    state = torch.load(latest, map_location='cpu')
    models = []
    for sd in state['model_state_dicts']:
        m = CoordGNN(66, hidden1, hidden2, hidden3, 4, k)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    return EnsembleModel(models).to(device).eval()


def load_processed_split(processed_dir, split_name):
    """
    Loads processed dataset split.
    """
    path_map = {
        'train': 'processed_train.pt',
        'test': 'processed_test.pt',
        'test_ood': 'processed_test_ood.pt',
        'test_bump': 'processed_test_bump.pt',
        'val' : 'processed_val_bump.pt',
        'test_bump_cat': 'processed_test_bump_cat.pt'
    }
    path = os.path.join(processed_dir, path_map[split_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed split file not found: {path}")

    print(f"Loading processed split: {path}")
    graphs, shape_features, means, stds = torch.load(path)
    return graphs, shape_features, means, stds


@torch.no_grad()
def run_inference(model, graphs, shape_features, device):
    """
    Runs inference on processed merged graph.

    Returns:
        pred_phys: torch.Tensor of shape (total_nodes, 4), in physical units
    """
    sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 15])
    dataloader = dgl.dataloading.DataLoader(
        graphs,
        torch.arange(graphs.num_nodes()),
        sampler,
        batch_size=8192,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
    )

    model.eval()
    model = model.to(device)
    shape_features = shape_features.to(device)

    outputs = []
    for it, (_, _, blocks) in enumerate(dataloader):
        blocks = [block.to(device) for block in blocks]

        x = blocks[0].srcdata["features"].to(device).to(torch.float32)
        x_shape = shape_features[blocks[0].srcdata["sim_ids"].to(device)]
        x = torch.cat([x, x_shape], dim=1)

        y_hat = model(blocks, x)
        outputs.append(y_hat.cpu())

        if it % 1000 == 0:
            print(f"Iteration {it:05d}/{len(dataloader):05d} | Inference")

    pred_norm = torch.cat(outputs, dim=0)  # (total_nodes, 4)

    pred_phys = pred_norm.clone()
    for i in range(4):
        std = float(PREDICT_NORMALIZE["stds"][i + 5])
        mean = float(PREDICT_NORMALIZE["means"][i + 5])
        pred_phys[:, i] = pred_phys[:, i] * std + mean

    return pred_phys



def compute_global_stats(dataset, target_fields):
    """
    Helper function that computes mean and std of each target field in a given dataset.
    """
    stats = {}
    for field in target_fields:
        data = dataset.data[field]
        stats[field] = {
            'mean': np.mean(data),
            'std': np.std(data)
        }
    return stats


def evaluate(model_version, pred_all, dataset, target_fields, save_predictions=False, split_name='test', pred_root_dir=None):
    """
    Compute overall raw MSE and normalized MSE for each target field over a dataset.

    Predictions are split per simulation using dataset metadata, MSE is computed
    per simulation, and then averaged so each simulation is weighted equally
    (independent of mesh size).

    Optionally saves per-simulation predictions to disk.

    Parameters
    ----------
    pred_all : torch.Tensor (total_nodes, num_fields)
        Concatenated predictions aligned with dataset ordering.
    dataset : object
        Contains ground truth in dataset.data[field] and simulation metadata.
    target_fields : list of str
        Fields corresponding to prediction columns.
    save_predictions : bool, optional
        If True, saves per-simulation predictions.
    split_name : str, optional
        Name of dataset split (used for saving).
    pred_root_dir : str, optional
        Directory for saved predictions.

    Returns
    -------
    mean_mse : np.ndarray
        Mean raw MSE per field (equal weighting over simulations).
    mean_mse_norm : np.ndarray
        Mean normalized MSE per field.
    """
    stats = compute_global_stats(dataset, target_fields)

    if save_predictions:
        if pred_root_dir is None:
            raise ValueError("pred_root_dir is required when save_predictions=True")
        out_dir = os.path.join(pred_root_dir, model_version, split_name)
        os.makedirs(out_dir, exist_ok=True)

    sim_entries = dataset.extra_data['simulation_names']

    # node sanity check
    expected_total = sum(int(size) for _, size in sim_entries)
    if pred_all.shape[0] != expected_total:
        raise ValueError(
            f"Total node count mismatch: predictions have {pred_all.shape[0]} nodes, "
            f"but split metadata sums to {expected_total}"
        )

    p = 0 # simulation node pointer
    per_sim_mse_rows = []  # per-simulation mse
    per_sim_mse_norm_rows = [] # per-simulation mse normalized
 
    for idx, (sim_name, size) in enumerate(sim_entries):
        size = int(size)
        pred_sim = pred_all[p:p+size]  # (N, 4)
        # prediction sanity check
        if pred_sim.shape != (size, 4):
            raise ValueError(
                f"Prediction chunk shape mismatch for {sim_name}: "
                f"expected {(size, 4)}, got {tuple(pred_sim.shape)}"
            )
        # optional save entire prediction
        if save_predictions:
            pred_sim = pred_sim.clone()
            out_path = os.path.join(out_dir, f"pred_{sim_name}.pt")
            # atomic write
            tmp_path = out_path + ".tmp"
            torch.save(pred_sim, tmp_path)
            os.replace(tmp_path, out_path)
            print(f"[{idx}/{len(sim_entries)}] Saved -> {out_path}")

        # compute per-simulation mse for each target field
        pred_sim = pred_sim.numpy()
        mse_row = []
        mse_normalized_row = []
        for i, field in enumerate(target_fields):
            gt = dataset.data[field][p:p+size]

            field_mse = np.mean((gt - pred_sim[:, i])**2)
            field_mse_norm = field_mse / (stats[field]['std']**2)
            mse_row.append(field_mse)
            mse_normalized_row.append(field_mse_norm)

        per_sim_mse_rows.append(mse_row)
        per_sim_mse_norm_rows.append(mse_normalized_row)
        p += size

    per_sim_mse_rows = np.array(per_sim_mse_rows)
    per_sim_mse_norm_rows = np.array(per_sim_mse_norm_rows)

    # average over all simulations
    mean_mse = per_sim_mse_rows.mean(axis=0)
    mean_mse_norm = per_sim_mse_norm_rows.mean(axis=0)

    return mean_mse, mean_mse_norm


# ---- Main ----

def run_eval(benchmark, model_version, model, device, split, output_csv, path_config, processed_dir, save_predictions=False):
    target_fields =  ['x-velocity', 'y-velocity', 'pressure', 'turbulent_viscosity']

    def make_summary_mse_table(split):
        """
        Returns dataframe of one row containing the overall MSE of a given test split dataset. 
        """
        if split == 'test':
            dataset = benchmark._test_dataset
        elif split == 'test_ood':
            dataset = benchmark._test_ood_dataset
        elif split == 'test_bump':
            dataset = BumpComboAirfoilDataset('test',dict(),dict(),
                            benchmark.train_dataset._attr_names,
                            bumped_dir=path_config.bumped_dir,
                            split='test',
                            split_csv=path_config.bumped_split_csv
                        )
        elif split == 'val':
            dataset = BumpComboAirfoilDataset('val',dict(),dict(),
                            benchmark.train_dataset._attr_names,
                            bumped_dir=path_config.bumped_dir,
                            split='val',
                            split_csv=path_config.bumped_split_csv
            )

        print(f"Loading {split} graph...")
        graphs, shape_features, means, stds = load_processed_split(processed_dir, split)
        print("Running inference...")
        pred_all = run_inference(model, graphs, shape_features, device)
        print("Evaluating...")
        mean_mse, mean_mse_norm = evaluate(
            model_version,
            pred_all,
            dataset,
            target_fields,
            split_name=split,
            save_predictions=save_predictions,
            pred_root_dir=path_config.predictions_dir,
        )

        return pd.DataFrame([{
            'model_version':model_version,
            'split': split,
            'x-vel_mse': mean_mse[0],
            'y-vel_mse': mean_mse[1],
            'pressure_mse': mean_mse[2],
            'turbulent_viscosity_mse': mean_mse[3],
            'x-vel_mse_norm': mean_mse_norm[0],
            'y-vel_mse_norm': mean_mse_norm[1],
            'pressure_mse_norm': mean_mse_norm[2],
            'turbulent_viscosity_mse_norm': mean_mse_norm[3]
        }])

    # Save MSE CSV 
    output_df = pd.read_csv(output_csv) if os.path.isfile(output_csv) else None
    splits_to_run = ['test','test_ood','test_bump'] if split == 'test_all' else [split]

    for s in splits_to_run:
        s_df = make_summary_mse_table(s)
        output_df = pd.concat((output_df, s_df)) if output_df is not None else s_df

    output_df.to_csv(output_csv, index=False)
    print("Saved to:", output_csv)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_version',
        type=str,
        required=True,
        help='Model version name'
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'test', 'test_ood', 'test_bump','test_all', 'val'],
        help='Which split to run inference on.'
    )
    parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default='checkpoints',
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--all_checkpoints',
        type=bool,
        default=False,
        help='Run eval for all checkpoints'
    )
    parser.add_argument(
        '--checkpoint_epoch',
        type=str,
        default=None,
        help='Load specific checkpoint epoch. Must be left padded with zeros to 4 chars (e.g. 0012 for epoch 12).'
    )
    parser.add_argument(
        '--mse_csv',
        type=str,
        default='summary_mse.csv',
        help='Summary MSE CSV filename'
    )
    parser.add_argument(
        '--save_predictions',
        type=bool,
        default=False,
        help='Optional save all simulation predictions.'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default=None,
        help='Compatibility alias for --processed_data_dir.'
    )
    add_path_config_args(parser)
    args, _ = parser.parse_known_args()
    paths = load_path_config(args)
    processed_dir = resolve_path(args.processed_dir, paths.base_dir) if args.processed_dir else paths.processed_data_dir
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script as written (use_uva=True).")
    
     # Load benchmark
    benchmark = AirfRANSBenchmark(
        benchmark_path=str(paths.dataset_dir),
        config_path=str(paths.bench_config_path),
        benchmark_name='DEFAULT',
        log_path=get_root_path() + 'lips_logs.log'
    )
    benchmark.load(path=str(paths.dataset_dir))

    device = torch.device('cuda:0')
    checkpoint_dir = resolve_path(args.checkpoints_dir, paths.base_dir)
    output_csv = args.mse_csv if os.path.isabs(args.mse_csv) else os.path.join(paths.evaluation_dir, args.mse_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    if args.all_checkpoints:
        for ckpt_file in sorted(os.listdir(checkpoint_dir)):
            # load model
            model = load_model(checkpoint_dir, device, ckpt_file=ckpt_file)
            # make model version label
            epoch_label = ckpt_file.split('.')[0].split('_')[-1]
            full_model_version = f'{args.model_version}_{epoch_label}'
            run_eval(
                benchmark=benchmark,
                model_version=full_model_version,
                model=model,
                device=device,
                split=args.split,
                output_csv=output_csv,
                path_config=paths,
                processed_dir=processed_dir,
                save_predictions=args.save_predictions,
            )
    else:
        # load model
        ckpt_file = f'ckpt_bag00_ep{args.checkpoint_epoch}.pt' if args.checkpoint_epoch is not None else None
        model = load_model(checkpoint_dir, device, ckpt_file=ckpt_file)
        run_eval(
            benchmark=benchmark,
            model_version=args.model_version,
            model=model,
            device=device,
            split=args.split,
            output_csv=output_csv,
            path_config=paths,
            processed_dir=processed_dir,
            save_predictions=args.save_predictions,
        )
 
