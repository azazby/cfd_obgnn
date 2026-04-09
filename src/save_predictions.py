import os
import sys
import numpy as np
import torch
import dgl

sys.path.insert(0, '/orcd/home/002/jlu25/ML4CFD-Offset-based-Graph-Convolution/src')
from main import CoordGNN, EnsembleModel

from lips import get_root_path
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark


BASE_DIR = '/orcd/home/002/jlu25/ML4CFD-Offset-based-Graph-Convolution'
DATASET_DIR = os.path.join(BASE_DIR, 'airfrans_data/Dataset')
BENCH_CONFIG = os.path.join(BASE_DIR, 'src/confAirfoil.ini')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
PRED_ROOT_DIR = os.path.join(BASE_DIR, 'predictions')

# Point this to wherever your cached processed split files actually live.
PROCESSED_DIR = '/home/jlu25/orcd/scratch/airfrans_data/processed'
# If needed instead:
# PROCESSED_DIR = '/orcd/home/002/jlu25/orcd/scratch/airfrans_data/processed'


# This matches the original predict() function exactly.
PREDICT_NORMALIZE = {
    "means": [62, 4.5, 12.5, 62, 5, 42, 10, -475, 0.0008],
    "stds":  [20, 5.5, 4.2, 20, 6.5, 30, 31, 2800, 0.003],
    "maxs":  [0.025, 0.065],
}


# ── Load model ────────────────────────────────────────────────────────────────
def load_model(checkpoint_dir, device, hidden1=256, hidden2=256, hidden3=128, k=1):
    ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest = os.path.join(checkpoint_dir, ckpt_files[-1])
    print(f"Loading checkpoint: {latest}")

    state = torch.load(latest, map_location='cpu')
    models = []
    for sd in state['model_state_dicts']:
        m = CoordGNN(66, hidden1, hidden2, hidden3, 4, k)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)

    return EnsembleModel(models).to(device).eval()


# ── Load processed split ──────────────────────────────────────────────────────
def load_processed_split(processed_dir, split_name):
    path_map = {
        'train': 'processed_train.pt',
        'test': 'processed_test.pt',
        'test_ood': 'processed_test_ood.pt',
        'deviated': 'processed_deviated.pt'
    }
    path = os.path.join(processed_dir, path_map[split_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed split file not found: {path}")

    print(f"Loading processed split: {path}")
    graphs, shape_features, means, stds = torch.load(path)
    return graphs, shape_features, means, stds


# ── Run inference on processed merged graph ───────────────────────────────────
@torch.no_grad()
def run_inference_processed(model, graphs, shape_features, device):
    """
    Runs inference in the same style as the original predict() path.

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


# ── Save one tensor per simulation ────────────────────────────────────────────
def save_predictions_by_sim(pred_all, dataset, split_name, pred_root_dir):
    """
    pred_all: torch.Tensor of shape (total_nodes, 4)
    dataset.extra_data['simulation_names']: list of (sim_name, num_nodes)
    """
    out_dir = os.path.join(pred_root_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    sim_entries = dataset.extra_data['simulation_names']
    expected_total = sum(int(size) for _, size in sim_entries)

    if pred_all.shape[0] != expected_total:
        raise ValueError(
            f"Total node count mismatch: predictions have {pred_all.shape[0]} nodes, "
            f"but split metadata sums to {expected_total}"
        )

    p = 0
    for idx, (sim_name, size) in enumerate(sim_entries, start=1):
        size = int(size)
        pred_sim = pred_all[p:p+size].clone()  # (N, 4)

        if pred_sim.shape != (size, 4):
            raise ValueError(
                f"Prediction chunk shape mismatch for {sim_name}: "
                f"expected {(size, 4)}, got {tuple(pred_sim.shape)}"
            )

        out_path = os.path.join(out_dir, f"pred_{sim_name}.pt")
        # atomic write
        tmp_path = out_path + ".tmp"
        torch.save(pred_sim, tmp_path)
        os.replace(tmp_path, out_path)
        print(f"[{idx}/{len(sim_entries)}] Saved -> {out_path}")

        p += size


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'test', 'test_ood'],
        help='Which split to run inference on.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='CUDA device index.'
    )
    parser.add_argument(
        '--processed_dir',
        type=str,
        default=PROCESSED_DIR,
        help='Directory containing processed_train.pt / processed_test.pt / processed_test_ood.pt'
    )
    args, _ = parser.parse_known_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script as written (use_uva=True).")

    device = torch.device(f'cuda:{args.gpu}')

    # Load raw benchmark only to get dataset metadata / simulation_names
    LOG_PATH = get_root_path() + 'lips_logs.log'
    benchmark = AirfRANSBenchmark(
        benchmark_path=DATASET_DIR,
        config_path=BENCH_CONFIG,
        benchmark_name='DEFAULT',
        log_path=LOG_PATH,
    )
    benchmark.load(path=DATASET_DIR)

    split_map = {
        'train': benchmark.train_dataset,
        'test': benchmark._test_dataset,
        'test_ood': benchmark._test_ood_dataset,
    }
    dataset = split_map[args.split]

    print("Loading processed graph split...")
    graphs, shape_features, means, stds = load_processed_split(args.processed_dir, args.split)

    print("Loading model...")
    model = load_model(CHECKPOINT_DIR, device)

    print(f"Running inference for split '{args.split}'...")
    pred_all = run_inference_processed(model, graphs, shape_features, device)

    print(f"Saving per-simulation prediction tensors for split '{args.split}'...")
    save_predictions_by_sim(pred_all, dataset, args.split, PRED_ROOT_DIR)

    print("Done.")