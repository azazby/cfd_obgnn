import configparser
import os
from dataclasses import dataclass


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@dataclass(frozen=True)
class PathConfig:
    base_dir: str
    dataset_dir: str
    processed_data_dir: str
    bumped_dir: str
    bumped_split_csv: str
    bench_config_path: str
    predictions_dir: str
    evaluation_dir: str


def resolve_path(path_value, base_dir):
    path_value = os.path.expandvars(os.path.expanduser(path_value))
    path_value = path_value.replace("{repo_root}", REPO_ROOT)
    path_value = path_value.replace("{base_dir}", base_dir)
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)
    return os.path.abspath(os.path.join(base_dir, path_value))


def add_path_config_args(parser):
    parser.add_argument("--path_config", type=str, default=os.path.join(REPO_ROOT, "config", "paths.ini"),
        help="INI file containing dataset/cache paths.",
    )
    parser.add_argument("--base_dir", type=str, default=None,
        help="Override paths.base_dir from the path config.",
    )
    parser.add_argument("--dataset_dir", type=str, default=None,
        help="Override paths.dataset_dir from the path config.",
    )
    parser.add_argument("--processed_data_dir", type=str, default=None,
        help="Override paths.processed_data_dir from the path config.",
    )
    parser.add_argument("--bumped_dir", type=str, default=None,
        help="Override paths.bumped_dir from the path config.",
    )
    parser.add_argument("--bumped_split_csv", type=str, default=None,
        help="Override paths.bumped_split_csv from the path config.",
    )
    parser.add_argument("--bench_config_path", type=str, default=None,
        help="Override paths.bench_config_path from the path config.",
    )
    parser.add_argument("--predictions_dir", type=str, default=None,
        help="Override paths.predictions_dir from the path config.",
    )
    parser.add_argument("--evaluation_dir", type=str, default=None,
        help="Override paths.evaluation_dir from the path config.",
    )


def load_path_config(args):
    parser = configparser.ConfigParser()
    parser["paths"] = {
        "base_dir": REPO_ROOT,
        "dataset_dir": "airfrans_data/Dataset",
        "processed_data_dir": "airfrans_data/processed",
        "bumped_dir": "airfrans_data/bumped_dataset",
        "bumped_split_csv": "airfrans_data/bumped_dataset_split.csv",
        "bench_config_path": "src/confAirfoil.ini",
        "predictions_dir": "predictions",
        "evaluation_dir": "evaluation",
    }
    parser.read(args.path_config)

    paths = parser["paths"]
    base_dir = os.path.abspath(
        os.path.expandvars(os.path.expanduser(args.base_dir or paths.get("base_dir", REPO_ROOT))).replace("{repo_root}", REPO_ROOT)
    )

    def get_path(cli_value, config_key):
        return resolve_path(cli_value or paths[config_key], base_dir)

    return PathConfig(
        base_dir=base_dir,
        dataset_dir=get_path(args.dataset_dir, "dataset_dir"),
        processed_data_dir=get_path(args.processed_data_dir, "processed_data_dir"),
        bumped_dir=get_path(args.bumped_dir, "bumped_dir"),
        bumped_split_csv=get_path(args.bumped_split_csv, "bumped_split_csv"),
        bench_config_path=get_path(args.bench_config_path, "bench_config_path"),
        predictions_dir=get_path(args.predictions_dir, "predictions_dir"),
        evaluation_dir=get_path(args.evaluation_dir, "evaluation_dir"),
    )
