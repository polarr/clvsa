import argparse
from dataset.load_dataset import TICKER, YEARS
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="clsa",
        choices=["lstm", "cls", "clsa", "clsa_inter", "clsa_self", "ls", "lsa", "lsa_inter", "lsa_self"],
    )
    parser.add_argument("--ticker", type=str, default=TICKER)
    parser.add_argument("--years", type=int, nargs="+", default=YEARS)
    parser.add_argument("--lam", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument(
        "--use_row_specific_conv",
        action="store_true",
        help="Use separate Conv1d kernels per OHLCV row instead of shared row kernels.",
    )

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min_epochs", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=0.0)
    

    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="balanced_accuracy",
        choices=["loss", "accuracy", "balanced_accuracy", "map"],
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    return parser.parse_args()