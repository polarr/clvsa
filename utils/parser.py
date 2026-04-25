import argparse
import torch

DEFAULTS = {
    "equity": {
        "ticker": "AAPL",
        "years": [2022, 2023, 2024],
        "lam": 0.2,
    },
    "futures": {
        "ticker": "BTCUSDT",
        "years": [2022, 2023, 2024],
        "lam": 0.001,
    },
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_only",
        action="store_true",
        help="Only build dataloaders and print dataset summary, don't train model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="clvsa",
        choices=["lstm", "cls", "clsa", "clvsa", "clsa_inter", "clsa_self", "ls", "lsa", "lsa_inter", "lsa_self"],
    )

    parser.add_argument(
        "--data_source",
        type=str,
        default="futures",
        choices=["equity", "stock", "futures"],
    )
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--years", type=int, nargs="+", default=None)
    parser.add_argument("--lam", type=float, default=None)

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

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--prior_fc_dim", type=int, default=512)
    parser.add_argument("--posterior_fc_dim", type=int, default=256)

    parser.add_argument("--alpha", type=float, default=2.5e-4)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--beta_anneal_epochs", type=int, default=10)
    
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

    args = parser.parse_args()

    defaults = DEFAULTS[args.data_source]

    if args.ticker is None:
        args.ticker = defaults["ticker"]

    if args.years is None:
        args.years = defaults["years"]

    if args.lam is None:
        args.lam = defaults["lam"]
    
    return args