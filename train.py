import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from model import ModelConfig, build_model
from dataset.load_dataset import TICKER, YEAR, build_datasets


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_multiclass_map(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()

    metrics = {}
    ap_values = []

    for cls in range(probs.shape[1]):
        binary_targets = (targets_np == cls).astype(np.int32)
        ap = average_precision_score(binary_targets, probs[:, cls])
        metrics[f"class_{cls}_ap"] = float(ap)
        ap_values.append(float(ap))

    metrics["map"] = float(np.mean(ap_values))
    return metrics


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits, dim=1)

    total = targets.numel()
    accuracy = (preds == targets).sum().item() / total if total > 0 else 0.0

    per_class_recall = []
    metrics = {"accuracy": accuracy}

    for cls in [0, 1, 2]:
        cls_mask = targets == cls
        pred_mask = preds == cls

        n_cls = cls_mask.sum().item()
        n_pred = pred_mask.sum().item()
        n_correct = (cls_mask & pred_mask).sum().item()

        metrics[f"class_{cls}_target_rate"] = n_cls / total if total > 0 else 0.0
        metrics[f"class_{cls}_prediction_rate"] = n_pred / total if total > 0 else 0.0
        metrics[f"class_{cls}_recall"] = n_correct / n_cls if n_cls > 0 else 0.0
        metrics[f"class_{cls}_precision"] = n_correct / n_pred if n_pred > 0 else 0.0

        per_class_recall.append(metrics[f"class_{cls}_recall"])

    metrics["balanced_accuracy"] = sum(per_class_recall) / 3.0
    metrics.update(compute_multiclass_map(logits, targets))
    return metrics


class EarlyStopping:
    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("inf") if mode == "min" else float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience


def build_dataloaders(
    seq_len: int = 32,
    batch_size: int = 16,
    ticker: str = TICKER,
    year: int = YEAR,
    num_workers: int = 0,
    lam: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    data = build_datasets(
        ticker=ticker,
        year=year,
        seq_len=seq_len,
        lam=lam,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print("Input dim:", data["input_dim"])
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    print("\nSequence class balance:")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        vals, counts = np.unique(y, return_counts=True)
        ratios = {int(v): float(c / len(y)) for v, c in zip(vals, counts)}
        print(name, ratios)

    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, int(data["input_dim"])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_targets = []

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(X)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach())
        all_targets.append(y.detach())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / total_samples
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    return run_epoch(model, loader, criterion, device, optimizer=None)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="seq2seq_attn",
        choices=["lstm", "seq2seq_attn", "s2s_attn", "seq2seq_attention"],
    )
    parser.add_argument("--ticker", type=str, default=TICKER)
    parser.add_argument("--year", type=int, default=YEAR)

    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--lam", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--use_block_conv", action="store_true")
    parser.add_argument("--conv_channels", type=int, default=32)
    parser.add_argument("--conv_kernel_size", type=int, default=3)
    parser.add_argument("--conv_proj_dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument(
        "--monitor_metric",
        type=str,
        default="map",
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, input_dim = build_dataloaders(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        ticker=args.ticker,
        year=args.year,
        num_workers=args.num_workers,
        lam=args.lam,
    )

    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=False,  # keep unidirectional for seq2seq
        output_dim=3,
        block_rows=5,
        block_cols=6,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        conv_proj_dim=args.conv_proj_dim,
        use_block_conv=args.use_block_conv,
    )
    model = build_model(args.model_name, config).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    monitor_mode = "min" if args.monitor_metric == "loss" else "max"
    early_stopper = EarlyStopping(
        patience=args.patience,
        mode=monitor_mode,
        min_delta=args.min_delta,
    )

    run_name = (
        f"{args.model_name}_{args.ticker}_{args.year}"
        f"_seq{args.seq_len}"
        f"_lam{args.lam}"
        f"_h{args.hidden_dim}"
        f"_nl{args.num_layers}"
        f"_bs{args.batch_size}"
    )
    if args.use_block_conv:
        run_name += (
            f"_conv"
            f"_cc{args.conv_channels}"
            f"_ck{args.conv_kernel_size}"
            f"_cp{args.conv_proj_dim}"
        )

    save_dir = Path(args.save_dir)
    best_path = save_dir / f"{run_name}_best.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    history_path = save_dir / f"{run_name}_history.json"

    best_metric = float("inf") if args.monitor_metric == "loss" else float("-inf")
    history = []

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Ticker: {args.ticker} | Year: {args.year}")
    print(f"seq_len={args.seq_len} | lam={args.lam}")
    print(
        f"hidden_dim={args.hidden_dim} | num_layers={args.num_layers} "
        f"| dropout={args.dropout} | use_block_conv={args.use_block_conv}"
    )
    if args.use_block_conv:
        print(
            f"conv_channels={args.conv_channels} | "
            f"conv_kernel_size={args.conv_kernel_size} | "
            f"conv_proj_dim={args.conv_proj_dim}"
        )
    print(
        f"batch_size={args.batch_size} | lr={args.lr} | "
        f"monitor_metric={args.monitor_metric}"
    )
    print(
        f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} "
        f"| Test batches: {len(test_loader)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        current_metric = val_metrics[args.monitor_metric]
        is_better = (
            current_metric < best_metric
            if args.monitor_metric == "loss"
            else current_metric > best_metric
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_bacc={train_metrics['balanced_accuracy']:.4f} "
            f"train_map={train_metrics['map']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_map={val_metrics['map']:.4f} "
            f"val_pred_rates=("
            f"{val_metrics['class_0_prediction_rate']:.3f}, "
            f"{val_metrics['class_1_prediction_rate']:.3f}, "
            f"{val_metrics['class_2_prediction_rate']:.3f})"
        )

        if is_better:
            best_metric = current_metric
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "monitor_metric": args.monitor_metric,
                    "args": vars(args),
                },
                best_path,
            )

        if epoch >= args.min_epochs and early_stopper.step(current_metric):
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save(
        {
            "epoch": history[-1]["epoch"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "monitor_metric": args.monitor_metric,
            "args": vars(args),
        },
        last_path,
    )

    save_json(history_path, history)

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    print(f"\nBest checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()