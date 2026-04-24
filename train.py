import argparse
import json
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from model.model import ModelConfig, build_model
from dataset.load_dataset import TICKER, YEARS, build_datasets
from utils.parser import parse_args
import model.config


class Seq2SeqDataset(Dataset):
    def __init__(self, X_enc: np.ndarray, X_dec: np.ndarray, y_dec: np.ndarray):
        self.X_enc = torch.tensor(X_enc, dtype=torch.float32)
        self.X_dec = torch.tensor(X_dec, dtype=torch.float32)
        self.y_dec = torch.tensor(y_dec, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X_enc)

    def __getitem__(self, idx: int):
        return self.X_enc[idx], self.X_dec[idx], self.y_dec[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten_logits_targets(
    logits_seq: torch.Tensor,
    targets_seq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits_flat = logits_seq.reshape(-1, logits_seq.size(-1))
    targets_flat = targets_seq.reshape(-1)
    return logits_flat, targets_flat


@torch.no_grad()
def compute_multiclass_map(logits_flat: torch.Tensor, targets_flat: torch.Tensor) -> Dict[str, float]:
    probs = torch.softmax(logits_flat, dim=1).cpu().numpy()
    targets_np = targets_flat.cpu().numpy()

    metrics = {}
    ap_values = []

    for cls in range(probs.shape[1]):
        binary_targets = (targets_np == cls).astype(np.int32)

        if binary_targets.sum() == 0:
            ap = float("nan")
        else:
            ap = float(average_precision_score(binary_targets, probs[:, cls]))

        metrics[f"class_{cls}_ap"] = ap
        if not np.isnan(ap):
            ap_values.append(ap)

    metrics["map"] = float(np.mean(ap_values)) if ap_values else float("nan")
    return metrics


@torch.no_grad()
def compute_metrics(logits_flat: torch.Tensor, targets_flat: torch.Tensor) -> Dict[str, float]:
    preds = torch.argmax(logits_flat, dim=1)

    total = targets_flat.numel()
    accuracy = (preds == targets_flat).sum().item() / total if total > 0 else 0.0

    per_class_recall = []
    metrics = {"accuracy": accuracy}

    for cls in [0, 1, 2]:
        cls_mask = targets_flat == cls
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
    metrics.update(compute_multiclass_map(logits_flat, targets_flat))
    return metrics


class EarlyStopping:
    def __init__(
        self,
        patience: int = 6,
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


def format_years(years: Sequence[int]) -> str:
    years = list(years)
    if not years:
        return "none"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def build_dataloaders(
    batch_size: int = 16,
    ticker: str = TICKER,
    years: Sequence[int] = YEARS,
    num_workers: int = 0,
    lam: float = 0.2,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    data = build_datasets(
        ticker=ticker,
        years=years,
        lam=lam,
    )

    X_enc_train = data["X_enc_train"]
    X_dec_train = data["X_dec_train"]
    y_dec_train = data["y_dec_train"]

    X_enc_val = data["X_enc_val"]
    X_dec_val = data["X_dec_val"]
    y_dec_val = data["y_dec_val"]

    X_enc_test = data["X_enc_test"]
    X_dec_test = data["X_dec_test"]
    y_dec_test = data["y_dec_test"]

    print("Years:", data["years"])
    print("Input dim:", data["input_dim"])
    print("Blocks per day:", data["blocks_per_day"])

    print(
        "X_enc_train:", X_enc_train.shape,
        "X_dec_train:", X_dec_train.shape,
        "y_dec_train:", y_dec_train.shape,
    )
    print(
        "X_enc_val:  ", X_enc_val.shape,
        "X_dec_val:  ", X_dec_val.shape,
        "y_dec_val:  ", y_dec_val.shape,
    )
    print(
        "X_enc_test: ", X_enc_test.shape,
        "X_dec_test: ", X_dec_test.shape,
        "y_dec_test: ", y_dec_test.shape,
    )

    def print_stats(name, x):
        print(f"{name} global mean/std:", float(x.mean()), float(x.std()))

    print("\nNormalized input stats:")
    print_stats("X_enc_train", X_enc_train)
    print_stats("X_dec_train", X_dec_train)
    print_stats("X_enc_val", X_enc_val)
    print_stats("X_dec_val", X_dec_val)
    print_stats("X_enc_test", X_enc_test)
    print_stats("X_dec_test", X_dec_test)

    print("\nDecoder-sequence class balance:")
    for name, y in [("Train", y_dec_train), ("Val", y_dec_val), ("Test", y_dec_test)]:
        flat = y.reshape(-1)
        vals, counts = np.unique(flat, return_counts=True)
        ratios = {int(v): float(c / len(flat)) for v, c in zip(vals, counts)}
        print(name, ratios)

    train_dataset = Seq2SeqDataset(X_enc_train, X_dec_train, y_dec_train)
    val_dataset = Seq2SeqDataset(X_enc_val, X_dec_val, y_dec_val)
    test_dataset = Seq2SeqDataset(X_enc_test, X_dec_test, y_dec_test)

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
    total_positions = 0
    all_logits = []
    all_targets = []

    for X_enc, X_dec, y_dec in loader:
        X_enc = X_enc.to(device)
        X_dec = X_dec.to(device)
        y_dec = y_dec.to(device)

        with torch.set_grad_enabled(is_train):
            logits_seq = model(X_enc, X_dec)                    # (B, 13, C)
            logits_flat, targets_flat = flatten_logits_targets(logits_seq, y_dec)

            loss = criterion(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        num_positions = targets_flat.size(0)
        total_loss += loss.item() * num_positions
        total_positions += num_positions

        all_logits.append(logits_flat.detach())
        all_targets.append(targets_flat.detach())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # if not is_train:
    #     probs = torch.softmax(all_logits, dim=1)
    #     print("val mean probs:", probs.mean(dim=0).cpu().numpy())
    #     print("val std probs: ", probs.std(dim=0).cpu().numpy())
    #     print("val mean logits:", all_logits.mean(dim=0).cpu().numpy())
    #     print("val std logits: ", all_logits.std(dim=0).cpu().numpy())

    metrics = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / total_positions
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    return run_epoch(model, loader, criterion, device, optimizer=None)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, input_dim = build_dataloaders(
        batch_size=args.batch_size,
        ticker=args.ticker,
        years=args.years,
        num_workers=args.num_workers,
        lam=args.lam,
    )

    config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=False,
        output_dim=3,
        block_rows=5,
        block_cols=6,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        use_row_specific_conv=args.use_row_specific_conv
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

    years_tag = format_years(args.years)
    run_name = (
        f"{args.model_name}_{args.ticker}_{years_tag}"
        f"_lam{args.lam}"
        f"_h{args.hidden_dim}"
        f"_nl{args.num_layers}"
        f"_bs{args.batch_size}"
        f"_cc{args.conv_channels}"
        f"_ck{args.conv_kernel_size}"
    )

    save_dir = Path(args.save_dir)
    best_path = save_dir / f"{run_name}_best.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    history_path = save_dir / f"{run_name}_history.json"

    best_metric = float("inf") if args.monitor_metric == "loss" else float("-inf")
    history = []

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Ticker: {args.ticker} | Years: {args.years}")
    print(f"lam={args.lam}")
    print(
        f"hidden_dim={args.hidden_dim} | num_layers={args.num_layers} "
        f"| dropout={args.dropout} | conv_channels={args.conv_channels} "
        f"| conv_kernel_size={args.conv_kernel_size}"
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

    # right before test evaluation
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    print(f"\nLoaded best checkpoint from epoch {best_ckpt['epoch']} "
        f"with {best_ckpt['monitor_metric']}={best_ckpt['best_metric']:.4f}")

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