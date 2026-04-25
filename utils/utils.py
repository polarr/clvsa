import json
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

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

def format_years(years: Sequence[int]) -> str:
    years = list(years)
    if not years:
        return "none"
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"