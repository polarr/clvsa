from pathlib import Path
from dotenv import load_dotenv
import os 

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler

load_dotenv()

# https://huggingface.co/datasets/mito0o852/OHLCV-1m
TICKER = "AAPL"
YEAR = 2024
DATASET_REPO = "mito0o852/OHLCV-1m"
HF_TOKEN = os.environ["HF_TOKEN"]

def download_year_of_parquets(year: int) -> str:
    """
    Download only parquet files for one year from the Hugging Face dataset.
    """
    local_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"data/ohlcv_{year}-*.parquet"],
        token=HF_TOKEN,
    )
    return local_dir


def load_one_ticker_from_year(local_dir: str, ticker: str, year: int) -> pd.DataFrame:
    """
    Load all monthly parquet files for a given year and keep only one ticker.
    """
    data_dir = Path(local_dir) / "data"
    files = sorted(data_dir.glob(f"ohlcv_{year}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for year {year}")

    dfs = []
    for f in files:
        df = pd.read_parquet(
            f,
            columns=["timestamp", "open", "high", "low", "close", "volume", "ticker"],
        )
        df = df[df["ticker"] == ticker]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError(f"No rows found for ticker={ticker} in year={year}")

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute OHLCV into 5-minute OHLCV.
    """
    df = df.set_index("timestamp")

    out = df.resample("5min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    out = out.dropna().reset_index()
    return out


def split_into_30min_blocks(df_5m: pd.DataFrame, block_size: int = 6) -> list[pd.DataFrame]:
    """
    Split 5-minute data into consecutive non-overlapping 30-minute blocks.
    Each block contains `block_size` rows of 5-minute OHLCV data.
    """
    df = df_5m.reset_index(drop=True)
    n_blocks = len(df) // block_size
    if n_blocks < 2:
        raise ValueError("Not enough 5-minute rows to build 30-minute blocks.")

    blocks = []
    for i in range(n_blocks):
        block = df.iloc[i * block_size : (i + 1) * block_size].copy().reset_index(drop=True)
        blocks.append(block)

    return blocks


def block_close_series(blocks: list[pd.DataFrame]) -> pd.Series:
    """
    Close price of each 30-minute block, taken from the last 5-minute row in the block.
    """
    closes = [float(block["close"].iloc[-1]) for block in blocks]
    return pd.Series(closes, dtype=np.float64)


def make_ternary_targets_from_blocks(blocks: list[pd.DataFrame], lam: float = 0.001) -> np.ndarray:
    """
    Paper-style 3-class labeling using log return between consecutive 30-minute blocks.

    Classes:
      0 = down
      1 = flat
      2 = up
    """
    closes = block_close_series(blocks)
    current_close = closes.iloc[:-1].to_numpy(dtype=np.float64)
    next_close = closes.iloc[1:].to_numpy(dtype=np.float64)

    log_ret = np.log(next_close / current_close)
    b_up = np.log((current_close + lam) / current_close)
    b_down = np.log((current_close - lam) / current_close)

    targets = np.ones(len(log_ret), dtype=np.int64)  # flat = 1
    targets[log_ret > b_up] = 2                      # up
    targets[log_ret < b_down] = 0                    # down

    return targets


def chronological_split_blocks(
    blocks: list[pd.DataFrame],
    targets: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Chronological split on block-level samples.
    Since target for block t depends on block t+1, targets has length len(blocks)-1.
    We therefore use blocks[:-1] as the labeled samples.
    """
    sample_blocks = blocks[:-1]
    n = len(sample_blocks)

    if len(targets) != n:
        raise ValueError(f"targets length {len(targets)} does not match block sample length {n}")

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_blocks = sample_blocks[:n_train]
    val_blocks = sample_blocks[n_train : n_train + n_val]
    test_blocks = sample_blocks[n_train + n_val :]

    y_train = targets[:n_train]
    y_val = targets[n_train : n_train + n_val]
    y_test = targets[n_train + n_val :]

    return train_blocks, val_blocks, test_blocks, y_train, y_val, y_test


def fit_block_scaler(train_blocks: list[pd.DataFrame], feature_cols: list[str]) -> StandardScaler:
    """
    Fit scaler on all 5-minute rows appearing inside train blocks only.
    Each attribute is normalized separately, using training data only.
    """
    train_rows = pd.concat([block[feature_cols] for block in train_blocks], axis=0, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(train_rows)
    return scaler


def transform_blocks(
    blocks: list[pd.DataFrame],
    scaler: StandardScaler,
    feature_cols: list[str],
) -> list[np.ndarray]:
    """
    Transform each block into a normalized (6, 5) numpy array.
    """
    transformed = []
    for block in blocks:
        values = scaler.transform(block[feature_cols]).astype(np.float32)  # (6, 5)
        transformed.append(values)
    return transformed


def flatten_blocks(blocks_2d: list[np.ndarray]) -> np.ndarray:
    """
    Flatten each (6, 5) block into a 30-dim vector.
    Output shape: (N_blocks, 30)
    """
    if not blocks_2d:
        raise ValueError("No blocks to flatten.")
    flat = np.stack([block.reshape(-1) for block in blocks_2d], axis=0).astype(np.float32)
    return flat


def make_sequences_from_block_features(
    block_features: np.ndarray,
    targets: np.ndarray,
    seq_len: int = 32,
):
    """
    Build sliding windows over flattened 30-minute block features.

    block_features: (N, 30)
    targets:        (N,)

    Output:
      X: (num_samples, seq_len, 30)
      y: (num_samples,)
    """
    if len(block_features) != len(targets):
        raise ValueError(
            f"block_features length {len(block_features)} != targets length {len(targets)}"
        )
    if len(block_features) < seq_len:
        raise ValueError(
            f"Need at least seq_len={seq_len} labeled blocks, got {len(block_features)}"
        )

    X, y = [], []
    for i in range(len(block_features) - seq_len + 1):
        X.append(block_features[i : i + seq_len])
        y.append(targets[i + seq_len - 1])

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


def build_datasets(
    ticker: str = TICKER,
    year: int = YEAR,
    seq_len: int = 32,
    lam: float = 0.001,
):
    feature_cols = ["open", "high", "low", "close", "volume"]

    local_dir = download_year_of_parquets(year)
    df_1m = load_one_ticker_from_year(local_dir, ticker, year)
    df_5m = resample_to_5min(df_1m)

    blocks = split_into_30min_blocks(df_5m, block_size=6)
    targets = make_ternary_targets_from_blocks(blocks, lam=lam)

    (
        train_blocks,
        val_blocks,
        test_blocks,
        y_train_blocks,
        y_val_blocks,
        y_test_blocks,
    ) = chronological_split_blocks(blocks, targets)

    scaler = fit_block_scaler(train_blocks, feature_cols)

    train_blocks_2d = transform_blocks(train_blocks, scaler, feature_cols)
    val_blocks_2d = transform_blocks(val_blocks, scaler, feature_cols)
    test_blocks_2d = transform_blocks(test_blocks, scaler, feature_cols)

    X_train_blocks = flatten_blocks(train_blocks_2d)
    X_val_blocks = flatten_blocks(val_blocks_2d)
    X_test_blocks = flatten_blocks(test_blocks_2d)

    X_train, y_train = make_sequences_from_block_features(
        X_train_blocks, y_train_blocks, seq_len=seq_len
    )
    X_val, y_val = make_sequences_from_block_features(
        X_val_blocks, y_val_blocks, seq_len=seq_len
    )
    X_test, y_test = make_sequences_from_block_features(
        X_test_blocks, y_test_blocks, seq_len=seq_len
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "input_dim": X_train.shape[-1],
        "feature_cols": feature_cols,
    }


def main():
    seq_len = 32
    lam = 0.001

    data = build_datasets(
        ticker=TICKER,
        year=YEAR,
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


if __name__ == "__main__":
    main()