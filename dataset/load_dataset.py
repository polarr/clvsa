from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler

# https://huggingface.co/datasets/mito0o852/OHLCV-1m
TICKER = "AAPL"
YEAR = 2024
DATASET_REPO = "mito0o852/OHLCV-1m"


def download_year_of_parquets(year: int) -> str:
    """
    Download only parquet files for one year from the Hugging Face dataset.
    """
    local_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"data/ohlcv_{year}-*.parquet"],
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

    out = df.resample("5min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })

    out = out.dropna().reset_index()
    return out


def add_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label:
      1 if next close >= current close
      0 otherwise
    """
    out = df.copy()
    out["target"] = (out["close"].shift(-1) >= out["close"]).astype(int)
    out = out.iloc[:-1].reset_index(drop=True)  # last row has no next interval
    return out


def chronological_split(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split to avoid leakage.
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    return train_df, val_df, test_df


def normalize_splits(train_df, val_df, test_df, feature_cols):
    """
    Fit scaler on train only, transform all splits.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler


def make_sequences(df: pd.DataFrame, feature_cols, seq_len=32):
    """
    Build sliding windows:
      X[i] = feature rows [i, i+seq_len)
      y[i] = target at last row of that window
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df["target"].to_numpy(dtype=np.int64)

    X, y = [], []

    for i in range(len(df) - seq_len + 1):
        X.append(values[i:i + seq_len])
        y.append(targets[i + seq_len - 1])

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    return X, y


def main():
    feature_cols = ["open", "high", "low", "close", "volume"]

    local_dir = download_year_of_parquets(YEAR)

    df_1m = load_one_ticker_from_year(local_dir, TICKER, YEAR)
    print("Raw 1-minute shape:", df_1m.shape)
    print(df_1m.head())

    # Resample to 5 minute intervals
    df_5m = resample_to_5min(df_1m)
    print("\n5-minute shape:", df_5m.shape)
    print(df_5m.head())

    # Add labels
    df_labeled = add_binary_label(df_5m)
    print("\nLabeled shape:", df_labeled.shape)
    print(df_labeled.head())

    train_df, val_df, test_df = chronological_split(df_labeled)

    print("\nSplit sizes:")
    print("Train:", train_df.shape)
    print("Val:  ", val_df.shape)
    print("Test: ", test_df.shape)

    train_scaled, val_scaled, test_scaled, scaler = normalize_splits(
        train_df, val_df, test_df, feature_cols
    )

    # Sequence windows
    seq_len = 32
    X_train, y_train = make_sequences(train_scaled, feature_cols, seq_len=seq_len)
    X_val, y_val = make_sequences(val_scaled, feature_cols, seq_len=seq_len)
    X_test, y_test = make_sequences(test_scaled, feature_cols, seq_len=seq_len)

    print("\nSequence shapes:")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
    print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

    # Class balance
    print("\nClass balance:")
    print("Train positive ratio:", y_train.mean())
    print("Val positive ratio:  ", y_val.mean())
    print("Test positive ratio: ", y_test.mean())

if __name__ == "__main__":
    main()