from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from dotenv import load_dotenv
import os

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.preprocessing import StandardScaler

load_dotenv()

# https://huggingface.co/datasets/mito0o852/OHLCV-1m
TICKER = "AAPL"
YEARS = [2022, 2023, 2024]
DATASET_REPO = "mito0o852/OHLCV-1m"
HF_TOKEN = os.environ.get("HF_TOKEN")

NY_TZ = "America/New_York"


def download_years_of_parquets(years: Sequence[int]) -> str:
    """
    Download parquet files for the requested years from the Hugging Face dataset.
    """
    allow_patterns = [f"data/ohlcv_{year}-*.parquet" for year in years]
    local_dir = snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        token=HF_TOKEN,
    )
    return local_dir


def load_one_ticker_from_years(local_dir: str, ticker: str, years: Sequence[int]) -> pd.DataFrame:
    """
    Load all monthly parquet files for the requested years and keep only one ticker.
    """
    data_dir = Path(local_dir) / "data"

    files = []
    for year in years:
        files.extend(sorted(data_dir.glob(f"ohlcv_{year}-*.parquet")))

    if not files:
        raise FileNotFoundError(f"No parquet files found for years={list(years)}")

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
        raise ValueError(f"No rows found for ticker={ticker} in years={list(years)}")

    df = pd.concat(dfs, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df


def resample_to_5min_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute OHLCV into 5-minute OHLCV, then keep only regular
    U.S. equity session bars (09:30 to 15:55 New York time).

    Resulting full trading days should have exactly 78 rows:
      09:30, 09:35, ..., 15:55
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NY_TZ)
    df = df.set_index("timestamp").sort_index()

    out = df.resample("5min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    out = out.dropna()
    out = out.between_time("09:30", "15:55")
    out = out[out.index.dayofweek < 5]

    out = out.reset_index()
    out["trade_date"] = out["timestamp"].dt.date
    return out

def add_stationary_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to log-relative features
    """
    df = df_5m.copy()

    df["raw_close"] = df["close"]

    prev_close = df["close"].shift(1)

    df["open"] = np.log(df["open"] / prev_close)
    df["high"] = np.log(df["high"] / prev_close)
    df["low"] = np.log(df["low"] / prev_close)
    df["close"] = np.log(df["close"] / prev_close)
    df["volume"] = np.log1p(df["volume"])

    df = df.dropna().reset_index(drop=True)
    return df

def build_day_blocks(
    df_5m: pd.DataFrame,
    feature_cols: List[str],
    block_size: int = 6,
    expected_blocks_per_day: int = 13,
) -> List[Dict]:
    """
    Build per-day 30-minute blocks from regular-session 5-minute bars.

    Each retained day must have exactly:
      expected_blocks_per_day * block_size rows
    For regular U.S. equity hours:
      13 * 6 = 78 rows

    Each block stores:
      - block_2d: shape (6, 5), time-major
      - close: close price of the last 5-minute row in the block
    """
    day_records: List[Dict] = []
    expected_rows = block_size * expected_blocks_per_day

    for trade_date, day_df in df_5m.groupby("trade_date", sort=True):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)

        if len(day_df) != expected_rows:
            continue

        blocks_2d = []
        closes = []

        for i in range(expected_blocks_per_day):
            block = day_df.iloc[i * block_size : (i + 1) * block_size]
            block_2d = block[feature_cols].to_numpy(dtype=np.float32)  # (6, 5), time-major
            blocks_2d.append(block_2d)
            closes.append(float(block["raw_close"].iloc[-1]))

        day_records.append(
            {
                "date": trade_date,
                "blocks_2d": blocks_2d,
                "closes": closes,
            }
        )

    if len(day_records) < 12:
        raise ValueError(
            f"Not enough full trading days after building 30-minute blocks. "
            f"Found only {len(day_records)} full days."
        )

    return day_records


def add_global_ternary_targets(day_records: List[Dict], lam: float = 0.2) -> List[Dict]:
    """
    Labels classified using Log-return with lam
    """
    global_closes = []
    global_owner_day_idx = []

    for day_idx, day in enumerate(day_records):
        for close in day["closes"]:
            global_closes.append(close)
            global_owner_day_idx.append(day_idx)

    global_closes = np.array(global_closes, dtype=np.float64)

    current_close = global_closes[:-1]
    next_close = global_closes[1:]

    log_ret = np.log(next_close / current_close)
    b_up = np.log((current_close + lam) / current_close)
    b_down = np.log((current_close - lam) / current_close)

    targets = np.ones(len(log_ret), dtype=np.int64)
    targets[log_ret > b_up] = 2
    targets[log_ret < b_down] = 0

    for day in day_records:
        day["targets"] = []

    for global_idx, y in enumerate(targets):
        owner_day_idx = global_owner_day_idx[global_idx]
        day_records[owner_day_idx]["targets"].append(int(y))

    filtered = []
    for day in day_records:
        if len(day["targets"]) == len(day["blocks_2d"]):
            filtered.append(day)

    if len(filtered) < 12:
        raise ValueError(
            f"Not enough fully labeled trading days after target construction. "
            f"Found only {len(filtered)} labeled days."
        )

    return filtered


def split_days_chronologically(
    day_records: List[Dict],
    val_days_count: int = 10,
    test_days_count: int = 10,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Paper-style fixed-duration split.

    For a 3-year dataset:
      train = all days except final 4 trading weeks
      val   = next 2 trading weeks
      test  = final 2 trading weeks

    Approximation:
      1 trading week = 5 trading days
      2 trading weeks = 10 trading days
      4 trading weeks = 20 trading days
    """
    n_days = len(day_records)
    holdout_days = val_days_count + test_days_count

    if n_days <= holdout_days + 3:
        raise ValueError(
            f"Not enough days for fixed split. "
            f"Got {n_days}, need more than {holdout_days + 3}."
        )

    train_days = day_records[:-holdout_days]
    val_days = day_records[-holdout_days:-test_days_count]
    test_days = day_records[-test_days_count:]

    if min(len(train_days), len(val_days), len(test_days)) < 3:
        raise ValueError(
            f"Need at least 3 days in each split. "
            f"Got train={len(train_days)}, val={len(val_days)}, test={len(test_days)}."
        )

    return train_days, val_days, test_days


def fit_scaler_on_days(day_records: List[Dict]) -> StandardScaler:
    """
    Fit feature-wise scaler on all 5-minute rows from training days only.
    """
    rows = []
    for day in day_records:
        for block in day["blocks_2d"]:
            rows.append(block)

    stacked = np.concatenate(rows, axis=0)  # (num_rows, 5)

    scaler = StandardScaler()
    scaler.fit(stacked)
    return scaler


def flatten_block_feature_major(block_2d: np.ndarray) -> np.ndarray:
    """
    Convert block from time-major (6, 5) to feature-major flattened vector (30,).

    Paper-style frame:
      rows = features
      cols = 6 consecutive 5-minute steps
    """
    if block_2d.shape != (6, 5):
        raise ValueError(f"Expected block shape (6, 5), got {block_2d.shape}")

    block_feature_major = block_2d.T  # (5, 6)
    return block_feature_major.reshape(-1).astype(np.float32)


def transform_days(
    day_records: List[Dict],
    scaler: StandardScaler,
) -> List[Dict]:
    """
    Scale each 5-minute row inside each block and flatten each day to shape (13, 30).
    """
    transformed = []

    for day in day_records:
        scaled_blocks_2d = []
        blocks_flat = []

        for block in day["blocks_2d"]:
            scaled_block = scaler.transform(block).astype(np.float32)
            scaled_blocks_2d.append(scaled_block)
            blocks_flat.append(flatten_block_feature_major(scaled_block))

        transformed.append(
            {
                "date": day["date"],
                "blocks_2d": scaled_blocks_2d,
                "blocks_flat": np.stack(blocks_flat).astype(np.float32),  # (13, 30)
                "targets": np.array(day["targets"], dtype=np.int64),      # (13,)
            }
        )

    return transformed


def build_pairs_within_split(
    day_records: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build consecutive-day Seq2Seq samples within a split.

    We exclude the final day of the split from pair construction so that the
    decoder day's final block label does not depend on the next split.
    """
    if len(day_records) < 3:
        raise ValueError("Need at least 3 days in a split to build safe consecutive-day pairs.")

    usable_days = day_records[:-1]

    X_enc, X_dec, y_dec = [], [], []

    for i in range(len(usable_days) - 1):
        prev_day = usable_days[i]
        curr_day = usable_days[i + 1]

        X_enc.append(prev_day["blocks_flat"])
        X_dec.append(curr_day["blocks_flat"])
        y_dec.append(curr_day["targets"])

    X_enc = np.stack(X_enc).astype(np.float32)
    X_dec = np.stack(X_dec).astype(np.float32)
    y_dec = np.stack(y_dec).astype(np.int64)

    return X_enc, X_dec, y_dec


def build_datasets(
    ticker: str = TICKER,
    years: Sequence[int] = YEARS,
    lam: float = 0.2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Build day-paired Seq2Seq datasets.

    Output shapes:
      X_enc_*: (N, 13, 30)
      X_dec_*: (N, 13, 30)
      y_dec_*: (N, 13)
    """
    years = list(years)
    feature_cols = ["open", "high", "low", "close", "volume"]

    local_dir = download_years_of_parquets(years)
    df_1m = load_one_ticker_from_years(local_dir, ticker, years)
    df_5m = resample_to_5min_regular_session(df_1m)
    df_5m = add_stationary_features(df_5m)

    day_records = build_day_blocks(
        df_5m=df_5m,
        feature_cols=feature_cols,
        block_size=6,
        expected_blocks_per_day=13,
    )
    day_records = add_global_ternary_targets(day_records, lam=lam)

    train_days, val_days, test_days = split_days_chronologically(
        day_records=day_records,
        val_days_count=40,
        test_days_count=40,
    )

    scaler = fit_scaler_on_days(train_days)

    train_days = transform_days(train_days, scaler)
    val_days = transform_days(val_days, scaler)
    test_days = transform_days(test_days, scaler)

    X_enc_train, X_dec_train, y_dec_train = build_pairs_within_split(train_days)
    X_enc_val, X_dec_val, y_dec_val = build_pairs_within_split(val_days)
    X_enc_test, X_dec_test, y_dec_test = build_pairs_within_split(test_days)

    return {
        "X_enc_train": X_enc_train,
        "X_dec_train": X_dec_train,
        "y_dec_train": y_dec_train,
        "X_enc_val": X_enc_val,
        "X_dec_val": X_dec_val,
        "y_dec_val": y_dec_val,
        "X_enc_test": X_enc_test,
        "X_dec_test": X_dec_test,
        "y_dec_test": y_dec_test,
        "input_dim": 30,
        "blocks_per_day": 13,
        "feature_cols": feature_cols,
        "years": years,
    }


def _print_split_class_balance(name: str, y_seq: np.ndarray) -> None:
    flat = y_seq.reshape(-1)
    vals, counts = np.unique(flat, return_counts=True)
    ratios = {int(v): float(c / len(flat)) for v, c in zip(vals, counts)}
    print(name, ratios)


def main():
    lam = 0.2

    data = build_datasets(
        ticker=TICKER,
        years=YEARS,
        lam=lam,
    )

    print("Years:", data["years"])
    print("Input dim:", data["input_dim"])
    print("Blocks per day:", data["blocks_per_day"])

    print(
        "X_enc_train:", data["X_enc_train"].shape,
        "X_dec_train:", data["X_dec_train"].shape,
        "y_dec_train:", data["y_dec_train"].shape,
    )
    print(
        "X_enc_val:  ", data["X_enc_val"].shape,
        "X_dec_val:  ", data["X_dec_val"].shape,
        "y_dec_val:  ", data["y_dec_val"].shape,
    )
    print(
        "X_enc_test: ", data["X_enc_test"].shape,
        "X_dec_test: ", data["X_dec_test"].shape,
        "y_dec_test: ", data["y_dec_test"].shape,
    )

    print("\nDecoder-sequence class balance:")
    _print_split_class_balance("Train", data["y_dec_train"])
    _print_split_class_balance("Val", data["y_dec_val"])
    _print_split_class_balance("Test", data["y_dec_test"])


if __name__ == "__main__":
    main()