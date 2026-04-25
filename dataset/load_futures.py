from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()

# https://huggingface.co/datasets/ResearchRL/diffquant-data
DATASET_REPO = "ResearchRL/diffquant-data"
HF_TOKEN = os.environ.get("HF_TOKEN")

# 5x6 inputs
BLOCK_SIZE = 6

# 24h * 60 / (5 * BLOCK_SIZE) = 48 blocks per day
BLOCKS_PER_DAY = 48

def download_futures_npz() -> str:
    return hf_hub_download(
        repo_id=DATASET_REPO,
        filename="btcusdt_1min_2021_2025.npz",
        repo_type="dataset",
        token=HF_TOKEN
    )


def load_btcusdt_1m(
    years: Sequence[int],
    ticker: str,
) -> pd.DataFrame:
    """
    Load BTCUSDT 1-minute perpetual futures.

    Returns:
      timestamp, open, high, low, close, volume, ticker
    """
    path = download_futures_npz()
    data = np.load(path, allow_pickle=True)

    bars = data["bars"]
    timestamps = data["timestamps"]
    columns = list(data["columns"])

    df = pd.DataFrame(bars, columns=columns)
    df["timestamp"] = pd.to_datetime(timestamps, unit="ms", utc=True)
    df["ticker"] = ticker

    df = df[["timestamp", "open", "high", "low", "close", "volume", "ticker"]]
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    years = set(years)
    df = df[df["timestamp"].dt.year.isin(years)].reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No rows found for years={list(years)}")

    return df


def resample_to_5min_crypto_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 1-minute BTCUSDT futures bars into 5-minute OHLCV bars.

    Unlike the stock loader:
      - no NYSE market-hours filter
      - no weekday filter
      - full UTC crypto days are retained

    Full retained days should have exactly 288 five-minute rows.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
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

    out = out.dropna().reset_index()
    out["trade_date"] = out["timestamp"].dt.date

    return out


def add_stationary_features(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw OHLCV to stationary-ish log-relative features.

    raw_close is preserved only for label construction.
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
    block_size: int = BLOCK_SIZE,
    expected_blocks_per_day: int = BLOCKS_PER_DAY,
) -> List[Dict]:
    """
    Build per-day 30-minute blocks from full 24h crypto days.

    Each retained day must have:
      48 blocks/day * 6 rows/block = 288 five-minute rows

    Each block has shape:
      (6, 5)

    After flattening:
      (5, 6) -> 30 features
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

            block_2d = block[feature_cols].to_numpy(dtype=np.float32)
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
            f"Not enough full crypto days after block construction. "
            f"Found only {len(day_records)} full days."
        )

    return day_records


def add_global_ternary_targets(
    day_records: List[Dict],
    lam: float = 0.001,
) -> List[Dict]:
    """
    Add ternary labels using log-return threshold.

    Important:
      For futures/crypto, lam is NOT a dollar threshold.

    lam=0.001 means:
      up   if next block close is > roughly +0.10%
      down if next block close is < roughly -0.10%
      flat otherwise

    Classes:
      0 = down
      1 = flat
      2 = up
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

    targets = np.ones(len(log_ret), dtype=np.int64)
    targets[log_ret > lam] = 2
    targets[log_ret < -lam] = 0

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
            f"Not enough fully labeled crypto days after target construction. "
            f"Found only {len(filtered)} labeled days."
        )

    return filtered


def split_days_chronologically(
    day_records: List[Dict],
    val_days_count: int = 100,
    test_days_count: int = 100,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Chronological fixed holdout split.

    Default:
      final 100 days = test
      previous 100 days = validation
      all earlier days = train
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
    Fit feature-wise scaler on training days only.
    """
    rows = []

    for day in day_records:
        for block in day["blocks_2d"]:
            rows.append(block)

    stacked = np.concatenate(rows, axis=0)

    scaler = StandardScaler()
    scaler.fit(stacked)

    return scaler


def flatten_block_feature_major(block_2d: np.ndarray) -> np.ndarray:
    """
    Convert block from time-major to feature-major flattened vector.

    Input:
      (6, 5)

    Output:
      (30,)
    """
    if block_2d.shape != (6, 5):
        raise ValueError(f"Expected block shape (6, 5), got {block_2d.shape}")

    block_feature_major = block_2d.T
    return block_feature_major.reshape(-1).astype(np.float32)


def transform_days(
    day_records: List[Dict],
    scaler: StandardScaler,
) -> List[Dict]:
    """
    Scale each 5-minute row inside each block and flatten each block.

    Output per day:
      blocks_flat: (48, 30)
      targets:     (48,)
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
                "blocks_flat": np.stack(blocks_flat).astype(np.float32),
                "targets": np.array(day["targets"], dtype=np.int64),
            }
        )

    return transformed


def build_pairs_within_split(
    day_records: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build consecutive-day Seq2Seq samples within a split.

    X_enc = previous day
    X_dec = current day
    y_dec = current day labels
    """
    if len(day_records) < 3:
        raise ValueError("Need at least 3 days in a split to build consecutive-day pairs.")

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
    ticker: str,
    years: Sequence[int],
    lam: float = 0.001,
) -> Dict[str, np.ndarray]:
    """
    Build futures-style day-paired Seq2Seq datasets.

    Output shapes:
      X_enc_*: (N, 48, 30)
      X_dec_*: (N, 48, 30)
      y_dec_*: (N, 48)

    input_dim remains 30 because each 30-minute block is still:
      5 OHLCV features x 6 five-minute bars
    """
    years = list(years)
    feature_cols = ["open", "high", "low", "close", "volume"]

    df_1m = load_btcusdt_1m(years=years, ticker=ticker)
    df_5m = resample_to_5min_crypto_day(df_1m)
    df_5m = add_stationary_features(df_5m)

    day_records = build_day_blocks(
        df_5m=df_5m,
        feature_cols=feature_cols,
        block_size=BLOCK_SIZE,
        expected_blocks_per_day=BLOCKS_PER_DAY,
    )

    day_records = add_global_ternary_targets(
        day_records=day_records,
        lam=lam,
    )

    train_days, val_days, test_days = split_days_chronologically(
        day_records=day_records,
        val_days_count=100,
        test_days_count=100,
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
        "blocks_per_day": BLOCKS_PER_DAY,
        "feature_cols": feature_cols,
        "years": years,
    }