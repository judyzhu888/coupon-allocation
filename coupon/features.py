"""Schema validation and feature construction."""
import numpy as np
import pandas as pd

from .config import ARMS, LOG_COLS, USER_COLS

FEATURE_COLS = ["recency_days", "log_orders_90d", "log_aov", "tenure_days", "past_redeem_rate", "log_spend_90d"]


def validate(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    nulls = [c for c in cols if df[c].isna().any()]
    if nulls:
        raise ValueError(f"null values in columns: {nulls}")


def build_features(users: pd.DataFrame) -> pd.DataFrame:
    """Model features from a user snapshot. Only whitelisted columns leave this function."""
    validate(users, USER_COLS)
    if (users["aov"] <= 0).any():
        raise ValueError("aov must be positive")
    return pd.DataFrame({
        "recency_days": users["recency_days"].astype(float),
        "log_orders_90d": np.log1p(users["orders_90d"].astype(float)),
        "log_aov": np.log(users["aov"].astype(float)),
        "tenure_days": users["tenure_days"].astype(float),
        "past_redeem_rate": users["past_redeem_rate"].astype(float),
        "log_spend_90d": np.log1p(users["orders_90d"] * users["aov"]),
    }, index=users.index)[FEATURE_COLS]


def arm_index(amount: pd.Series) -> np.ndarray:
    idx = np.searchsorted(ARMS, amount.to_numpy())
    idx = np.clip(idx, 0, len(ARMS) - 1)
    if not np.allclose(ARMS[idx], amount.to_numpy()):
        raise ValueError(f"coupon_amount values must be in {ARMS.tolist()}")
    return idx


def prepare_training(*logs: pd.DataFrame, drop_mixed: bool = True) -> dict:
    """Stack logged campaigns into model inputs.

    Returns dict with X, t (arm index), y (14d net profit), redeemed, propensity
    (known for randomized rows, NaN where it must be estimated), source.
    """
    frames = []
    for log in logs:
        validate(log, USER_COLS + LOG_COLS)
        log = log.copy()
        if "propensity" not in log:
            log["propensity"] = np.nan
        if drop_mixed and "mixed_exposure" in log:
            log = log[~log["mixed_exposure"].astype(bool)]
        frames.append(log)
    df = pd.concat(frames, ignore_index=True)
    return {
        "X": build_features(df),
        "t": arm_index(df["coupon_amount"]),
        "y": df["net_profit_14d"].to_numpy(float),
        "redeemed": df["redeemed"].to_numpy(int),
        "propensity": df["propensity"].to_numpy(float),
        "source": df["source"].to_numpy() if "source" in df else np.full(len(df), "unknown"),
    }
