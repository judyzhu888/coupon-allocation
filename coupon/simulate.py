"""Synthetic data with known ground-truth effects.

Structural model (per user, 14d window):
    lambda_k  = lambda_0 * (1 + 2.0*s*g(k)) + 0.5*s*g(k),   g(k) = k / (k + 4)
    orders    ~ Poisson(lambda_k)
    redeemed  = 1[orders > 0 and k > 0]
    profit    = orders * (MARGIN*aov - VAR_COST) - k * redeemed
so E[profit | x, k] = cpo * lambda_k - k * (1 - exp(-lambda_k)) is available in closed form.
`s` (coupon sensitivity) is latent; `past_redeem_rate` is a noisy proxy of it.
"""
import numpy as np
import pandas as pd

from .config import ARMS

MARGIN = 0.25
VAR_COST = 2.0


def make_users(n: int, rng: np.random.Generator, n_regions: int = 20, id_offset: int = 0) -> pd.DataFrame:
    recency = rng.exponential(15, n).clip(0, 90)
    orders_90d = rng.poisson(3.0 * np.exp(-recency / 30) + 0.3)
    aov = rng.lognormal(np.log(30), 0.4, n)
    tenure = rng.uniform(30, 1500, n)
    s = 1 / (1 + np.exp(-(1.0 - 0.6 * orders_90d + 0.03 * recency + rng.normal(0, 0.5, n))))
    return pd.DataFrame({
        "user_id": np.arange(id_offset, id_offset + n),
        "region": rng.integers(0, n_regions, n),
        "recency_days": recency,
        "orders_90d": orders_90d,
        "aov": aov,
        "tenure_days": tenure,
        "past_redeem_rate": (s + rng.normal(0, 0.1, n)).clip(0, 1),
        "sensitivity_latent": s,
    })


def _rate(users: pd.DataFrame, amount) -> np.ndarray:
    lam0 = 0.05 + users["orders_90d"].to_numpy() * 14 / 90 * np.exp(-users["recency_days"].to_numpy() / 60)
    s = users["sensitivity_latent"].to_numpy()
    g = np.asarray(amount, dtype=float) / (np.asarray(amount, dtype=float) + 4)
    return lam0 * (1 + 2.0 * s * g) + 0.5 * s * g


def expected_profit(users: pd.DataFrame, amount) -> np.ndarray:
    """Ground-truth E[14d net profit | x, do(amount)]; `amount` is a scalar or per-user array."""
    lam = _rate(users, amount)
    cpo = MARGIN * users["aov"].to_numpy() - VAR_COST
    return cpo * lam - np.asarray(amount, dtype=float) * (1 - np.exp(-lam))


def true_effects(users: pd.DataFrame) -> np.ndarray:
    """(n, K) matrix of true incremental profit vs no coupon."""
    base = expected_profit(users, 0.0)
    return np.column_stack([expected_profit(users, a) - base for a in ARMS])


def true_costs(users: pd.DataFrame) -> np.ndarray:
    """(n, K) matrix of true expected redeemed coupon cost."""
    return np.column_stack([a * (1 - np.exp(-_rate(users, a))) for a in ARMS])


def _realize(users: pd.DataFrame, amount: np.ndarray, lam: np.ndarray, rng) -> pd.DataFrame:
    orders = rng.poisson(lam)
    redeemed = ((orders > 0) & (amount > 0)).astype(int)
    cpo = MARGIN * users["aov"].to_numpy() - VAR_COST
    out = users.copy()
    out["coupon_amount"] = amount
    out["orders_14d"] = orders
    out["redeemed"] = redeemed
    out["net_profit_14d"] = orders * cpo - amount * redeemed
    return out


def simulate_switchback(n: int, rng, n_regions: int = 20, n_windows: int = 14,
                        mixed_rate: float = 0.15, id_offset: int = 0) -> pd.DataFrame:
    """Each (region, daily window) is randomized to an arm; user takes the arm of first exposure.
    A `mixed_rate` share of users also saw a second arm (carryover contaminates their outcome)."""
    users = make_users(n, rng, n_regions, id_offset)
    schedule = rng.integers(0, len(ARMS), size=(n_regions, n_windows))
    window = rng.integers(0, n_windows, n)
    arm_idx = schedule[users["region"].to_numpy(), window]
    amount = ARMS[arm_idx]
    mixed = rng.random(n) < mixed_rate
    other = ARMS[rng.integers(0, len(ARMS), n)]
    lam = np.where(mixed, 0.5 * (_rate(users, amount) + _rate(users, other)), _rate(users, amount))
    out = _realize(users, amount, lam, rng)
    out["window"] = window
    out["propensity"] = 1.0 / len(ARMS)
    out["mixed_exposure"] = mixed
    out["source"] = "switchback"
    return out


def simulate_campaign(n: int, rng, id_offset: int = 0) -> pd.DataFrame:
    """Targeted campaign: marketers gave bigger coupons to frequent users (confounded, no logged propensity)."""
    users = make_users(n, rng, id_offset=id_offset)
    z = (users["orders_90d"].to_numpy() - 2) / 2
    logits = np.column_stack([-0.5 * z, 0.2 * z, 0.6 * z, 1.0 * z])
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    arm_idx = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
    amount = ARMS[arm_idx]
    out = _realize(users, amount, _rate(users, amount), rng)
    out["propensity"] = np.nan
    out["mixed_exposure"] = False
    out["source"] = "campaign"
    return out


def simulate_all(n_switchback: int = 30000, n_campaign: int = 20000, n_score: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "switchback": simulate_switchback(n_switchback, rng, id_offset=0),
        "campaign": simulate_campaign(n_campaign, rng, id_offset=1_000_000),
        "users": make_users(n_score, rng, id_offset=2_000_000),
    }
