"""Budget-constrained arm allocation (multiple-choice knapsack via Lagrangian relaxation)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Allocation:
    arms: np.ndarray        # chosen arm index per user
    lam: float              # shadow price of $1 of budget
    spend: float            # expected coupon spend
    uplift: float           # expected total incremental profit


def choose_arms(tau: np.ndarray, cost: np.ndarray, lam: float) -> np.ndarray:
    # Arm 0 has tau=cost=0, so users with no positive adjusted score get no coupon (argmax ties -> first).
    return np.argmax(tau - lam * cost, axis=1)


def _total(mat: np.ndarray, arms: np.ndarray) -> float:
    return float(mat[np.arange(len(arms)), arms].sum())


def allocate(tau: np.ndarray, cost: np.ndarray, budget: float, n_iter: int = 60) -> Allocation:
    """Maximize sum tau[i, a_i] s.t. sum cost[i, a_i] <= budget. Always returns a feasible solution."""
    tau, cost = np.asarray(tau, float), np.asarray(cost, float)
    if tau.shape != cost.shape or tau.ndim != 2:
        raise ValueError("tau and cost must be (n, K) arrays of the same shape")
    if not (np.allclose(tau[:, 0], 0) and np.allclose(cost[:, 0], 0)):
        raise ValueError("column 0 must be the no-coupon arm with zero effect and cost")
    if budget < 0 or (cost < 0).any():
        raise ValueError("budget and costs must be non-negative")

    def result(lam):
        arms = choose_arms(tau, cost, lam)
        return Allocation(arms, lam, _total(cost, arms), _total(tau, arms))

    best = result(0.0)
    if best.spend <= budget:
        return best

    hi = 1.0
    while result(hi).spend > budget:
        hi *= 2
        if hi > 1e9:
            return Allocation(np.zeros(len(tau), int), np.inf, 0.0, 0.0)
    lo = 0.0
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        if result(mid).spend > budget:
            lo = mid
        else:
            hi = mid
    return result(hi)


def assignment_table(user_ids, alloc: Allocation, tau: np.ndarray, cost: np.ndarray,
                     arm_values: np.ndarray) -> pd.DataFrame:
    rows = np.arange(len(alloc.arms))
    return pd.DataFrame({
        "user_id": np.asarray(user_ids),
        "coupon_amount": arm_values[alloc.arms],
        "expected_uplift": tau[rows, alloc.arms],
        "expected_cost": cost[rows, alloc.arms],
    })
