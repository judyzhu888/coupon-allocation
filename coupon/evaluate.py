"""Off-policy evaluation on randomized data, baseline policies, and Qini."""
import numpy as np
import pandas as pd


def dr_policy_value(policy, t, y, propensity, mu=None) -> float:
    """Per-user value of `policy` (arm index per user). DR if mu (n, K) given, else IPW.

    `propensity` is the probability of the arm actually received (t)."""
    policy, t, y, e = map(np.asarray, (policy, t, y, propensity))
    match = (policy == t).astype(float)
    rows = np.arange(len(t))
    if mu is None:
        return float(np.mean(match * y / e))
    mu = np.asarray(mu)
    return float(np.mean(mu[rows, policy] + match * (y - mu[rows, t]) / e))


def budget_policy(cost: np.ndarray, budget: float, rng, arm=None) -> np.ndarray:
    """Random users get `arm` (or a random non-zero arm) until expected spend reaches the budget."""
    n, k = cost.shape
    arms = np.full(n, arm) if arm is not None else rng.integers(1, k, n)
    order = rng.permutation(n)
    spend = np.cumsum(cost[order, arms[order]])
    policy = np.zeros(n, int)
    chosen = order[spend <= budget]
    policy[chosen] = arms[chosen]
    return policy


def compare_policies(policies: dict, t, y, cost_obs, propensity, mu_y=None, mu_cost=None) -> pd.DataFrame:
    rows = []
    for name, pol in policies.items():
        rows.append({
            "policy": name,
            "profit_per_user": dr_policy_value(pol, t, y, propensity, mu_y),
            "spend_per_user": dr_policy_value(pol, t, cost_obs, propensity, mu_cost),
            "share_couponed": float(np.mean(np.asarray(pol) > 0)),
        })
    return pd.DataFrame(rows).set_index("policy")


def qini_auc(score, treated, y) -> float:
    """Area between the Qini curve and the random-targeting line, per user (randomized data, binary arm).
    Positive => the score ranks high-uplift users first."""
    score, treated, y = np.asarray(score), np.asarray(treated).astype(bool), np.asarray(y, float)
    order = np.argsort(-score, kind="stable")
    tr, yy = treated[order], y[order]
    n_t, n_c = np.cumsum(tr), np.cumsum(~tr)
    y_t, y_c = np.cumsum(yy * tr), np.cumsum(yy * ~tr)
    qini = y_t - y_c * np.divide(n_t, n_c, out=np.zeros(len(n_t)), where=n_c > 0)
    random_line = qini[-1] * np.arange(1, len(y) + 1) / len(y)
    return float(np.mean(qini - random_line) / len(y))
