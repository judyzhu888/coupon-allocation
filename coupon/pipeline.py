"""Daily batch job: train -> evaluate on randomized holdout -> allocate budget -> write assignments.

    python -m coupon.pipeline --date 2026-10-17 --budget 20000 --out artifacts [--data-dir data/]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import simulate
from .config import ARMS
from .evaluate import budget_policy, compare_policies, qini_auc
from .features import build_features, prepare_training
from .model import CouponUpliftModel
from .optimize import allocate, assignment_table


def load_data(data_dir: str | None, seed: int = 0, **sim_kwargs) -> dict:
    """Real data: {switchback,campaign,users}.parquet in data_dir. Otherwise simulate."""
    if data_dir is None:
        return simulate.simulate_all(seed=seed, **sim_kwargs)
    d = Path(data_dir)
    return {name: pd.read_parquet(d / f"{name}.parquet") for name in ("switchback", "campaign", "users")}


def split_holdout(switchback: pd.DataFrame, frac: float, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole switchback windows (not users) so evaluation respects the randomization unit."""
    unit = switchback["region"].astype(str) + "_" + switchback.get("window", 0).astype(str)
    units = unit.unique()
    held = set(rng.choice(units, size=max(1, int(len(units) * frac)), replace=False))
    mask = unit.isin(held).to_numpy()
    return switchback[~mask], switchback[mask]


def evaluate_holdout(model: CouponUpliftModel, holdout: pd.DataFrame, budget_per_user: float, rng) -> dict:
    h = holdout[~holdout["mixed_exposure"].astype(bool)] if "mixed_exposure" in holdout else holdout
    d = prepare_training(h)
    X, t = d["X"], d["t"]
    tau, cost = model.predict_effects(X), model.predict_cost(X)
    budget = budget_per_user * len(h)
    alloc = allocate(tau, cost, budget)
    policies = {
        "model": alloc.arms,
        "no_coupon": np.zeros(len(h), int),
        "random_at_budget": budget_policy(cost, budget, rng),
        "flat_$5_at_budget": budget_policy(cost, budget, rng, arm=int(np.where(ARMS == 5)[0][0])),
    }
    cost_obs = h["coupon_amount"].to_numpy() * h["redeemed"].to_numpy()
    table = compare_policies(policies, t, d["y"], cost_obs, d["propensity"],
                             mu_y=model.predict_mu(X), mu_cost=cost)
    qini = {}
    for k in range(1, len(ARMS)):
        m = (t == k) | (t == 0)
        qini[f"${ARMS[k]:g}"] = round(qini_auc(tau[m, k], t[m] == k, d["y"][m]), 5)
    report = {"n_holdout": int(len(h)), "budget_per_user": budget_per_user,
              "policies": table.round(4).to_dict(orient="index"), "qini": qini}
    if "sensitivity_latent" in h:  # simulator only: ground-truth value of each policy
        true_mu = np.column_stack([simulate.expected_profit(h, a) for a in ARMS])
        rows = np.arange(len(h))
        report["true_profit_per_user"] = {n: round(float(true_mu[rows, p].mean()), 4) for n, p in policies.items()}
    return report


def run(run_date: str, budget: float, out_dir: str, data_dir: str | None = None,
        holdout_frac: float = 0.3, seed: int = 0, **sim_kwargs) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    data = load_data(data_dir, seed=seed, **sim_kwargs)
    train_sb, holdout = split_holdout(data["switchback"], holdout_frac, rng)

    d = prepare_training(train_sb, data["campaign"])
    model = CouponUpliftModel(random_state=seed).fit(d["X"], d["t"], d["y"], d["redeemed"], d["propensity"])

    users = data["users"]
    report = evaluate_holdout(model, holdout, budget / len(users), rng)

    X = build_features(users)
    tau, cost = model.predict_effects(X), model.predict_cost(X)
    alloc = allocate(tau, cost, budget)
    table = assignment_table(users["user_id"], alloc, tau, cost, ARMS)

    out = Path(out_dir) / run_date
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "assignments.parquet", index=False)
    table.to_parquet(Path(out_dir) / "latest_assignments.parquet", index=False)

    summary = {
        "run_date": run_date, "budget": budget, "n_users": int(len(users)),
        "n_couponed": int((table["coupon_amount"] > 0).sum()),
        "amount_mix": table["coupon_amount"].value_counts().sort_index().to_dict(),
        "expected_spend": round(alloc.spend, 2), "expected_uplift": round(alloc.uplift, 2),
        "lambda": round(alloc.lam, 4), "n_train": int(len(d["y"])),
        "holdout_eval": report, "runtime_sec": round(time.time() - t0, 1),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(run(a.date, a.budget, a.out, a.data_dir, seed=a.seed), indent=2, default=float))


if __name__ == "__main__":
    main()
