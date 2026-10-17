import json

import numpy as np
import pandas as pd
import pytest

from coupon import pipeline, simulate
from coupon.config import ARMS


@pytest.fixture(scope="module")
def run_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    summary = pipeline.run("2026-10-17", budget=3000, out_dir=str(out),
                           n_switchback=12000, n_campaign=6000, n_score=5000, seed=2)
    return out, summary


def test_split_holdout_by_window_unit(rng):
    sb = simulate.simulate_switchback(3000, rng)
    tr, ho = pipeline.split_holdout(sb, 0.3, rng)
    assert len(tr) + len(ho) == len(sb)
    unit = lambda d: set(zip(d["region"], d["window"]))
    assert not (unit(tr) & unit(ho))


def test_writes_artifacts(run_out):
    out, _ = run_out
    df = pd.read_parquet(out / "2026-10-17" / "assignments.parquet")
    assert df["user_id"].is_unique and len(df) == 5000
    assert set(df["coupon_amount"]) <= set(ARMS)
    assert (out / "latest_assignments.parquet").exists()
    assert json.loads((out / "2026-10-17" / "report.json").read_text())["run_date"] == "2026-10-17"


def test_budget_respected_and_no_negative_uplift(run_out):
    out, s = run_out
    df = pd.read_parquet(out / "latest_assignments.parquet")
    assert s["expected_spend"] <= 3000 + 1e-6
    assert (df.loc[df["coupon_amount"] > 0, "expected_uplift"] > 0).all()


def test_model_beats_baselines_on_truth(run_out):
    truth = run_out[1]["holdout_eval"]["true_profit_per_user"]
    assert truth["model"] > truth["no_coupon"]
    assert truth["model"] > truth["random_at_budget"]
    assert truth["model"] > truth["flat_$5_at_budget"]


def test_dr_estimate_tracks_truth(run_out):
    ev = run_out[1]["holdout_eval"]
    for name, true_v in ev["true_profit_per_user"].items():
        assert abs(ev["policies"][name]["profit_per_user"] - true_v) < 0.5


def test_load_data_from_dir(tmp_path):
    d = simulate.simulate_all(500, 500, 500, seed=4)
    for k, v in d.items():
        v.to_parquet(tmp_path / f"{k}.parquet")
    loaded = pipeline.load_data(str(tmp_path))
    assert set(loaded) == {"switchback", "campaign", "users"}
    assert len(loaded["users"]) == 500
