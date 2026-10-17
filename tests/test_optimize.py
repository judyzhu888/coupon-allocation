import itertools

import numpy as np
import pytest

from coupon.config import ARMS
from coupon.optimize import allocate, assignment_table, choose_arms


def _instance(rng, n=200, k=4):
    cost = np.column_stack([np.zeros(n), rng.uniform(0.5, 3, (n, k - 1)).cumsum(1)])
    tau = np.column_stack([np.zeros(n), rng.normal(0.5, 1.5, (n, k - 1))])
    return tau, cost


def test_choose_arms_skips_negative_uplift():
    tau = np.array([[0, -1.0, -2.0], [0, 2.0, 1.0]])
    cost = np.array([[0, 1.0, 2.0], [0, 1.0, 2.0]])
    assert choose_arms(tau, cost, 0.0).tolist() == [0, 1]


def test_unconstrained_budget_picks_best_positive_arm(rng):
    tau, cost = _instance(rng)
    a = allocate(tau, cost, budget=1e9)
    assert a.lam == 0.0
    assert np.array_equal(a.arms, tau.argmax(1))


@pytest.mark.parametrize("budget", [0.0, 10.0, 50.0, 200.0])
def test_respects_budget(rng, budget):
    tau, cost = _instance(rng)
    a = allocate(tau, cost, budget)
    assert a.spend <= budget + 1e-9
    assert np.isclose(a.spend, cost[np.arange(len(tau)), a.arms].sum())


def test_zero_budget_gives_no_coupons(rng):
    tau, cost = _instance(rng)
    assert (allocate(tau, cost, 0.0).arms == 0).all()


def test_uplift_monotone_in_budget(rng):
    tau, cost = _instance(rng)
    uplifts = [allocate(tau, cost, b).uplift for b in (5, 20, 80, 320)]
    assert all(x <= y + 1e-9 for x, y in zip(uplifts, uplifts[1:]))


def test_near_optimal_vs_brute_force(rng):
    tau, cost = _instance(rng, n=6, k=3)
    budget = cost[:, 1].sum() * 0.6
    best = max(
        (tau[np.arange(6), c].sum() for c in itertools.product(range(3), repeat=6)
         if cost[np.arange(6), c].sum() <= budget),
    )
    a = allocate(tau, cost, budget)
    assert a.uplift >= 0.7 * best  # Lagrangian is near-optimal; exact at scale


def test_input_validation(rng):
    tau, cost = _instance(rng, n=5)
    with pytest.raises(ValueError):
        allocate(tau[:, 1:], cost[:, 1:], 10)
    with pytest.raises(ValueError):
        allocate(tau, cost[:, :2], 10)
    with pytest.raises(ValueError):
        allocate(tau, cost, -1)


def test_assignment_table(rng):
    tau, cost = _instance(rng, n=10)
    a = allocate(tau, cost, 5.0)
    df = assignment_table(np.arange(100, 110), a, tau, cost, ARMS)
    assert list(df.columns) == ["user_id", "coupon_amount", "expected_uplift", "expected_cost"]
    assert set(df["coupon_amount"]) <= set(ARMS)
    assert np.isclose(df["expected_cost"].sum(), a.spend)
