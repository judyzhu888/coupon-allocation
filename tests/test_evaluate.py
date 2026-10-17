import numpy as np

from coupon.evaluate import budget_policy, compare_policies, dr_policy_value, qini_auc


def _rct(rng, n=40000, k=3):
    """Randomized data with known arm means [1, 2, 0.5]."""
    means = np.array([1.0, 2.0, 0.5])
    t = rng.integers(0, k, n)
    y = means[t] + rng.normal(0, 1, n)
    return t, y, np.full(n, 1 / k), means


def test_ipw_unbiased_for_constant_policies(rng):
    t, y, e, means = _rct(rng)
    for k, m in enumerate(means):
        assert abs(dr_policy_value(np.full(len(t), k), t, y, e) - m) < 0.08


def test_dr_with_correct_mu_is_exact_and_robust_to_wrong_mu(rng):
    t, y, e, means = _rct(rng)
    pol = np.full(len(t), 1)
    good = np.tile(means, (len(t), 1))
    bad = np.zeros_like(good)
    assert abs(dr_policy_value(pol, t, y, e, good) - 2.0) < 0.03
    assert abs(dr_policy_value(pol, t, y, e, bad) - 2.0) < 0.1  # still consistent via IPW correction


def test_budget_policy_respects_budget(rng):
    cost = np.column_stack([np.zeros(1000), np.full(1000, 2.0), np.full(1000, 4.0)])
    pol = budget_policy(cost, 100.0, rng, arm=1)
    assert set(np.unique(pol)) <= {0, 1}
    assert cost[np.arange(1000), pol].sum() <= 100.0
    assert (pol == 1).sum() == 50
    rand = budget_policy(cost, 100.0, rng)
    assert cost[np.arange(1000), rand].sum() <= 100.0


def test_compare_policies_table(rng):
    t, y, e, _ = _rct(rng, n=5000)
    cost = (t > 0) * 1.0
    tbl = compare_policies({"a": np.zeros(5000, int), "b": np.ones(5000, int)}, t, y, cost, e)
    assert list(tbl.index) == ["a", "b"]
    assert tbl.loc["a", "spend_per_user"] == 0 and tbl.loc["a", "share_couponed"] == 0
    assert tbl.loc["b", "share_couponed"] == 1


def test_qini_sign(rng):
    n = 20000
    tau = rng.normal(0, 1, n)
    treated = rng.random(n) < 0.5
    y = treated * tau + rng.normal(0, 0.5, n)
    assert qini_auc(tau, treated, y) > 0
    assert qini_auc(-tau, treated, y) < 0
    assert abs(qini_auc(rng.normal(size=n), treated, y)) < abs(qini_auc(tau, treated, y)) / 3
