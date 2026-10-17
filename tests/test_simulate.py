import numpy as np

from coupon import simulate
from coupon.config import ARMS


def test_make_users_shape_and_ranges(rng):
    u = simulate.make_users(500, rng)
    assert len(u) == 500 and u["user_id"].is_unique
    assert (u["aov"] > 0).all()
    assert u["past_redeem_rate"].between(0, 1).all()
    assert u["sensitivity_latent"].between(0, 1).all()


def test_true_effects_control_column_zero_and_heterogeneous(rng):
    u = simulate.make_users(2000, rng)
    tau = simulate.true_effects(u)
    assert tau.shape == (2000, len(ARMS))
    assert np.allclose(tau[:, 0], 0)
    # Some users profitable, some not -> targeting matters.
    assert (tau[:, 1] > 0).any() and (tau[:, 1] < 0).any()


def test_true_cost_bounded_by_face_value(rng):
    u = simulate.make_users(1000, rng)
    c = simulate.true_costs(u)
    assert np.allclose(c[:, 0], 0)
    assert ((c >= 0) & (c <= ARMS)).all()


def test_realized_profit_matches_expectation(rng):
    u = simulate.make_users(1, rng)
    big = u.loc[u.index.repeat(200_000)].reset_index(drop=True)
    out = simulate._realize(big, np.full(len(big), 5.0), simulate._rate(big, 5.0), rng)
    assert np.isclose(out["net_profit_14d"].mean(), simulate.expected_profit(u, 5.0)[0], rtol=0.03)


def test_switchback_randomized_and_flagged(rng):
    sb = simulate.simulate_switchback(8000, rng)
    shares = sb["coupon_amount"].value_counts(normalize=True)
    assert set(shares.index) == set(ARMS)
    assert (shares.between(0.15, 0.35)).all()
    assert np.allclose(sb["propensity"], 1 / len(ARMS))
    assert 0.1 < sb["mixed_exposure"].mean() < 0.2
    # Arm constant within each (region, window) unit.
    assert (sb.groupby(["region", "window"])["coupon_amount"].nunique() == 1).all()


def test_campaign_is_confounded(rng):
    camp = simulate.simulate_campaign(8000, rng)
    assert camp["propensity"].isna().all()
    big = camp.loc[camp["coupon_amount"] == 10, "orders_90d"].mean()
    none = camp.loc[camp["coupon_amount"] == 0, "orders_90d"].mean()
    assert big > none  # frequent users targeted with bigger coupons


def test_simulate_all_disjoint_ids():
    d = simulate.simulate_all(1000, 1000, 1000, seed=3)
    ids = [set(d[k]["user_id"]) for k in ("switchback", "campaign", "users")]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
