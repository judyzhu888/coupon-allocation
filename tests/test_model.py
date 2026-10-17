import numpy as np
import pytest

from coupon import simulate
from coupon.config import ARMS
from coupon.features import build_features
from coupon.model import CouponUpliftModel


def test_prediction_shapes_and_control_column(fitted, sim):
    X = build_features(sim["users"])
    tau, cost, mu = fitted.predict_effects(X), fitted.predict_cost(X), fitted.predict_mu(X)
    for m in (tau, cost, mu):
        assert m.shape == (len(X), len(ARMS))
    assert np.allclose(tau[:, 0], 0) and np.allclose(cost[:, 0], 0)


def test_cost_within_face_value(fitted, sim):
    cost = fitted.predict_cost(build_features(sim["users"]))
    assert ((cost >= 0) & (cost <= ARMS + 1e-9)).all()


def test_recovers_true_effect_ranking(fitted, sim):
    users = sim["users"]
    tau_hat = fitted.predict_effects(build_features(users))
    tau = simulate.true_effects(users)
    for k in range(1, len(ARMS)):
        assert np.corrcoef(tau_hat[:, k], tau[:, k])[0, 1] > 0.6


def test_effect_bias_small(fitted, sim):
    users = sim["users"]
    tau_hat = fitted.predict_effects(build_features(users))
    tau = simulate.true_effects(users)
    # Average treatment effect within $0.5/user despite confounded campaign rows.
    assert np.all(np.abs(tau_hat.mean(0) - tau.mean(0)) < 0.5)


def test_redemption_cost_calibrated(fitted, sim):
    users = sim["users"]
    cost_hat = fitted.predict_cost(build_features(users))
    assert np.allclose(cost_hat.mean(0), simulate.true_costs(users).mean(0), atol=0.3)


def test_missing_arm_raises(train):
    keep = train["t"] != 2
    with pytest.raises(ValueError, match="no training rows"):
        CouponUpliftModel().fit(train["X"][keep], train["t"][keep], train["y"][keep],
                                train["redeemed"][keep], train["propensity"][keep])


def test_fit_without_propensity_runs(train):
    n = 3000
    m = CouponUpliftModel(n_folds=2).fit(train["X"][:n], train["t"][:n], train["y"][:n], train["redeemed"][:n])
    assert m.predict_effects(train["X"][:5]).shape == (5, len(ARMS))
