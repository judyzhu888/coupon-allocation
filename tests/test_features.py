import numpy as np
import pandas as pd
import pytest

from coupon import simulate
from coupon.features import FEATURE_COLS, arm_index, build_features, prepare_training, validate


def test_build_features_whitelist_excludes_latent_and_ids(rng):
    u = simulate.make_users(100, rng)
    X = build_features(u)
    assert list(X.columns) == FEATURE_COLS
    assert "sensitivity_latent" not in X and "user_id" not in X
    assert not X.isna().any().any()


def test_validate_missing_and_null():
    with pytest.raises(ValueError, match="missing"):
        validate(pd.DataFrame({"a": [1]}), ["a", "b"])
    with pytest.raises(ValueError, match="null"):
        validate(pd.DataFrame({"a": [1, None]}), ["a"])


def test_build_features_rejects_nonpositive_aov(rng):
    u = simulate.make_users(10, rng)
    u.loc[0, "aov"] = 0
    with pytest.raises(ValueError, match="aov"):
        build_features(u)


def test_arm_index_maps_and_rejects_unknown():
    assert arm_index(pd.Series([0.0, 3.0, 5.0, 10.0])).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        arm_index(pd.Series([4.0]))
    with pytest.raises(ValueError):
        arm_index(pd.Series([20.0]))


def test_prepare_training_drops_mixed_and_keeps_propensity(rng):
    sb = simulate.simulate_switchback(2000, rng)
    camp = simulate.simulate_campaign(1000, rng)
    d = prepare_training(sb, camp)
    n_clean = int((~sb["mixed_exposure"]).sum())
    assert len(d["y"]) == n_clean + 1000
    assert np.allclose(d["propensity"][:n_clean], 0.25)
    assert np.isnan(d["propensity"][n_clean:]).all()
    assert set(d["source"]) == {"switchback", "campaign"}
    assert len(prepare_training(sb, drop_mixed=False)["y"]) == 2000


def test_prepare_training_adds_missing_propensity(rng):
    camp = simulate.simulate_campaign(200, rng).drop(columns=["propensity"])
    assert np.isnan(prepare_training(camp)["propensity"]).all()
