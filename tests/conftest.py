import numpy as np
import pytest

from coupon import simulate
from coupon.features import prepare_training
from coupon.model import CouponUpliftModel


@pytest.fixture(scope="session")
def sim():
    return simulate.simulate_all(n_switchback=12000, n_campaign=6000, n_score=3000, seed=1)


@pytest.fixture(scope="session")
def train(sim):
    return prepare_training(sim["switchback"], sim["campaign"])


@pytest.fixture(scope="session")
def fitted(train):
    return CouponUpliftModel(random_state=0).fit(train["X"], train["t"], train["y"], train["redeemed"],
                                                 train["propensity"])


@pytest.fixture
def rng():
    return np.random.default_rng(0)
