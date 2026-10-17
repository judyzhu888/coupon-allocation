"""Multi-arm DR-learner for incremental 14d net profit + redemption model for expected cost."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .config import ARMS


def _regressor(seed, min_leaf=100):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=min_leaf, random_state=seed)


def _classifier(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=100, random_state=seed)


@dataclass
class CouponUpliftModel:
    arms: np.ndarray = field(default_factory=lambda: ARMS.copy())
    n_folds: int = 2
    prop_clip: float = 0.02
    random_state: int = 0

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    # ---------- nuisance ----------
    def _fit_outcome(self, X, t, y):
        models = []
        for k in range(self.n_arms):
            if not (t == k).any():
                raise ValueError(f"no training rows for arm {self.arms[k]}")
            models.append(_regressor(self.random_state).fit(X[t == k], y[t == k]))
        return models

    def _propensity(self, X_tr, t_tr, X_te, known_te):
        """Estimated arm probabilities on X_te, overridden by logged propensities where known."""
        e = np.full((len(X_te), self.n_arms), np.nan)
        need = np.isnan(known_te)
        if need.any():
            clf = _classifier(self.random_state).fit(X_tr, t_tr)
            proba = np.zeros((need.sum(), self.n_arms))
            proba[:, clf.classes_] = clf.predict_proba(X_te[need])
            e[need] = proba
        # Randomized rows: only P(received arm) enters the DR score, so broadcast the logged value.
        e[~need] = known_te[~need, None]
        return np.clip(e, self.prop_clip, 1.0)

    # ---------- fit ----------
    def fit(self, X: pd.DataFrame, t, y, redeemed, propensity=None) -> "CouponUpliftModel":
        X = np.asarray(X, float)
        t, y, redeemed = np.asarray(t), np.asarray(y, float), np.asarray(redeemed)
        known = np.full(len(t), np.nan) if propensity is None else np.asarray(propensity, float)

        mu = np.zeros((len(t), self.n_arms))
        e = np.zeros((len(t), self.n_arms))
        folds = KFold(self.n_folds, shuffle=True, random_state=self.random_state)
        for tr, te in folds.split(X):
            mu[te] = np.column_stack([m.predict(X[te]) for m in self._fit_outcome(X[tr], t[tr], y[tr])])
            e[te] = self._propensity(X[tr], t[tr], X[te], known[te])

        # DR pseudo-outcome for each arm vs control, then regress on X.
        rows = np.arange(len(t))
        resid = (y - mu[rows, t]) / e[rows, t]
        self.effect_models_ = [None]
        for k in range(1, self.n_arms):
            phi = mu[:, k] - mu[:, 0] + (t == k) * resid - (t == 0) * resid
            self.effect_models_.append(_regressor(self.random_state, min_leaf=300).fit(X, phi))

        self.outcome_models_ = self._fit_outcome(X, t, y)

        treated = t > 0
        Xa = np.column_stack([X[treated], self.arms[t[treated]]])
        self.redeem_model_ = _classifier(self.random_state).fit(Xa, redeemed[treated])
        return self

    # ---------- predict ----------
    def predict_effects(self, X) -> np.ndarray:
        """(n, K) incremental net profit vs no coupon; column 0 is 0."""
        X = np.asarray(X, float)
        return np.column_stack([np.zeros(len(X))] + [m.predict(X) for m in self.effect_models_[1:]])

    def predict_cost(self, X) -> np.ndarray:
        """(n, K) expected redeemed coupon cost = amount * P(redeem | x, amount)."""
        X = np.asarray(X, float)
        cols = [np.zeros(len(X))]
        for a in self.arms[1:]:
            p = self.redeem_model_.predict_proba(np.column_stack([X, np.full(len(X), a)]))[:, 1]
            cols.append(a * p)
        return np.column_stack(cols)

    def predict_mu(self, X) -> np.ndarray:
        """(n, K) predicted E[profit | x, arm]; used as the direct-method term in DR evaluation."""
        X = np.asarray(X, float)
        return np.column_stack([m.predict(X) for m in self.outcome_models_])
