# Coupon allocation (causal uplift + budget knapsack)

```
pip install -r requirements.txt
python -m coupon.pipeline --date 2026-10-17 --budget 20000 --out artifacts        # simulated data
python -m coupon.pipeline --date 2026-10-17 --budget 20000 --data-dir data/        # real data
COUPON_ASSIGNMENTS=artifacts/latest_assignments.parquet uvicorn coupon.serve:app  # GET /coupon/{user_id}
python -m pytest -q
```

Real data: `data/{switchback,campaign,users}.parquet` with columns in `coupon/config.py`
(`switchback` also needs `region`, `window`, `propensity`, `mixed_exposure`).

| Module | Role |
|---|---|
| `simulate.py` | Synthetic data with closed-form true effects |
| `features.py` | Schema checks, feature whitelist, training frame |
| `model.py` | Multi-arm DR-learner (cross-fit) + redemption-cost model |
| `optimize.py` | Lagrangian multiple-choice knapsack under budget |
| `evaluate.py` | DR off-policy value, budget-matched baselines, Qini |
| `pipeline.py` | Daily job; holdout = whole switchback windows |
| `serve.py` | FastAPI lookup of precomputed assignments |
