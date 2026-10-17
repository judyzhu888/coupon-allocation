"""Online lookup of precomputed daily assignments (O(1) dict lookup, well under 200ms).

    COUPON_ASSIGNMENTS=artifacts/latest_assignments.parquet uvicorn coupon.serve:app
"""
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI


def load_assignments(path: str | Path) -> dict[int, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {int(r.user_id): {"coupon_amount": float(r.coupon_amount), "expected_uplift": float(r.expected_uplift)}
            for r in df.itertuples(index=False)}


def create_app(path: str | Path) -> FastAPI:
    app = FastAPI(title="coupon-allocation")
    table = load_assignments(path)

    @app.get("/health")
    def health():
        return {"ready": bool(table), "n_users": len(table)}

    @app.get("/coupon/{user_id}")
    def coupon(user_id: int):
        hit = table.get(user_id)
        if hit is None:  # safe default: no coupon spend for unscored users
            return {"user_id": user_id, "coupon_amount": 0.0, "expected_uplift": 0.0, "reason": "not_scored"}
        return {"user_id": user_id, **hit, "reason": "model"}

    return app


app = create_app(os.environ.get("COUPON_ASSIGNMENTS", "artifacts/latest_assignments.parquet"))
