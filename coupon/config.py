"""Shared constants."""
import numpy as np

# Coupon amounts ($). Index 0 must be the no-coupon control arm.
ARMS = np.array([0.0, 3.0, 5.0, 10.0])

# Raw columns the system expects for every user snapshot.
USER_COLS = ["user_id", "recency_days", "orders_90d", "aov", "tenure_days", "past_redeem_rate"]

# Columns required on logged (training) rows in addition to USER_COLS.
LOG_COLS = ["coupon_amount", "net_profit_14d", "redeemed"]
