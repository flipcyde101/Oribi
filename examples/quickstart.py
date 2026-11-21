"""Runnable Oribi example: profile a small order-items DataFrame."""

import pandas as pd

from oribi import inspect

orders = pd.DataFrame(
    {
        "order_id": [101, 101, 102, 102],
        "product_id": [7, 8, 7, 8],
        "revenue": [25.0, 40.0, 30.0, -5.0],
        "order_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
        ),
    }
)

context = inspect(orders)
print(context.summary())
print(context.to_context(max_chars=2_000))
