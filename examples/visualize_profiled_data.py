"""Profile a small sales DataFrame, then visualize additive revenue by region.

Install Matplotlib for this optional example with ``python -m pip install matplotlib``.
"""

import matplotlib.pyplot as plt
import pandas as pd

from oribi import inspect

orders = pd.DataFrame(
    {
        "order_id": [101, 101, 102, 103, 103, 104],
        "region": ["North", "North", "South", "North", "West", "South"],
        "revenue": [35.0, 20.0, 48.0, 18.0, -4.0, 12.0],
    }
)

# Inspect the data first so the chart is read with row grain and measure context.
context = inspect(orders)
print(context.summary())
print(context.to_context(max_chars=2_000))

# Revenue is additive in this example; the negative value represents a refund.
revenue_by_region = (
    orders.groupby("region")["revenue"]
    .sum()
    .sort_values()
)

ax = revenue_by_region.plot(kind="barh", color="#4C78A8", figsize=(8, 4.5))
ax.axvline(0, color="#555555", linewidth=0.8)
ax.set_title("Net revenue by region")
ax.set_xlabel("Revenue (USD)")
ax.set_ylabel("Region")
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)
ax.figure.tight_layout()
plt.show()
