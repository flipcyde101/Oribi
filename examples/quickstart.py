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

# Candidate keys report observed uniqueness in this DataFrame only.
# A verified_unique=True flag means no collisions or null components were
# observed here; domain validation is still required for a business key.
print("\nCandidate keys (observed in this DataFrame, not validated business keys):")
for candidate in context.candidate_keys:
    columns = " + ".join(candidate.columns)
    print(f"- {columns} ({candidate.kind}, verified_unique={candidate.verified_unique})")

# Grain is a separate heuristic inference selected from observed candidates.
if context.grain and context.grain.status == "estimated":
    grain_columns = " + ".join(context.grain.columns)
    print(
        f"Inferred grain: {grain_columns} "
        f"(status={context.grain.status}, confidence={context.grain.confidence:.2f})"
    )
else:
    status = context.grain.status if context.grain else "none"
    print(f"Inferred grain: undetermined (status={status})")

print()
print(context.to_context(max_chars=2_000))
