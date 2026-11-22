import json

import pandas as pd
import pytest

from oribi import MetricDefinition, inspect


def _metric(context, name):
    return next(metric for metric in context.metrics if metric.name == name)


def test_orders_dataset_infers_common_analytical_metrics():
    frame = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c4"],
            "revenue": [25.0, 40.0, 30.0, 50.0, 15.0, 60.0],
            "order_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
        }
    )
    context = inspect(frame)
    by_name = {metric.name: metric for metric in context.metrics}

    assert {
        "order_count",
        "customer_count",
        "total_revenue",
        "average_order_value",
        "revenue_per_customer",
        "orders_per_customer",
    } <= set(by_name)

    order_count = by_name["order_count"]
    assert isinstance(order_count, MetricDefinition)
    assert order_count.kind == "entity_count"
    assert order_count.sql == "COUNT(DISTINCT order_id)"
    assert order_count.columns == ("order_id",)
    assert 0.5 < order_count.confidence <= 0.98
    assert order_count.evidence

    aov = by_name["average_order_value"]
    assert aov.kind == "measure_per_entity"
    assert aov.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"
    assert aov.columns == ("revenue", "order_id")
    assert 0.5 < aov.confidence <= 0.98
    assert any(e.code == "additive_measure_and_entity" for e in aov.evidence)

    rpc = by_name["revenue_per_customer"]
    assert rpc.kind == "measure_per_entity"
    assert rpc.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert rpc.columns == ("revenue", "customer_id")
    assert any(e.code == "repeated_entity_observations" for e in rpc.evidence)

    opc = by_name["orders_per_customer"]
    assert opc.kind == "entity_ratio"
    assert opc.sql == "1.0 * COUNT(DISTINCT order_id) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert opc.columns == ("order_id", "customer_id")
    assert any(e.code == "one_to_many_entity_hierarchy" for e in opc.evidence)


def test_order_items_grain_uses_distinct_order_count_for_average_order_value():
    frame = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 102, 103, 103],
            "product_id": [1, 2, 1, 3, 2, 3],
            "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            "revenue": [20.0, 30.0, 15.0, 25.0, 40.0, 10.0],
            "quantity": [1, 2, 3, 4, 5, 2],
        }
    )
    context = inspect(frame)
    aov = _metric(context, "average_order_value")
    assert aov.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"
    assert any(e.code == "repeated_entity_observations" for e in aov.evidence)

    rpu = _metric(context, "revenue_per_unit")
    assert rpu.kind == "measure_ratio"
    assert rpu.sql == "SUM(revenue) / NULLIF(SUM(quantity), 0)"
    assert rpu.columns == ("revenue", "quantity")


def test_integer_measure_division_avoids_integer_truncation_in_sql():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "revenue": pd.Series([10, 20, 35, 80], dtype="int64"),
            "profit": pd.Series([2, 5, 10, 25], dtype="int64"),
        }
    )
    context = inspect(frame)
    aov = _metric(context, "average_order_value")
    assert aov.sql == "1.0 * SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"

    margin = _metric(context, "profit_margin")
    assert margin.kind == "measure_ratio"
    assert margin.sql == "1.0 * SUM(profit) / NULLIF(SUM(revenue), 0)"
    assert margin.columns == ("profit", "revenue")


@pytest.mark.parametrize(
    "name", ["conversion_rate", "price_per_unit", "profit_ratio", "gross_margin (%)"]
)
def test_non_additive_measures_do_not_generate_sum_based_metrics(name):
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            name: [0.15, 0.20, 0.25, 0.40],
        }
    )
    context = inspect(frame)
    assert [m.name for m in context.metrics] == ["order_count"]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"order_id": pd.Series(dtype="int64"), "revenue": pd.Series(dtype="float64")}),
        pd.DataFrame({"order_id": [None, None, None], "revenue": [None, None, None]}),
        pd.DataFrame({"order_id": [1], "revenue": [25.0]}),
        pd.DataFrame({"order_id": ["same"] * 6, "revenue": [10.0] * 6}),
    ],
)
def test_empty_null_single_row_and_constant_frames_emit_no_metrics(frame):
    assert inspect(frame).metrics == ()


def test_conservative_guardrails_skip_low_diversity_ids_dimensions_and_row_keys():
    frame = pd.DataFrame(
        {
            "row_id": list(range(40)),
            "status_id": [1, 2] * 20,
            "region": ["east", "west"] * 20,
            "revenue": [10.0 + float(i) for i in range(40)],
        }
    )
    context = inspect(frame)
    names = {m.name for m in context.metrics}
    assert "row_count" in names
    assert "total_revenue" in names
    assert "status_count" not in names
    assert "revenue_per_row" not in names
    assert "revenue_per_region" not in names
    assert "revenue_per_status" not in names


def test_arbitrary_unique_key_without_name_hint_skips_speculative_per_entity_metric():
    frame = pd.DataFrame(
        {
            "K": [f"k{i}" for i in range(20)],
            "M": [float(i % 7) + 1.0 for i in range(20)],
        }
    )
    context = inspect(frame)
    names = {m.name for m in context.metrics}
    assert "m_per_k" not in names


def test_metric_caveats_reflect_measure_warnings_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 4],
            "customer_id": ["c1", "c1", "c2", "c3", "c3"],
            "revenue": [10.0, -5.0, 20.0, 1000.0, 1000.0],
        }
    )
    context = inspect(frame)
    order_count = _metric(context, "order_count")
    assert not any("duplicate rows" in caveat.lower() for caveat in order_count.caveats)

    aov = _metric(context, "average_order_value")
    combined_caveats = " ".join(aov.caveats).lower()
    assert "negative values" in combined_caveats
    assert "duplicate rows" in combined_caveats


def test_camelcase_and_special_column_names_quote_sql_and_render_safely():
    frame = pd.DataFrame(
        {
            "orderId": [1, 2, 3, 4],
            "customerId": ["c1", "c1", "c2", "c3"],
            "gross revenue ($)": [10.0, 20.0, 30.0, 40.0],
            "totalSales": [12.0, 24.0, 36.0, 48.0],
        }
    )
    context = inspect(frame)
    by_name = {m.name: m for m in context.metrics}
    assert "order_count" in by_name
    assert "customer_count" in by_name
    assert "total_sales" in by_name
    assert "total_total_sales" not in by_name
    assert by_name["total_gross_revenue"].sql == 'SUM("gross revenue ($)")'
    assert (
        by_name["average_order_value"].sql
        == 'SUM("gross revenue ($)") / NULLIF(COUNT(DISTINCT orderId), 0)'
    )


def test_metrics_serialize_privately_and_render_in_markdown_and_compact_context():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "patient_id": ["p1", "p1", "p2", "p3"],
            "diagnosis": pd.Categorical(["HIV", "cancer", "HIV", "cancer"]),
            "charge": [100.0, 250.0, 150.0, 400.0],
        }
    )
    context = inspect(frame)
    payload = context.to_dict()
    assert payload["schema_version"] == "2"
    assert isinstance(payload["metrics"], list)
    assert any(item["name"] == "order_count" for item in payload["metrics"])

    encoded = context.to_json()
    assert json.loads(encoded) == payload
    assert "HIV" not in encoded
    assert "cancer" not in encoded

    markdown = context.to_markdown()
    assert "## Analytical metrics" in markdown
    assert "COUNT(DISTINCT order_id)" in markdown
    assert "SUM(charge) / NULLIF(COUNT(DISTINCT order_id), 0)" in markdown
    assert "HIV" not in markdown

    compact = context.to_context()
    assert "METRICS" in compact
    assert compact.index("CARDINALITY") < compact.index("METRICS") < compact.index("COLUMNS")
    assert "average_order_value" in compact
    assert "HIV" not in compact

    tight = context.to_context(max_chars=250)
    assert len(tight) <= 250
    assert tight.endswith("[Context truncated.]")


def test_metrics_are_deterministic_across_column_order_and_bounded_on_wide_frames():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c4"],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    reordered = frame[["revenue", "customer_id", "order_id"]]
    assert inspect(frame).metrics == inspect(reordered).metrics

    wide_data = {
        "order_id": list(range(30)),
        "customer_id": [f"c{i // 2}" for i in range(30)],
        "revenue": [float(i + 10) for i in range(30)],
        **{f"entity_{i}_id": [f"e{j}_{i}" for j in range(30)] for i in range(12)},
        **{f"amount_{i}": [float(j + i + 1) for j in range(30)] for i in range(12)},
    }
    wide_context = inspect(pd.DataFrame(wide_data))
    assert len(wide_context.metrics) <= 20
    assert any(m.name == "average_order_value" for m in wide_context.metrics)
