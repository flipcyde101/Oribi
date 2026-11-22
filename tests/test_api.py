import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from oribi import DatasetContext, ProfileOptions, StatisticsContext, inspect
from oribi.exceptions import InvalidInputError


def test_statistics_context_preserves_positional_date_range():
    date_range = "2026-01-01 to 2026-01-05"
    statistics = StatisticsContext(
        0, 0.0, 2, 1.0, None, None, None, None, None, {}, None, None, None, (), date_range
    )
    assert statistics.date_range == date_range
    assert statistics.top_frequencies == ()


def test_inspect_returns_context_and_is_deterministic(sales_frame):
    a = inspect(sales_frame)
    b = inspect(sales_frame)
    assert a.to_dict() == b.to_dict()
    assert a.profile.row_count == 4
    assert a.profile.column_count == 5
    assert "Dataset:" in a.summary()


def test_options_control_quantiles(sales_frame):
    context = inspect(sales_frame, options=ProfileOptions(include_quantiles=False))
    assert context.columns[2].statistics.quantiles == {}


def test_max_exact_rows_includes_quantiles_at_limit_and_omits_them_above():
    def quantiles(row_count):
        frame = pd.DataFrame({"value": [float(index) for index in range(row_count)]})
        return inspect(frame, options=ProfileOptions(max_exact_rows=3)).columns[0].statistics.quantiles

    assert quantiles(2)
    assert quantiles(3)
    assert quantiles(4) == {}


def test_rejects_invalid_inputs():
    with pytest.raises(InvalidInputError):
        inspect([1, 2, 3])
    frame = pd.DataFrame([[1, 2]], columns=["x", "x"])
    with pytest.raises(InvalidInputError):
        inspect(frame)


@pytest.mark.parametrize("name, value", [
    ("top_k", 1.5), ("top_k", np.float64(2.0)), ("top_k", True), ("top_k", np.bool_(True)), ("top_k", "2"),
    ("max_exact_rows", 1.5), ("max_key_columns", False), ("max_key_columns", np.bool_(False)),
    ("max_relationships", 2.5), ("max_relationship_columns", "4"),
    ("max_gap_examples", True), ("max_gap_warnings", 1.5),
    ("top_k", -1), ("top_k", np.int64(-1)),
    ("max_exact_rows", 0), ("max_exact_rows", np.uint8(0)),
    ("max_key_columns", 1), ("max_key_columns", np.int32(1)),
    ("max_relationships", -1), ("max_relationships", np.int8(-1)),
    ("max_relationship_columns", -1), ("max_relationship_columns", np.int64(-1)),
    ("max_gap_examples", -1), ("max_gap_examples", np.int16(-1)),
    ("max_gap_warnings", 0), ("max_gap_warnings", np.int64(0)),
])
def test_invalid_integer_options_raise_at_configuration_time(name, value):
    with pytest.raises(ValueError, match=name):
        ProfileOptions(**{name: value})


def test_valid_option_boundaries_and_defaults():
    options = ProfileOptions(top_k=0, max_exact_rows=1, max_key_columns=2,
                             max_relationships=0, max_relationship_columns=0,
                             max_gap_examples=0, max_gap_warnings=1)
    assert options.top_k == 0
    assert ProfileOptions().max_gap_warnings == 10
    assert ProfileOptions(top_k=pd.Series([2]).iloc[0]).top_k == 2


def test_numpy_integral_options_normalize_to_python_int_and_stay_frozen():
    int64_max = np.int64(np.iinfo("int64").max)
    uint64_max = np.uint64(np.iinfo("uint64").max)
    options = ProfileOptions(
        top_k=int64_max,
        max_exact_rows=uint64_max,
        max_key_columns=np.int8(8),
        max_relationships=np.int32(40),
        max_relationship_columns=np.uint8(12),
        max_gap_examples=int64_max,
        max_gap_warnings=np.uint16(5),
    )
    expected = ProfileOptions(
        top_k=int(int64_max),
        max_exact_rows=int(uint64_max),
        max_key_columns=8,
        max_relationships=40,
        max_relationship_columns=12,
        max_gap_examples=int(int64_max),
        max_gap_warnings=5,
    )

    assert options == expected
    for field_name in (
        "top_k",
        "max_exact_rows",
        "max_key_columns",
        "max_relationships",
        "max_relationship_columns",
        "max_gap_examples",
        "max_gap_warnings",
    ):
        assert type(getattr(options, field_name)) is int

    updated = replace(options, top_k=np.int16(3))
    assert type(updated.top_k) is int
    assert updated.top_k == 3

    with pytest.raises(FrozenInstanceError):
        options.top_k = 5  # type: ignore[misc]


@pytest.mark.parametrize("value", [1, "false", None])
def test_include_quantiles_requires_boolean(value):
    with pytest.raises(ValueError, match="include_quantiles"):
        ProfileOptions(include_quantiles=value)


@pytest.mark.parametrize("values", [[1 + 2j, 3 + 4j], [1 + 2j, None]])
def test_complex_column_raises_public_input_error(values):
    frame = pd.DataFrame({"signal": pd.Series(values, dtype="complex128")})
    with pytest.raises(InvalidInputError, match="signal"):
        inspect(frame)


def test_valid_numeric_column_still_profiles():
    column = inspect(pd.DataFrame({"signal": [1.0, 2.0, 3.0]})).columns[0]
    assert column.statistics.quantiles["0.5"] == 2.0
@pytest.mark.parametrize("infinite", [float("inf"), float("-inf")])
def test_numeric_quantiles_with_infinity_are_finite(infinite):
    frame = pd.DataFrame({"value": [1.0, 2.0, infinite, 4.0]})
    stats = inspect(frame).columns[0].statistics
    assert all(q is None or (pd.notna(q) and math.isfinite(q)) for q in stats.quantiles.values())


@pytest.mark.parametrize("invalid_options", [False, 0, {}])
def test_inspect_rejects_falsy_invalid_options(invalid_options):
    frame = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(InvalidInputError, match="ProfileOptions"):
        inspect(frame, options=invalid_options)


def test_inspect_none_and_omitted_options_match_defaults():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    assert inspect(frame).to_dict() == inspect(frame, options=None).to_dict()


def test_categorical_context_complete_vs_truncated():
    frame = pd.DataFrame({
        "status": pd.Categorical(["open", "closed", "open", "pending", "open"]),
        "region": ["north", "south", "east", "north", "south"],
        "amount": [10.5, 20.0, 30.25, 40.0, 50.0],
        "user_id": [1, 2, 3, 4, 5]
    })
    context = inspect(frame, options=ProfileOptions(top_k=2))
    by_name = {col.name: col for col in context.columns}

    status_cats = by_name["status"].categories
    assert status_cats is not None
    assert status_cats.is_complete is False
    assert status_cats.values == (("open", 3), ("closed", 1))

    region_cats = by_name["region"].categories
    assert region_cats is not None
    assert region_cats.is_complete is False
    assert len(region_cats.values) == 2

    full_context = inspect(frame, options=ProfileOptions(top_k=10))
    full_by_name = {col.name: col for col in full_context.columns}
    assert full_by_name["status"].categories is not None
    assert full_by_name["status"].categories.is_complete is True
    assert dict(full_by_name["status"].categories.values) == {"open": 3, "closed": 1, "pending": 1}

    assert full_by_name["amount"].categories is None
    assert full_by_name["user_id"].categories is None


def test_categorical_context_not_populated_when_values_unknown():
    frame = pd.DataFrame({
        "empty_cat": pd.Series([None] * 200, dtype="category"),
        "skipped_int_dim": [i % 15 for i in range(200)],
    })
    context = inspect(frame, options=ProfileOptions(top_k=2))
    by_name = {col.name: col for col in context.columns}

    assert by_name["empty_cat"].categories is None
    assert by_name["skipped_int_dim"].semantic.role == "dimension"
    assert by_name["skipped_int_dim"].statistics.top_values == ()
    assert by_name["skipped_int_dim"].categories is None


def test_top_k_zero_disables_top_value_sampling_and_preserves_other_stats():
    frame = pd.DataFrame({
        "status": pd.Categorical(["open", "closed", "open", "pending", None]),
        "region": ["north", "south", "east", "north", "south"],
        "active": [True, False, True, True, None],
        "amount": [-5.0, 0.0, 10.5, 20.0, None],
        "tier": [1, 2, 1, 2, 1],
    })
    default = inspect(frame)
    disabled = inspect(frame, options=ProfileOptions(top_k=0))
    limited = inspect(frame, options=ProfileOptions(top_k=2))

    default_by_name = {col.name: col for col in default.columns}
    disabled_by_name = {col.name: col for col in disabled.columns}
    limited_by_name = {col.name: col for col in limited.columns}

    for name, default_col in default_by_name.items():
        disabled_col = disabled_by_name[name]
        assert disabled_col.statistics.top_values == ()
        assert disabled_col.statistics.top_frequencies == ()
        assert disabled_col.categories is None
        assert disabled_col.statistics.distinct_count == default_col.statistics.distinct_count
        assert disabled_col.statistics.distinct_rate == default_col.statistics.distinct_rate
        assert disabled_col.statistics.missing_count == default_col.statistics.missing_count
        assert disabled_col.statistics.missing_rate == default_col.statistics.missing_rate

    for numeric_name in ("amount", "tier"):
        default_stats = default_by_name[numeric_name].statistics
        disabled_stats = disabled_by_name[numeric_name].statistics
        assert disabled_stats.minimum == default_stats.minimum
        assert disabled_stats.maximum == default_stats.maximum
        assert disabled_stats.mean == default_stats.mean
        assert disabled_stats.median == default_stats.median
        assert disabled_stats.standard_deviation == default_stats.standard_deviation
        assert disabled_stats.quantiles == default_stats.quantiles
        assert disabled_stats.skewness == default_stats.skewness
        assert disabled_stats.zero_count == default_stats.zero_count
        assert disabled_stats.negative_count == default_stats.negative_count

    limited_status = limited_by_name["status"]
    assert limited_status.statistics.top_values == (("open", 2), ("closed", 1))
    assert limited_status.statistics.top_frequencies == (2, 1)
    assert limited_status.categories is not None
    assert limited_status.categories.is_complete is False

    limited_region = limited_by_name["region"]
    assert len(limited_region.statistics.top_values) == 2
    assert len(limited_region.statistics.top_frequencies) == 2
    assert limited_region.categories is not None


def test_get_column_returns_exact_column_and_preserves_order_and_exports(sales_frame):
    context = inspect(sales_frame)
    expected_names = tuple(col.name for col in context.columns)
    expected_dict = context.to_dict()

    revenue = context.get_column("revenue")
    assert revenue is context.columns[2]
    assert context.get_column("order_id") is context.columns[0]
    assert context.get_column("region") is context.columns[4]
    assert tuple(col.name for col in context.columns) == expected_names
    assert context.to_dict() == expected_dict


@pytest.mark.parametrize("missing_name", ["missing", "Revenue", "REVENUE", " revenue", "revenue ", ""])
def test_get_column_raises_key_error_for_absent_or_wrong_case_label(sales_frame, missing_name):
    context = inspect(sales_frame)
    with pytest.raises(KeyError) as exc_info:
        context.get_column(missing_name)
    assert exc_info.value.args == (missing_name,)


def test_get_column_handles_special_column_names():
    frame = pd.DataFrame({
        "": [1, 2],
        "gross revenue ($)": [10.0, 20.0],
        "café": ["a", "b"],
        "line\nbreak": [True, False],
    })
    context = inspect(frame)
    for expected in context.columns:
        assert context.get_column(expected.name) is expected


def test_get_column_returns_first_match_for_duplicate_names_and_handles_empty(sales_frame):
    base = inspect(sales_frame)
    first = base.columns[0]
    second = base.columns[1]
    later_duplicate = replace(first, dtype="duplicate_dtype")
    duplicate = DatasetContext(profile=base.profile, columns=(first, second, later_duplicate))
    assert duplicate.get_column(first.name) is first

    empty = DatasetContext(profile=base.profile, columns=())
    with pytest.raises(KeyError) as exc_info:
        empty.get_column("revenue")
    assert exc_info.value.args == ("revenue",)
