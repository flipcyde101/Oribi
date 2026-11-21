"""Observed dataset and column statistics."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class StatisticsContext:
    missing_count: int
    missing_rate: float
    distinct_count: int
    distinct_rate: float
    minimum: Any = None
    maximum: Any = None
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = None
    quantiles: Mapping[str, float] = field(default_factory=dict)
    zero_count: int | None = None
    negative_count: int | None = None
    skewness: float | None = None
    top_values: tuple[tuple[Any, int], ...] = ()
    date_range: str | None = None
    top_frequencies: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", MappingProxyType(dict(self.quantiles)))


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_row_count: int | None
    duplicate_row_rate: float | None
    index: "IndexContext | None" = None


@dataclass(frozen=True)
class IndexContext:
    names: tuple[Hashable, ...]
    dtypes: tuple[str, ...]
    level_count: int
    unique_count: int
    duplicate_count: int
    missing_count: int
    is_unique: bool
    is_default: bool = False
