"""JSON-compatible projection of the canonical model."""

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from typing import Any

from ..models.evidence import CategoricalContext


def _normalize(value: Any, include_values: bool = False, *, evidence_details: bool = False) -> Any:
    if is_dataclass(value):
        result: dict[str, Any] = {}
        for field in fields(value):
            if not include_values and (
                field.name == "top_values"
                or (isinstance(value, CategoricalContext) and field.name == "values")
            ):
                result[field.name] = []
            else:
                result[field.name] = _normalize(
                    getattr(value, field.name), include_values,
                    evidence_details=(type(value).__name__ == "Evidence" and field.name == "details"),
                )
        return result
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if evidence_details and not include_values and isinstance(item, str):
                continue
            if evidence_details and not include_values and isinstance(item, (tuple, list)) and any(isinstance(v, str) for v in item):
                continue
            result[str(key)] = _normalize(item, include_values, evidence_details=evidence_details)
        return result
    if isinstance(value, (tuple, list)):
        return [_normalize(v, include_values, evidence_details=evidence_details) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _normalize(value.item(), include_values, evidence_details=evidence_details)
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def to_dict(context: Any, *, include_values: bool = False) -> dict[str, Any]:
    """Return an export-safe projection by default; opt in to local sample values."""
    result = _normalize(context, include_values)
    if not include_values:
        for column in result.get("columns", []):
            statistics = column["statistics"]
            for field in ("minimum", "maximum", "date_range"):
                statistics[field] = None
            if result["profile"]["row_count"] - statistics["missing_count"] < 3:
                for field in ("mean", "median", "standard_deviation", "skewness"):
                    statistics[field] = None
                statistics["quantiles"] = {}
    return result


def to_json(context: Any, *, indent: int | None = 2, include_values: bool = False) -> str:
    """Serialize a context using the standard JSON encoder."""
    return json.dumps(to_dict(context, include_values=include_values), indent=indent, sort_keys=True, ensure_ascii=False)
