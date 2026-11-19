"""Conservative temporal cadence and gap observations."""

from collections import Counter
from dataclasses import replace

import pandas as pd

from ..models.evidence import Evidence, Warning

_MIN_UNIQUE_TIMESTAMPS = 4
_MIN_CADENCE_INTERVALS = 2
_MIN_CADENCE_SHARE = 0.6


def temporal_gap_warnings(series: pd.Series, column_name: str, *, max_examples: int = 3,
                          max_warnings: int = 10) -> tuple[Warning, ...]:
    """Report possible gaps only when a dominant fixed cadence is supported."""
    timestamps = pd.DatetimeIndex(series.dropna().drop_duplicates().sort_values())
    if len(timestamps) < _MIN_UNIQUE_TIMESTAMPS:
        return ()

    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    ranked = Counter(deltas).most_common()
    cadence, dominant_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == dominant_count:
        return ()
    if dominant_count < _MIN_CADENCE_INTERVALS or dominant_count / len(deltas) < _MIN_CADENCE_SHARE:
        return ()

    steps_by_interval: list[int] = []
    for delta in deltas:
        steps = delta / cadence
        if steps < 1 or steps != int(steps):
            return ()
        steps_by_interval.append(int(steps))
    if not any(steps > 1 for steps in steps_by_interval):
        return ()

    cadence_text = _format_timedelta(cadence)
    warnings: list[Warning] = []
    gap_count = 0
    for index, steps in enumerate(steps_by_interval):
        missing_count = steps - 1
        if not missing_count:
            continue
        gap_count += 1
        if gap_count > max_warnings:
            continue
        before, after = timestamps[index], timestamps[index + 1]
        examples = [before + cadence * step for step in range(1, min(steps, max_examples + 1))]
        gap_start = before + cadence
        gap_end = after - cadence
        evidence = Evidence(
            "temporal_gap",
            f"{missing_count} timestamp(s) may be absent relative to an inferred {cadence_text} cadence",
            {
                "cadence": cadence_text,
                "gap_start": gap_start.isoformat(),
                "gap_end": gap_end.isoformat(),
                "missing_intervals": missing_count,
                "expected_timestamps": [value.isoformat() for value in examples],
                "examples_omitted": missing_count - len(examples),
                "observed_before": before.isoformat(),
                "observed_after": after.isoformat(),
            },
        )
        warnings.append(Warning(
            "potential_temporal_gap", "low", "Potential temporal gap",
            f"{column_name} has a possible temporal gap relative to an inferred {cadence_text} cadence.",
            (evidence,),
            "The interval may reflect missing observations or a normal pause in data collection.",
            "Confirm the expected sampling schedule before treating the interval as missing data.",
        ))
    if gap_count > max_warnings and warnings:
        last = warnings[-1]
        evidence = last.evidence[0]
        omitted = gap_count - max_warnings
        warnings[-1] = replace(
            last,
            message=f"{last.message} {omitted} additional gap(s) omitted.",
            evidence=(replace(evidence, details={**evidence.details, "gaps_omitted": omitted}),),
        )
    return tuple(warnings)


def _format_timedelta(value: pd.Timedelta) -> str:
    """Format common fixed cadence durations consistently."""
    if value.microseconds or value.nanoseconds:
        return str(value)
    seconds = value.days * 86400 + value.seconds
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        if seconds % size == 0:
            amount = seconds // size
            return f"{amount} {unit}{'' if amount == 1 else 's'}"
    return str(value)
