"""Cold-chain temperature evidence for a single transfer batch.

A reading of ``v`` °C is a measurement of the true temperature with an error of
±0.5 °C. Between two readings the true temperature is assumed to vary linearly.
Under those assumptions:

* the **lower envelope** (every reading at ``v - 0.5``) is, at every instant, no
  warmer than any temperature curve consistent with the readings;
* the **upper envelope** (every reading at ``v + 0.5``) is, at every instant, no
  colder than any such curve.

"Time above 8 °C" is monotone in the temperature curve, so the longest
continuous excursion measured on the lower/upper envelope is a valid
lower/upper bound on the true longest excursion.

Evidence rules
--------------
* A pair of consecutive readings at most 10 minutes apart is evidence over the
  interval between them, with linear interpolation.
* Evidence is clipped to the batch interval ``[departed_at, arrived_at]``.
* A probe covers the start when its evidence reaches departure; it covers the
  end when its evidence reaches arrival. Complete coverage means there is no
  evidence gap at all inside the batch interval.
* Excursions are never joined across an evidence gap: the temperature there is
  unknown, so no continuous duration may be claimed across it.

Verdict (``evaluate``):

* ``anomalous`` when any probe's *lower* bound exceeds 15 minutes: a violation
  is already certain from the evidence we have;
* ``released`` only when both probes have complete coverage and both *upper*
  bounds are at most 15 minutes: no consistent temperature curve violates;
* ``insufficient_evidence`` otherwise. In particular, bounded uncertainty that
  could hide a violation is not a release, and neither is missing coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Sequence

TEMPERATURE_LIMIT_C = 8.0
MEASUREMENT_ERROR_C = 0.5
MAX_READING_GAP = timedelta(minutes=10)
MAX_EXCURSION = timedelta(minutes=15)

THRESHOLDS = {
    "temperature_limit_c": TEMPERATURE_LIMIT_C,
    "measurement_error_c": MEASUREMENT_ERROR_C,
    "max_reading_gap_minutes": MAX_READING_GAP.total_seconds() / 60.0,
    "max_excursion_minutes": MAX_EXCURSION.total_seconds() / 60.0,
}

#: Tolerance used only for interval merging (datetimes carry microseconds).
_EPSILON = timedelta(microseconds=1)
#: Tolerance on the 15-minute comparison. Interpolation is floating point, so
#: a mathematically exact 15-minute bound must not flip to "exceeds".
_LIMIT_TOLERANCE = timedelta(milliseconds=1)


class QualityFailure(Exception):
    """The evidence cannot be evaluated (malformed data or broken invariant)."""


@dataclass(frozen=True)
class Interval:
    start: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    @property
    def minutes(self) -> float:
        return self.duration.total_seconds() / 60.0


@dataclass(frozen=True)
class _Chunk:
    """One evidence interval, clipped to the batch window."""

    start: datetime
    end: datetime
    value_start: float
    value_end: float


def _parse(value: str | datetime) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _interpolate(
    t0: datetime, v0: float, t1: datetime, v1: float, at: datetime
) -> float:
    span = (t1 - t0).total_seconds()
    if span <= 0:
        return v0
    fraction = (at - t0).total_seconds() / span
    return v0 + (v1 - v0) * fraction


def _crossing(
    t0: datetime, v0: float, t1: datetime, v1: float, threshold: float
) -> datetime:
    """Time at which the segment crosses ``threshold`` (v0, v1 straddle it)."""
    if v1 == v0:
        return t0
    fraction = (threshold - v0) / (v1 - v0)
    fraction = min(1.0, max(0.0, fraction))
    return t0 + (t1 - t0) * fraction


def _chunks(
    readings: Sequence[tuple[datetime, float]],
    departed_at: datetime,
    arrived_at: datetime,
) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    for (t0, v0), (t1, v1) in zip(readings, readings[1:]):
        if t1 <= t0:
            raise QualityFailure("probe readings are not strictly ordered in time")
        if t1 - t0 > MAX_READING_GAP:
            continue
        start = max(t0, departed_at)
        end = min(t1, arrived_at)
        if start >= end:
            continue
        chunks.append(
            _Chunk(
                start=start,
                end=end,
                value_start=_interpolate(t0, v0, t1, v1, start),
                value_end=_interpolate(t0, v0, t1, v1, end),
            )
        )
    return chunks


def _merge_ranges(intervals: Iterable[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda item: (item.start, item.end)):
        if merged and interval.start <= merged[-1].end + _EPSILON:
            previous = merged[-1]
            if interval.end > previous.end:
                merged[-1] = Interval(previous.start, interval.end)
        else:
            merged.append(interval)
    return merged


def _gaps(
    chunks: Sequence[_Chunk], departed_at: datetime, arrived_at: datetime
) -> list[Interval]:
    gaps: list[Interval] = []
    cursor = departed_at
    for chunk in _merge_ranges(
        Interval(chunk.start, chunk.end) for chunk in chunks
    ):
        if chunk.start > cursor:
            gaps.append(Interval(cursor, chunk.start))
        if chunk.end > cursor:
            cursor = chunk.end
    if cursor < arrived_at:
        gaps.append(Interval(cursor, arrived_at))
    return gaps


def _runs(chunks: Sequence[_Chunk]) -> list[list[_Chunk]]:
    """Group evidence chunks that share a reading (one continuous curve)."""
    runs: list[list[_Chunk]] = []
    for chunk in chunks:
        if runs and chunk.start <= runs[-1][-1].end + _EPSILON:
            runs[-1].append(chunk)
        else:
            runs.append([chunk])
    return runs


def _envelope_excursions(
    run: Sequence[_Chunk], shift: float
) -> list[Interval]:
    """Maximal intervals where the shifted piecewise-linear curve is > 8 °C."""
    points: list[tuple[datetime, float]] = [
        (run[0].start, run[0].value_start + shift)
    ]
    for chunk in run:
        points.append((chunk.end, chunk.value_end + shift))

    intervals: list[Interval] = []
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if t1 <= t0:
            continue
        above_start = v0 > TEMPERATURE_LIMIT_C
        above_end = v1 > TEMPERATURE_LIMIT_C
        if above_start and above_end:
            start, end = t0, t1
        elif above_start:
            start, end = t0, _crossing(
                t0, v0, t1, v1, TEMPERATURE_LIMIT_C
            )
        elif above_end:
            start, end = (
                _crossing(t0, v0, t1, v1, TEMPERATURE_LIMIT_C),
                t1,
            )
        else:
            continue
        start = max(t0, min(start, t1))
        end = max(t0, min(end, t1))
        if end <= start:
            continue
        if intervals and start <= intervals[-1].end + _EPSILON:
            previous = intervals[-1]
            intervals[-1] = Interval(previous.start, max(previous.end, end))
        else:
            intervals.append(Interval(start, end))
    return intervals


def _longest(intervals: Sequence[Interval]) -> timedelta:
    return max((item.duration for item in intervals), default=timedelta(0))


def evaluate(
    *,
    batch: dict[str, Any],
    probes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate two probes against one batch's departure→arrival interval.

    Args:
        batch: ``batch_no``, ``version``, ``departed_at``, ``arrived_at``.
        probes: two entries with ``probe_id`` and ``readings`` (``at``,
            ``temperature_c``). Readings are sorted here; duplicate timestamps
            are rejected.

    Returns:
        A JSON-ready-by-the-caller dict with ``status``, ``releasable``,
        ``reason``, ``anomalous_probe_ids``, ``incomplete_probe_ids`` and one
        result per probe. Durations are in minutes; intervals carry datetimes.

    Raises:
        QualityFailure: malformed batch or probe evidence.
    """
    departed_at = _parse(batch["departed_at"])
    arrived_at = _parse(batch["arrived_at"])
    if departed_at >= arrived_at:
        raise QualityFailure("batch departure must precede arrival")

    if len(probes) != 2:
        raise QualityFailure("exactly two probes are required")
    probe_ids = [probe["probe_id"] for probe in probes]
    if len(set(probe_ids)) != len(probe_ids):
        raise QualityFailure("probe ids must be distinct")

    probe_results: list[dict[str, Any]] = []
    for probe in probes:
        raw = [
            (_parse(reading["at"]), float(reading["temperature_c"]))
            for reading in probe.get("readings", ())
        ]
        if not raw:
            raise QualityFailure(f"probe {probe['probe_id']!r} has no readings")
        raw.sort(key=lambda item: item[0])
        for (t0, _), (t1, _) in zip(raw, raw[1:]):
            if t1 == t0:
                raise QualityFailure(
                    f"probe {probe['probe_id']!r} has two readings at {t0.isoformat()}"
                )

        chunks = _chunks(raw, departed_at, arrived_at)
        gaps = _gaps(chunks, departed_at, arrived_at)
        covered_start = bool(chunks) and chunks[0].start <= departed_at
        covered_end = bool(chunks) and chunks[-1].end >= arrived_at
        complete = bool(chunks) and not gaps

        guaranteed: list[Interval] = []
        possible: list[Interval] = []
        for run in _runs(chunks):
            guaranteed.extend(
                _envelope_excursions(run, -MEASUREMENT_ERROR_C)
            )
            possible.extend(_envelope_excursions(run, +MEASUREMENT_ERROR_C))
        guaranteed = _merge_ranges(guaranteed)
        possible = _merge_ranges(possible)

        lower = _longest(guaranteed)
        upper = _longest(possible)
        evidenced = sum(
            (chunk.end - chunk.start for chunk in chunks), timedelta(0)
        )
        probe_results.append(
            {
                "probe_id": probe["probe_id"],
                "reading_count": len(raw),
                "first_reading_at": raw[0][0],
                "last_reading_at": raw[-1][0],
                "covers_start": covered_start,
                "covers_end": covered_end,
                "complete_coverage": complete,
                "evaluated_minutes": evidenced.total_seconds() / 60.0,
                "lower_bound_minutes": lower.total_seconds() / 60.0,
                "upper_bound_minutes": upper.total_seconds() / 60.0,
                "guaranteed_excursions": guaranteed,
                "possible_excursions": possible,
                "evidence_gaps": gaps,
                "_lower": lower,
                "_upper": upper,
            }
        )

    anomalous_ids = [
        item["probe_id"]
        for item in probe_results
        if item["_lower"] > MAX_EXCURSION + _LIMIT_TOLERANCE
    ]
    incomplete_ids = [
        item["probe_id"]
        for item in probe_results
        if not item["complete_coverage"]
    ]
    uncertain_ids = [
        item["probe_id"]
        for item in probe_results
        if item["_upper"] > MAX_EXCURSION + _LIMIT_TOLERANCE
    ]

    if anomalous_ids:
        status = "anomalous"
        releasable = False
        reason = (
            "probe(s) " + ", ".join(anomalous_ids) + " show a guaranteed"
            " excursion above 8 °C longer than 15 minutes"
        )
    elif not incomplete_ids and not uncertain_ids:
        status = "released"
        releasable = True
        reason = (
            "both probes cover the full batch interval and no temperature"
            " curve within measurement error exceeds 15 minutes above 8 °C"
        )
    else:
        status = "insufficient_evidence"
        releasable = False
        parts = []
        if incomplete_ids:
            parts.append(
                "incomplete evidence coverage for probe(s) "
                + ", ".join(incomplete_ids)
            )
        if uncertain_ids:
            parts.append(
                "the upper bound exceeds 15 minutes for probe(s) "
                + ", ".join(uncertain_ids)
            )
        reason = "; ".join(parts)

    for item in probe_results:
        item.pop("_lower")
        item.pop("_upper")

    return {
        "status": status,
        "releasable": releasable,
        "reason": reason,
        "departed_at": departed_at,
        "arrived_at": arrived_at,
        "thresholds": dict(THRESHOLDS),
        "anomalous_probe_ids": anomalous_ids,
        "incomplete_probe_ids": incomplete_ids,
        "uncertain_probe_ids": uncertain_ids,
        "probes": probe_results,
    }
